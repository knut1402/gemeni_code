# src/market_data_processors/wirp_processor.py

import pandas as pd
import numpy as np
import datetime
import QuantLib as ql
from typing import Any, Dict, List, Optional, Union
from src.data_manager import b_con, data_loader  #
from src.date_utils import to_bbg_date_str, datetime_to_ql, ql_to_datetime, get_next_imm  #
from src.config import config  #
from src.financial_math_utils import flat_lst  #


class WirpProcessor:
    """
    Processes WIRP (World Interest Rate Probability) data for central bank meetings.

    Fetches OIS future rates, central bank base rates, and meeting dates from Bloomberg.
    Calculates implied rate steps and cumulative changes, and manages historical WIRP data.
    """

    def __init__(self):
        """
        Initializes the WirpProcessor.
        """
        pass  # No specific state to initialize for the processor itself

    def get_wirp_data(self,
                      currency_codes: Union[str, List[str]],  # Original `a`
                      valuation_dates: List[Union[str, datetime.date, ql.Date]]  # Original `d`
                      ) -> Dict[str, Any]:
        """
        Retrieves and processes WIRP data for specified currencies and valuation dates.

        Args:
            currency_codes (Union[str, List[str]]): Single currency code or list of currency codes (e.g., 'USD', 'GBP').
            valuation_dates (List[Union[str, datetime.date, ql.Date]]):
                List of valuation dates for which to fetch WIRP data.

        Returns:
            Dict[str, Any]: A dictionary where keys are currency codes and values are lists
                            of DataFrames, each containing WIRP steps and cumulative changes
                            for a given valuation date. Also includes historical central bank
                            base rates if available.
        """
        if isinstance(currency_codes, str):  #
            currency_codes = [currency_codes]  #

        all_wirp_data: Dict[str, List[pd.DataFrame]] = {ccy: [] for ccy in currency_codes}  #
        historical_cb_rates: Dict[str, List[Any]] = {ccy: [] for ccy in currency_codes}  # Original `x5`

        for ccy_code in currency_codes:  #
            ccy_config = config.get_currency_config(ccy_code)  #
            bbg_plot_tickers = ccy_config.get('bbgplot_tickers', [])  #
            cb_contrib_type = ccy_config.get('cb_contrib_type')  #
            cb_num_meets = ccy_config.get('cb_num_meets')  #
            base_ticker = ccy_config.get('base_ticker')  #
            fixing_index = ccy_config.get('fixing_index')  #

            if not bbg_plot_tickers or not cb_contrib_type or not cb_num_meets:
                print(f"Warning: Missing WIRP-related conventions for {ccy_code}. Skipping.")  #
                continue  #

            # Determine OIS future tickers based on conventions
            # NOTE: Bloomberg ticker construction logic is inferred. Please verify.
            # `bbg_plot_tickers` typically holds general curve tickers, the third element
            # is used as the base for STIR options in some plots (e.g. `bbgplot_tickers[2]`).
            # Here, assuming `bbg_plot_tickers[2]` is the base for OIS futures.
            #
            # The original code's `ticker+str(i)+' '+contrib+' Curncy'` is used.
            # `ticker` was `c.bbgplot_tickers[2]`. `contrib` was `c.contrib[0]`.
            # `c.contrib[0]` is not in the conventions.yaml. Let's assume it maps to `cb_contrib_type`.
            #
            ois_future_tickers_base = bbg_plot_tickers[2]  # Inferred base ticker for OIS futures

            # The original code had `if c.curncy == ql.USDCurrency():` and `elif c.curncy in [ql.EURCurrency(), ...]`
            # which changed the suffix from just `str(i)` to `str(i)+'A'`.
            # This logic should be externalized to conventions if possible.
            # For now, hardcoding the 'A' suffix for non-USD as per original:
            if ccy_code == 'USD':  #
                ois_future_tickers = [f"{ois_future_tickers_base}{i} {cb_contrib_type} Curncy" for i in
                                      range(1, cb_num_meets)]  #
            else:  #
                ois_future_tickers = [f"{ois_future_tickers_base}{i}A {cb_contrib_type} Curncy" for i in
                                      range(1, cb_num_meets)]  #

            for val_date in valuation_dates:  #
                val_date_ql = datetime_to_ql(val_date) if isinstance(val_date, datetime.date) else val_date  #
                bbg_val_date_str = to_bbg_date_str(val_date_ql, ql_date=1)  #

                # Fetch OIS future prices for this valuation date
                try:
                    futures_df = b_con.bdh(ois_future_tickers, 'PX_LAST', bbg_val_date_str, bbg_val_date_str,
                                           longdata=True)  #
                    futures_df = futures_df.pivot(index='date', columns='ticker', values='value').iloc[
                        0].reset_index()  #
                    futures_df.columns = ['Ticker', 'value']  #

                    # Sort by implied meeting number for consistent ordering
                    futures_df['meet_num'] = [int(re.search(r'\d+', t).group()) for t in
                                              futures_df['Ticker']]  # Extract number from ticker
                    futures_df = futures_df.sort_values('meet_num').reset_index(drop=True)  #

                    # Fetch Central Bank meeting dates
                    # Original `ECO_FUTURE_RELEASE_DATE_LIST`
                    # For historical dates, need to override start/end for list.
                    if val_date_ql == ql.Settings.instance().evaluationDate:  #
                        cb_meeting_dates_raw = b_con.bulkref(base_ticker, 'ECO_FUTURE_RELEASE_DATE_LIST')['value']  #
                    else:  #
                        cb_meeting_dates_raw = b_con.bulkref(base_ticker, 'ECO_FUTURE_RELEASE_DATE_LIST',  #
                                                             ovrds=[("START_DT", bbg_val_date_str),  #
                                                                    ("END_DT", to_bbg_date_str(
                                                                        val_date_ql + ql.Period(2, ql.Years),
                                                                        ql_date=1))]  #
                                                             )['value']  #

                    cb_meeting_dates = pd.Series(
                        [datetime.datetime.strptime(d, '%Y/%m/%d %H:%M:%S').date() for d in cb_meeting_dates_raw])  #

                    # Filter for future meetings relevant to the fetched futures
                    # Original `y` filtering logic: `y[y > (datetime.datetime.combine(d[i], datetime.datetime.min.time())+pd.DateOffset(days=-1))]`
                    relevant_meetings = cb_meeting_dates[cb_meeting_dates > val_date].reset_index(drop=True)  #

                    # Generate own future FOMC dates if not enough from Bloomberg
                    # Original code had `fut_gen_n = len(t_list) - np.sum(z > ql_to_datetime(today)) + 1`
                    # It was based on `y1` (combined with generated dates) and `t_list` (ois_future_tickers).
                    #
                    # This logic handles cases where Bloomberg doesn't provide enough future meeting dates.
                    # It's a heuristic for extending the list based on average inter-meeting length.
                    if len(relevant_meetings) < len(ois_future_tickers):  #
                        if not relevant_meetings.empty:
                            last_known_meeting = relevant_meetings.iloc[-1]  #
                            # Use average period from actual meetings if available
                            avg_inter_meet_days = np.mean(np.diff(cb_meeting_dates)).days if len(
                                cb_meeting_dates) > 1 else 45  # Default to 45 days

                            num_to_generate = len(ois_future_tickers) - len(relevant_meetings)  #
                            generated_meetings = [
                                last_known_meeting + datetime.timedelta(days=int(avg_inter_meet_days * i)) for i in
                                range(1, num_to_generate + 1)]  #
                            relevant_meetings = pd.concat([relevant_meetings, pd.Series(generated_meetings)],
                                                          ignore_index=True)  #
                        else:  # No meetings found at all, just generate from valuation date
                            avg_inter_meet_days = 45  # Default
                            generated_meetings = [val_date + datetime.timedelta(days=int(avg_inter_meet_days * i)) for i
                                                  in range(1, len(ois_future_tickers) + 1)]  #
                            relevant_meetings = pd.Series(generated_meetings)  #

                    # Align meeting dates with futures (assuming 1st future is 1st meeting, etc.)
                    futures_df['meet_date'] = [d.date() for d in relevant_meetings.iloc[:len(futures_df)]]  #
                    futures_df['meet'] = futures_df['meet_date'].apply(lambda d: d.strftime('%b-%y'))  #

                    # Fetch Central Bank base rate for this valuation date
                    ois_fix_df = b_con.bdh(fixing_index, 'PX_LAST',
                                           to_bbg_date_str(val_date_ql - ql.Period(2, ql.Days), ql_date=1),
                                           bbg_val_date_str)  #
                    ois_fix = ois_fix_df['PX_LAST'].iloc[0] if not ois_fix_df.empty else \
                    b_con.ref(fixing_index, 'PX_LAST')['value'][0]  #

                    base_fix_df = b_con.bdh(base_ticker, 'PX_LAST',
                                            to_bbg_date_str(val_date_ql - ql.Period(2, ql.Days), ql_date=1),
                                            bbg_val_date_str)  #
                    base_fix = base_fix_df['PX_LAST'].iloc[0] if not base_fix_df.empty else \
                    b_con.ref(base_ticker, 'PX_LAST')['value'][0]  #

                    # Store historical CB base rates
                    historical_cb_rates[ccy_code].append(('base', base_fix))  #
                    # The original `x5` accumulated historical CB rates and was also returned.
                    # It was a list of (meet_index, x4_value) for historical.
                    # This will be integrated into the historical data loader.

                    # Calculate `cb` (Central Bank implied rate) and `step`
                    futures_df['cb'] = np.round(futures_df['value'] + (base_fix - ois_fix), 2)  #
                    futures_df['step'] = 100 * np.array(
                        [futures_df['cb'].iloc[0] - base_fix] + futures_df['cb'].diff()[1:].tolist())  #
                    futures_df['cum'] = futures_df['step'].cumsum()  #

                    all_wirp_data[ccy_code].append(futures_df[['meet_date', 'meet', 'cb', 'step', 'cum']])  #

                except Exception as e:
                    print(f"Error processing WIRP data for {ccy_code} on {val_date}: {e}")
                    all_wirp_data[ccy_code].append(pd.DataFrame())  # Append empty DF on error

        return {"wirp_data": all_wirp_data, "historical_cb_rates": historical_cb_rates}

    def update_wirp_history(self, currency_codes: Union[str, List[str]], write_to_disk: bool = True,
                            update_mode: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Updates the historical WIRP data for specified currencies.

        Fetches new data from Bloomberg and appends it to existing historical pickle files.

        Args:
            currency_codes (Union[str, List[str]]): Single currency code or list of currency codes.
            write_to_disk (bool): If True, saves the updated history to a pickle file.
            update_mode (bool): If True, loads existing history and appends new data.
                                If False, starts fresh from a fixed start date.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are currency codes and values are
                                     the updated historical WIRP DataFrames.
        """
        if isinstance(currency_codes, str):  #
            currency_codes = [currency_codes]  #

        updated_histories: Dict[str, pd.DataFrame] = {}  #

        for ccy_code in currency_codes:  #
            ccy_config = config.get_currency_config(ccy_code)  #
            ois_meet_hist_file = ccy_config.get('ois_meet_hist_file')  #
            bbg_plot_tickers = ccy_config.get('bbgplot_tickers', [])  #
            cb_contrib_type = ccy_config.get('cb_contrib_type')  #
            cb_num_meets = ccy_config.get('cb_num_meets')  #
            base_ticker = ccy_config.get('base_ticker')  #

            if not ois_meet_hist_file:
                print(f"No historical WIRP file configured for {ccy_code}. Skipping update.")  #
                continue  #

            # Determine start date for historical fetch
            # Original code `d_start = ql.Date(3,1,2019)` for initial write, else from last entry.
            existing_hist_df: Optional[pd.DataFrame] = None
            if update_mode:  #
                try:  #
                    existing_hist_df = data_loader.load_pickle(ois_meet_hist_file)  #
                    # Original code: `d_start = datetime_to_ql(prev_df['date'].iloc[-20].date())`
                    # This implies fetching from 20 days before the last historical entry.
                    # For robustness, fetch from a reasonable point before the last update date.
                    start_date_ql = datetime_to_ql(existing_hist_df['date'].max()) - ql.Period(30, ql.Days)  #
                except FileNotFoundError:  #
                    print(f"Existing history file '{ois_meet_hist_file}' not found. Starting from scratch.")  #
                    start_date_ql = ql.Date(3, 1, 2019)  # Default start date for initial write
                except Exception as e:  #
                    print(f"Error loading existing history for {ccy_code}: {e}. Starting from scratch.")  #
                    start_date_ql = ql.Date(3, 1, 2019)  # Default start date for initial write
            else:  #
                start_date_ql = ql.Date(3, 1, 2019)  # Default start date for initial write

            bbg_start_date_str = to_bbg_date_str(start_date_ql, ql_date=1)  #
            bbg_today_str = to_bbg_date_str(datetime.datetime.now(), ql_date=0)  #

            # Re-construct OIS future tickers
            if ccy_code == 'USD':  #
                ois_future_tickers = [f"{bbg_plot_tickers[2]}{i} {cb_contrib_type} Curncy" for i in
                                      range(1, cb_num_meets)]  #
            else:  #
                ois_future_tickers = [f"{bbg_plot_tickers[2]}{i}A {cb_contrib_type} Curncy" for i in
                                      range(1, cb_num_meets)]  #

            try:
                # Fetch historical futures data
                all_futures_data = b_con.bdh(ois_future_tickers, 'PX_LAST', bbg_start_date_str, bbg_today_str,
                                             longdata=True)  #

                if all_futures_data.empty:
                    print(f"No new historical futures data fetched for {ccy_code}. Skipping update.")  #
                    continue  #

                all_futures_data['meet_num'] = [int(re.search(r'\d+', t).group()) for t in
                                                all_futures_data['ticker']]  #

                # Fetch all relevant CB meeting dates for the historical period
                all_cb_meeting_dates_raw = b_con.bulkref(base_ticker, 'ECO_FUTURE_RELEASE_DATE_LIST',  #
                                                         ovrds=[("START_DT",
                                                                 to_bbg_date_str(start_date_ql - ql.Period(4, ql.Years),
                                                                                 ql_date=1)),  # Fetch wider range
                                                                ("END_DT",
                                                                 to_bbg_date_str(start_date_ql + ql.Period(4, ql.Years),
                                                                                 ql_date=1))]  #
                                                         )['value']  #
                all_cb_meeting_dates_dt = pd.Series(
                    [datetime.datetime.strptime(d, '%Y/%m/%d %H:%M:%S') for d in all_cb_meeting_dates_raw])  #

                # Generate full list of potential meeting dates (actual + generated)
                # This ensures we have enough meeting dates to map to all futures, even if official list is short.
                # Heuristic: fill up to `cb_num_meets` beyond last known meeting if needed.
                #
                # The original `y1` generation combined current known meetings with generated ones.
                #
                full_meeting_series = pd.Series(all_cb_meeting_dates_dt.tolist())  #
                if not full_meeting_series.empty:  #
                    last_official_meeting = full_meeting_series.iloc[-1]  #
                    num_to_generate = cb_num_meets  # Generate enough to map to all futures
                    generated_meetings = [last_official_meeting + datetime.timedelta(days=int(45 * i)) for i in
                                          range(1, num_to_generate + 1)]  # Assume 45 day cycle
                    full_meeting_series = pd.concat([full_meeting_series, pd.Series(generated_meetings)],
                                                    ignore_index=True)  #

                # Map futures data to meeting dates
                processed_df = pd.DataFrame()
                for date in all_futures_data['date'].unique():  #
                    daily_futures = all_futures_data[all_futures_data['date'] == date].sort_values(
                        'meet_num').reset_index(drop=True)  #

                    # Find relevant meeting dates for this `date` from the full generated series
                    # Original: `y_filter = np.sum(d_unique[i] > y1)`
                    # Needs meetings *after* the current futures data date
                    meetings_after_current_date = full_meeting_series[full_meeting_series > date].reset_index(
                        drop=True)  #
                    if meetings_after_current_date.empty:
                        print(f"Warning: No future meetings found after {date} for {ccy_code}. Skipping this date.")  #
                        continue  #

                    # Select `cb_num_meets` relevant future meetings
                    relevant_meetings_for_day = meetings_after_current_date.iloc[:cb_num_meets]  #

                    if len(relevant_meetings_for_day) < len(daily_futures):
                        print(
                            f"Warning: Not enough meeting dates for all futures on {date}. Some futures might be skipped.")  #

                    # Map futures to meetings
                    daily_futures['meet_date'] = [d.date() for d in
                                                  relevant_meetings_for_day.iloc[:len(daily_futures)]]  #
                    daily_futures['meet'] = daily_futures['meet_date'].apply(lambda d: d.strftime('%b-%y'))  #

                    # Calculate step and cum
                    daily_futures['step2'] = np.round(100 * (daily_futures['value'].diff()), 1)  #
                    daily_futures['step'] = daily_futures.apply(
                        lambda row: 0.0 if row['meet_num'] == 1 else row['step2'], axis=1)  #
                    daily_futures['cum'] = daily_futures['step'].cumsum()  #

                    processed_df = pd.concat([processed_df, daily_futures])  #

                # Concatenate with existing history, remove duplicates
                if existing_hist_df is not None and not existing_hist_df.empty:  #
                    # Remove overlap by filtering out dates already in `processed_df` from `existing_hist_df`
                    dates_in_new_data = processed_df['date'].unique()  #
                    df_feed = existing_hist_df[~existing_hist_df['date'].isin(dates_in_new_data)]  #
                    final_df = pd.concat([df_feed, processed_df], ignore_index=True)  #
                else:
                    final_df = processed_df  #

                final_df = final_df.sort_values(by=['date', 'meet_num']).reset_index(drop=True)  #

                updated_histories[ccy_code] = final_df  #

                if write_to_disk:  #
                    data_loader.save_pickle(final_df, ois_meet_hist_file)  #
                    print(f"Updated WIRP history saved to {ois_meet_hist_file} for {ccy_code}.")  #

            except Exception as e:
                print(f"Error updating WIRP history for {ccy_code}: {e}")
                if update_mode and existing_hist_df is not None:
                    updated_histories[ccy_code] = existing_hist_df  # Return existing if update fails
                else:
                    updated_histories[ccy_code] = pd.DataFrame()  # Return empty if initial fetch fails

        return updated_histories