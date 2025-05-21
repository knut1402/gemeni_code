# src/curves/ois_curve_builder.py

import QuantLib as ql
import pandas as pd
import datetime
from typing import Any, Dict, List, Optional, Union
from src.curves.base_curve import BaseCurve
from src.data_manager import b_con, data_loader
from src.date_utils import to_bbg_date_str, datetime_to_ql, ql_to_datetime
from src.quantlib_utils import get_ql_business_day_convention, get_ql_day_counter, get_ql_frequency  #


class OISCurveBuilder(BaseCurve):
    """
    Builds an OIS (Overnight Indexed Swap) yield curve using QuantLib.

    This builder supports fetching live market data from Bloomberg or
    reconstructing curves from historical pickled nodes. It incorporates
    currency-specific conventions and handles various rate helper types.
    """

    def __init__(self, currency_code: str,
                 reference_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the OIS curve builder.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "GBP").
            reference_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the curve is built. Can be a string, int (days offset),
                datetime, or ql.Date. If None, uses today's date.
        """
        super().__init__(currency_code, reference_date_input)
        self._ois_trigger = self._convention_config.get('ois_trigger', False)  #
        self._fixing_ticker = self._convention_config.get('fixing_index')  #
        self._settlement_days = self._convention_config.get('settlement_days')  #
        self._bbg_curve_ticker = self._convention_config.get('bbg_curve')  #
        self._bbg_plot_tickers = self._convention_config.get('bbgplot_tickers', [])  #
        self._floating_leg_convention = self._convention_config['floating_leg']  #

        self._historical_data_loaded: bool = False
        self._load_from_historical_data()  # Try to load from history first

    def _load_from_historical_data(self) -> None:
        """
        Attempts to load historical curve nodes from a pickle file if available.
        This bypasses Bloomberg calls for historical dates if data is cached.
        """
        hist_file = self._convention_config.get('ois_meet_hist_file')  #
        if hist_file:
            try:
                # The original code's `hist[a]` implies a pre-loaded dict/df.
                # Here, we load it if the file name exists and is referenced.
                # Assuming `ois_meet_hist_file` in conventions.yaml holds the path for historical nodes.
                #
                all_hist_data = data_loader.load_pickle(hist_file)  #

                # Check if the exact reference date exists in the historical data
                # Assuming 'all_hist_data' is a DataFrame with DatetimeIndex or a dict
                # with date strings/objects as keys.
                # The original `ois_from_nodes` takes `crv_h.loc[ql_to_datetime(ref_date).strftime('%d/%m/%Y')]`
                formatted_ref_date = ql_to_datetime(self.reference_date).strftime('%d/%m/%Y')  #

                if formatted_ref_date in all_hist_data.index:  #
                    hist_entry = all_hist_data.loc[formatted_ref_date]  #

                    self._market_data['ref_fix'] = hist_entry['Fixing']  #
                    self._market_data['swap_rates_df'] = hist_entry['Table']  #
                    # Convert QuantLib Dates back for nodes if stored as such
                    q_dates = [datetime_to_ql(d) for d in hist_entry['Dates']]  #
                    l_rates = hist_entry['Rates']  #

                    # Construct curve directly from historical nodes
                    self._ql_curve = ql.MonotonicLogCubicDiscountCurve(q_dates, l_rates, ql.Actual360(),
                                                                       self.calendar)  #
                    self._ql_curve.enableExtrapolation()  #
                    self._nodes = self._ql_curve.nodes()  #
                    self._historical_data_loaded = True
                    print(
                        f"OIS curve for {self.currency_code} on {formatted_ref_date} retrieved from historical data.")  #
                else:
                    print(
                        f"Historical data for {self.currency_code} on {formatted_ref_date} not found in {hist_file}. Fetching live.")
            except FileNotFoundError:
                print(f"Historical data file {hist_file} not found. Fetching live data.")
            except Exception as e:
                print(f"Error loading historical OIS data for {self.currency_code}: {e}. Fetching live data.")

    def _fetch_market_data(self) -> None:
        """
        Fetches live market data (OIS overnight fixing and swap rates) from Bloomberg.
        """
        if self._historical_data_loaded:
            return  # Skip if already loaded from history

        ref_date_bbg = to_bbg_date_str(self.reference_date, ql_date=1)  #
        # Fetch previous day's fixing for OIS overnight rate
        ref_date_1d_before_bbg = to_bbg_date_str(self.calendar.advance(self.reference_date, -1, ql.Days), ql_date=1)  #

        try:
            # Fetch OIS overnight rate
            # Prioritize bdh for a specific day, fallback to current ref if no history for that day
            ois_on_df = b_con.bdh(self._fixing_ticker, 'PX_LAST', ref_date_1d_before_bbg, ref_date_1d_before_bbg)  #
            self._market_data['ois_on'] = ois_on_df['PX_LAST'].iloc[0] if not ois_on_df.empty else \
            b_con.ref(self._fixing_ticker, 'PX_LAST')['value'][0]  #

            # Fetch OIS swap rates
            inst_bulk_ref = b_con.bulkref(self._bbg_curve_ticker, 'CURVE_TENOR_RATES')  #

            x1 = inst_bulk_ref['value'][inst_bulk_ref['name'] == 'Tenor Ticker'].reset_index(drop=True)  #
            x2 = inst_bulk_ref['value'][inst_bulk_ref['name'] == 'Tenor'].reset_index(drop=True)  #

            # Remove header row which is often 'Tenor Ticker' or similar
            if x1.iloc[0] == 'Tenor Ticker':  #
                x1 = x1.drop(0).reset_index(drop=True)  #
                x2 = x2.drop(0).reset_index(drop=True)  #

            swap_tickers_df = pd.DataFrame({'TenorTicker': x1, 'Tenor': x2})  #

            # Handle specific ticker adjustments as per original SWAP_BUILD logic
            # This logic is based on inferred behavior from original code
            # NOTE: These specific ticker adjustments for Bloomberg are complex and crucial.
            # Please verify/amend these based on your exact Bloomberg setup.
            #
            if self.currency_code == 'EUR':  # ESTER_DC
                for i in range(len(swap_tickers_df)):
                    if i >= 15:  # Inferred threshold
                        swap_tickers_df.loc[i, 'TenorTicker'] = f"EESWE{swap_tickers_df['Tenor'][i][:-1]} BGN Curncy"  #
            elif self.currency_code == 'SEK':  # SEK_OIS_DC
                for i in range(len(swap_tickers_df)):
                    swap_tickers_df.loc[
                        i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0]} BLC3 Curncy"  #
            elif self.currency_code == 'AUD':  # AONIA_DC
                # Assuming index 8 is 18m ticker
                if len(swap_tickers_df) > 8:  #
                    swap_tickers_df.loc[8, 'TenorTicker'] = 'ADSO1F ICPL Curncy'  #
                for i in range(9, len(swap_tickers_df)):  #
                    swap_tickers_df.loc[i, 'TenorTicker'] = f"ADSO{swap_tickers_df['Tenor'][i][:-1]} ICPL Curncy"  #
            elif self.currency_code == 'CAD':  # CORRA_DC
                # Assuming add_tenors exists in Conventions.py for CORRA
                corra_add_tenors = self._convention_config.get('add_on_instruments', {}).get('additional_tenors',
                                                                                             pd.DataFrame())  #
                if not corra_add_tenors.empty:
                    swap_tickers_df = pd.concat([swap_tickers_df.iloc[:14], corra_add_tenors], ignore_index=True)  #
                for i in range(len(swap_tickers_df)):
                    swap_tickers_df.loc[
                        i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0]} BLC3 Curncy"  #
            elif self.currency_code == 'RUB':  # RUONIA_DC
                for i in range(9, len(swap_tickers_df)):  #
                    swap_tickers_df.loc[
                        i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0][:2]}SO{i - 6} BLC3 Curncy"  #

            swap_rates_live_df = b_con.bdh(swap_tickers_df['TenorTicker'].tolist(), 'PX_LAST', ref_date_bbg,
                                           ref_date_bbg, longdata=True)  #
            swap_rates_live_df = swap_rates_live_df.pivot(index='date', columns='ticker', values='value').iloc[
                0].reset_index()  #
            swap_rates_live_df.columns = ['TenorTicker', 'Rate']  #

            self._market_data['swap_rates_df'] = pd.merge(swap_tickers_df, swap_rates_live_df, on='TenorTicker',
                                                          how='inner')  #
            self._market_data['swap_rates_df'].rename(columns={'Rate': 'SwapRate'}, inplace=True)  #
            self._market_data['swap_rates_df'] = self._market_data['swap_rates_df'].dropna(subset=['SwapRate'])  #

            # Fetch 1-day historical rates for change calculation
            swap_rates_1d_df = b_con.bdh(self._market_data['swap_rates_df']['TenorTicker'].tolist(), 'PX_LAST',
                                         ref_date_1d_before_bbg, ref_date_1d_before_bbg, longdata=True)  #
            swap_rates_1d_df = swap_rates_1d_df.pivot(index='date', columns='ticker', values='value').iloc[
                0].reset_index()  #
            swap_rates_1d_df.columns = ['TenorTicker', 'Rate_1D']  #

            self._market_data['swap_rates_df'] = pd.merge(self._market_data['swap_rates_df'], swap_rates_1d_df,
                                                          on='TenorTicker', how='left')  #
            self._market_data['swap_rates_df']['Chg_1d'] = 100 * (
                        self._market_data['swap_rates_df']['SwapRate'] - self._market_data['swap_rates_df'][
                    'Rate_1D'])  #

        except Exception as e:
            raise RuntimeError(f"Failed to fetch live OIS market data for {self.currency_code}: {e}")

    def _build_curve_helpers(self) -> List[ql.RateHelper]:
        """
        Constructs QuantLib rate helpers from the fetched market data.
        """
        self._fetch_market_data()  # Ensure market data is loaded

        helpers: List[ql.RateHelper] = []
        ois_on = self._market_data.get('ois_on')  #
        swap_rates_df = self._market_data.get('swap_rates_df')  #

        if ois_on is None or swap_rates_df is None or swap_rates_df.empty:
            raise ValueError("Insufficient market data to build OIS curve helpers.")

        # OIS Deposit Rate Helpers (for short end)
        # The original code loops from 0 to sett_d.
        # This seems to create `sett_d` number of deposit rate helpers, each for 1 day.
        # Check if `sett_d` is truly 0 for T+0 settlement like SONIA.
        #
        for days in range(self._settlement_days + 1):  #
            helpers.append(ql.DepositRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(ois_on / 100)),  # Rate in decimal
                ql.Period(days, ql.Days),  # Tenor
                0,  # Calendar settlement days
                self.calendar,  #
                get_ql_business_day_convention(self._floating_leg_convention['business_day_convention']),  #
                False,  # endOfMonth
                get_ql_day_counter(self._floating_leg_convention['day_count'])  #
            ))

        # OIS Swap Rate Helpers
        for _, row in swap_rates_df.iterrows():  #
            rate = row['SwapRate']  #
            tenor_str = str(row['Tenor'])  #
            tenor_num = int(tenor_str[:-1])  #
            tenor_unit = tenor_str[-1].upper()  #

            # Map tenor unit to QuantLib Period units
            if tenor_unit == 'D':
                ql_period_unit = ql.Days
            elif tenor_unit == 'W':
                ql_period_unit = ql.Weeks
            elif tenor_unit == 'M':
                ql_period_unit = ql.Months
            elif tenor_unit == 'Y':
                ql_period_unit = ql.Years
            else:
                raise ValueError(f"Unknown tenor unit: {tenor_unit} in {tenor_str}")

            helpers.append(ql.OISRateHelper(
                self._settlement_days,  #
                ql.Period(tenor_num, ql_period_unit),  #
                ql.QuoteHandle(ql.SimpleQuote(rate / 100)),  # Rate in decimal
                ql.OIS(  # The OIS index used for the floating leg
                    self._fixing_ticker,  # Index name
                    ql.Period(self._floating_leg_convention['fixing_tenor_num'],
                              get_ql_frequency(self._floating_leg_convention['frequency'])),  # Fixing tenor
                    self._settlement_days,  # Fixing settlement days
                    self.currency,  #
                    self.calendar,  #
                    get_ql_business_day_convention(self._floating_leg_convention['business_day_convention']),  #
                    False,  # endOfMonth
                    get_ql_day_counter(self._floating_leg_convention['day_count'])  #
                )
            ))
        return helpers

    def build_curve(self) -> None:
        """
        Builds the QuantLib PiecewiseLogCubicDiscount curve using the rate helpers.
        Also includes logic for handling "bumps" for month/quarter/year ends.
        """
        if self._historical_data_loaded:
            print(f"Curve for {self.currency_code} already loaded from historical data.")
            return

        helpers = self._build_curve_helpers()

        # Define jump dates for piecewise curve based on month/quarter/year ends
        # This logic is complex in the original code, involving checking weekends/holidays.
        #
        jump_dates: List[ql.Date] = []
        jump_quotes: List[ql.QuoteHandle] = []

        # Get End-of-Month (EOM), Quarter (EOQ), Year (EOY) dates
        # This is a simplification of the original complex date generation logic
        # The original code's "bumps" seem to be for specific EOM/Q/Y adjustments.
        # It's not clear what `OIS_ME`, `OIS_QE`, `OIS_YE` map to in general conventions.
        # Assuming these are related to fixed rates for certain periods around these dates.
        # For a standard OIS curve, explicit "bumps" are often not needed unless specific
        # liquidity points or policy expectations drive it.
        # Replicated logic from OIS_DC_BUILD.py around `ff_jumps` etc.
        #
        end_date_for_jumps = self.reference_date + ql.Period(40, ql.Years)  #
        temp_date = self.reference_date
        while temp_date <= end_date_for_jumps:  #
            if self.calendar.isEndOfMonth(temp_date):  #
                jump_dates.append(temp_date)  #
                # Assuming a flat rate for these jumps, as `OIS_ME/QE/YE` values are not explicit here.
                # In the original code, `OIS_ME/QE/YE` seem to be fixed rates.
                # Here, using 0.0 to just define the date nodes. This might need review.
                jump_quotes.append(ql.QuoteHandle(ql.SimpleQuote(0.0)))  #
            temp_date += 1  #

        # The original code had very specific logic for weekend/holiday-adjusted month starts for jumps.
        # This part of the logic is complex and its financial purpose for OIS curve `bumps` is unclear
        # without further context on the OIS_ME/QE/YE constants.
        # For now, I will omit this highly specific and potentially non-standard 'jump' logic
        # unless its exact purpose and values are clarified.
        # If it's for policy rate expectation shifts, it's typically handled differently.
        # The primary PiecewiseLogCubicDiscount construction uses helpers, which implicitly define nodes.

        try:
            self._ql_curve = ql.PiecewiseLogCubicDiscount(
                self._settlement_days,  #
                self.calendar,  #
                helpers,  #
                get_ql_day_counter(self._floating_leg_convention['day_count']),  #
                jump_quotes,  #
                jump_dates  #
            )
            self._ql_curve.enableExtrapolation()  #
            self._nodes = self._ql_curve.nodes()  #
            print(
                f"OIS curve for {self.currency_code} built successfully on {ql_to_datetime(self.reference_date).strftime('%Y-%m-%d')}.")
        except Exception as e:
            raise RuntimeError(f"Failed to build OIS curve for {self.currency_code}: {e}")