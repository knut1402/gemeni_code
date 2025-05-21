# src/curves/swap_curve_builder.py

import QuantLib as ql
import pandas as pd
import datetime
from typing import Any, Dict, List, Optional, Union
from src.curves.base_curve import BaseCurve
from src.data_manager import b_con, data_loader
from src.date_utils import to_bbg_date_str, datetime_to_ql, ql_to_datetime
from src.quantlib_utils import get_ql_business_day_convention, get_ql_day_counter, get_ql_frequency  #
from src.curves.ois_curve_builder import OISCurveBuilder  # For multi-curve discounting


class SwapCurveBuilder(BaseCurve):
    """
    Builds an interest rate swap (e.g., Libor, Euribor, TIIE) yield curve using QuantLib.

    This builder supports various market instruments as helpers (deposits, FRAs, futures, swaps)
    and can integrate with an OIS discount curve for multi-curve framework.
    """

    def __init__(self, currency_code: str,
                 reference_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the Swap curve builder.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "EUR").
            reference_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the curve is built. Can be a string, int (days offset),
                datetime, or ql.Date. If None, uses today's date.
        """
        super().__init__(currency_code, reference_date_input)

        self._ois_trigger = self._convention_config.get('ois_trigger', False)  #
        self._dc_index = self._convention_config.get('dc_index')  #
        self._fixing_ticker = self._convention_config.get('fixing_index')  #
        self._fixing_tenor_ql = ql.Period(self._convention_config.get('fixing_tenor'))  #
        self._settlement_days = self._convention_config.get('settlement_days')  #
        self._bbg_curve_ticker = self._convention_config.get('bbg_curve')  #
        self._add_on_instruments = self._convention_config.get('add_on_instruments', {})  #
        self._fixed_leg_convention = self._convention_config['fixed_leg']  #
        self._floating_leg_convention = self._convention_config['floating_leg']  #
        self._start_swap_tenor = self._convention_config.get('start_swap_tenor')  # # e.g., '1W', '1M', '2M'

        self._discount_curve: Optional[ql.YieldTermStructureHandle] = None
        self._historical_data_loaded: bool = False
        self._load_from_historical_data()  # Try to load from history first

    def _load_from_historical_data(self) -> None:
        """
        Attempts to load historical curve nodes from a pickle file if available.
        This bypasses Bloomberg calls for historical dates if data is cached.
        """
        # The original code (in SWAP_BUILD.py) implies a generic 'hist' object from Conventions.py
        # which loads different currency histories.
        #
        hist_file_key = f"{self.currency_code}_H.pkl"  # Inferred common pattern for swap curve historical files
        try:
            # Assuming a naming convention like 'USD_3M_H.pkl' for historical swap curve nodes.
            # This logic directly loads the historical data structure used by `libor_from_nodes`
            all_hist_data = data_loader.load_pickle(hist_file_key)  #

            formatted_ref_date = ql_to_datetime(self.reference_date).strftime('%d/%m/%Y')  #

            if formatted_ref_date in all_hist_data.index:  #
                hist_entry = all_hist_data.loc[formatted_ref_date]  #

                self._market_data['ref_fix'] = hist_entry['Fixing']  #
                self._market_data['swap_rates_df'] = hist_entry['Table']  #

                q_dates = [datetime_to_ql(d) for d in hist_entry['Dates']]  #
                l_rates = hist_entry['Rates']  #

                # Reconstruct curve from historical nodes
                self._ql_curve = ql.MonotonicLogCubicDiscountCurve(q_dates, l_rates,
                                                                   get_ql_day_counter(
                                                                       self._floating_leg_convention['day_count']),  #
                                                                   self.calendar)  #
                self._ql_curve.enableExtrapolation()  #
                self._nodes = self._ql_curve.nodes()  #
                self._historical_data_loaded = True
                print(f"Swap curve for {self.currency_code} on {formatted_ref_date} retrieved from historical data.")

                # If OIS triggered, load/build OIS discount curve for this historical date
                if self._ois_trigger and self._dc_index:  #
                    ois_builder = OISCurveBuilder(self._dc_index, self.reference_date)  #
                    # OISCurveBuilder's __init__ will attempt to load from its own history for this date
                    ois_builder.build_curve()  # Ensure the OIS curve is built or loaded
                    self._discount_curve = ql.YieldTermStructureHandle(ois_builder.curve)  #

            else:
                print(
                    f"Historical data for {self.currency_code} on {formatted_ref_date} not found in {hist_file_key}. Fetching live.")
        except FileNotFoundError:
            print(f"Historical data file {hist_file_key} not found. Fetching live data.")
        except Exception as e:
            print(f"Error loading historical Swap curve data for {self.currency_code}: {e}. Fetching live data.")

    def _fetch_market_data(self) -> None:
        """
        Fetches live market data (deposit rates, FRA/futures, swap rates) from Bloomberg.
        """
        if self._historical_data_loaded:
            return  # Skip if already loaded from history

        ref_date_bbg = to_bbg_date_str(self.reference_date, ql_date=1)  #
        ref_date_1d_before_bbg = to_bbg_date_str(self.calendar.advance(self.reference_date, -1, ql.Days), ql_date=1)  #

        try:
            # Fetch deposit rate
            depo_df = b_con.bdh(self._fixing_ticker, 'PX_LAST', ref_date_1d_before_bbg, ref_date_1d_before_bbg)  #
            self._market_data['deposit_rate'] = depo_df['PX_LAST'].iloc[0] if not depo_df.empty else \
            b_con.ref(self._fixing_ticker, 'PX_LAST')['value'][0]  #

            # Fetch swap rates from Bloomberg
            inst_bulk_ref = b_con.bulkref(self._bbg_curve_ticker, 'CURVE_TENOR_RATES')  #

            x1_tickers = inst_bulk_ref['value'][inst_bulk_ref['name'] == 'Tenor Ticker'].reset_index(drop=True)  #
            x2_tenors = inst_bulk_ref['value'][inst_bulk_ref['name'] == 'Tenor'].reset_index(drop=True)  #

            # Remove header row
            if x1_tickers.iloc[0] == 'Tenor Ticker':  #
                x1_tickers = x1_tickers.drop(0).reset_index(drop=True)  #
                x2_tenors = x2_tenors.drop(0).reset_index(drop=True)  #

            swap_tickers_df = pd.DataFrame({'TenorTicker': x1_tickers, 'Tenor': x2_tenors})  #

            # Filter from self._start_swap_tenor as per original code
            if self._start_swap_tenor:  #
                start_idx = swap_tickers_df[swap_tickers_df['Tenor'] == self._start_swap_tenor].index.values  #
                if start_idx.size > 0:
                    swap_tickers_df = swap_tickers_df.iloc[int(start_idx[0]):].reset_index(drop=True)  #
                else:
                    print(
                        f"Warning: start_swap_tenor '{self._start_swap_tenor}' not found. Using all available tenors.")

            # Apply specific Bloomberg ticker adjustments based on currency
            # NOTE: These specific ticker adjustments for Bloomberg are complex and crucial.
            # Please verify/amend these based on your exact Bloomberg setup.
            #
            if self.currency_code == 'AUD_3M':  #
                # Inferred logic: modify tickers up to 10Y and then for longer tenors
                ten_y_idx = swap_tickers_df[swap_tickers_df['Tenor'] == '10Y'].index  #
                if not ten_y_idx.empty:  #
                    for i in range(len(swap_tickers_df)):  #
                        if i < int(ten_y_idx[0]):  #
                            swap_tickers_df.loc[
                                i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0]}Q CBBT Curncy"  #
                        else:  #
                            swap_tickers_df.loc[
                                i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0][:4]}{swap_tickers_df['Tenor'][i][:2]}Q CMPN Curncy"  #
            elif self.currency_code == 'AUD_6M':  #
                # Inferred logic: special handling for first two tickers, then CBBT
                if len(swap_tickers_df) >= 2:  #
                    for i in range(2):  #
                        swap_tickers_df.loc[
                            i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0][:-1]} BGN Curncy"  #
                    for i in range(2, len(swap_tickers_df)):  #
                        swap_tickers_df.loc[
                            i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0]} CBBT Curncy"  #
            elif self.currency_code == 'NOK_3M':  #
                for i in range(len(swap_tickers_df)):  #
                    swap_tickers_df.loc[
                        i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0]}V3 BGN Curncy"  #
            elif self.currency_code in ['CZK_3M', 'HUF_3M']:  #
                for i in range(len(swap_tickers_df)):  #
                    swap_tickers_df.loc[
                        i, 'TenorTicker'] = f"{swap_tickers_df['TenorTicker'][i].split()[0]}V3 BLC3 Curncy"  #
            elif self.currency_code == 'MXN_TIIE':  #
                # Special handling for MXN_TIIE, which uses 28-day periods.
                # This conversion is implemented in `_build_curve_helpers` due to its impact on helpers.
                pass  # No direct ticker modification here, handled in helper construction

            live_swap_rates_df = b_con.bdh(swap_tickers_df['TenorTicker'].tolist(), 'PX_LAST', ref_date_bbg,
                                           ref_date_bbg, longdata=True)  #
            live_swap_rates_df = live_swap_rates_df.pivot(index='date', columns='ticker', values='value').iloc[
                0].reset_index()  #
            live_swap_rates_df.columns = ['TenorTicker', 'SwapRate']  #

            self._market_data['swap_rates_df'] = pd.merge(swap_tickers_df, live_swap_rates_df, on='TenorTicker',
                                                          how='inner')  #
            self._market_data['swap_rates_df'] = self._market_data['swap_rates_df'].dropna(subset=['SwapRate'])  #

            # Fetch additional instruments (FRA or FUT) if specified in conventions
            if self._add_on_instruments.get('type') == 'FRA':  #
                fra_tickers = self._add_on_instruments.get('tickers', [])  #
                fra_data = b_con.bdh(fra_tickers, 'PX_LAST', ref_date_bbg, ref_date_bbg, longdata=True)  #
                # Assuming original code's `SECURITY_TENOR_TWO_RT` to get start month
                fra_start_months_data = b_con.ref_hist(fra_tickers, 'SECURITY_TENOR_TWO_RT', dates=[ref_date_bbg])  #
                fra_start_months_data['StartStub'] = fra_start_months_data['value'].apply(
                    lambda x: int(x.split('M')[0]))  #
                fra_df = pd.merge(fra_data.pivot(index='date', columns='ticker', values='value').iloc[0].reset_index(),
                                  #
                                  fra_start_months_data[['ticker', 'StartStub']], on='ticker', how='inner')  #
                fra_df.columns = ['Ticker', 'Rate', 'StartStub']  #
                self._market_data['fra_rates_df'] = fra_df  #

            elif self._add_on_instruments.get('type') == 'FUT':  #
                fut_tickers = self._add_on_instruments.get('tickers', [])  #
                fut_conv_corr = self._add_on_instruments.get('conversion_corrections', {})  #

                fut_data = b_con.bdh(fut_tickers, 'PX_LAST', ref_date_1d_before_bbg, ref_date_bbg, longdata=True)  #
                fut_data = fut_data.pivot(index='date', columns='ticker', values='value').iloc[0].reset_index()  #
                fut_data.columns = ['Ticker', 'Price']  #

                # Apply conversion corrections
                fut_data['CCAdjPx'] = fut_data.apply(lambda row: row['Price'] + fut_conv_corr.get(row['Ticker'], 0.0),
                                                     axis=1)  #

                # Fetch start dates for futures
                # The original code checks for 'LAST_TRADEABLE_DT' or 'SW_EFF_DT'
                # and adjusts settlement days for CAD/NZD.
                # Assuming 'LAST_TRADEABLE_DT' for future helpers typically.
                fut_start_dates_data = b_con.ref_hist(fut_tickers, 'LAST_TRADEABLE_DT', dates=[ref_date_bbg])  #

                # Adjust settlement days based on currency for futures start dates if required
                fra_sett_adj = self._settlement_days  # Default
                if self.currency_code == 'CAD_3M':  #
                    fra_sett_adj = 2  #
                elif self.currency_code == 'NZD_3M':  #
                    fra_sett_adj = 0  #

                fut_start_dates_data['StartDate'] = fut_start_dates_data['value'].apply(
                    lambda x: datetime_to_ql(x) + ql.Period(fra_sett_adj, ql.Days))  #

                fut_df = pd.merge(fut_data, fut_start_dates_data[['ticker', 'StartDate']], left_on='Ticker',
                                  right_on='ticker', how='inner')  #
                self._market_data['futures_rates_df'] = fut_df  #

        except Exception as e:
            raise RuntimeError(f"Failed to fetch live Swap market data for {self.currency_code}: {e}")

    def _build_curve_helpers(self) -> List[ql.RateHelper]:
        """
        Constructs QuantLib rate helpers from the fetched market data.
        Includes Deposit, FRA, Futures, and Swap Rate Helpers.
        """
        self._fetch_market_data()  # Ensure market data is loaded

        helpers: List[ql.RateHelper] = []
        deposit_rate = self._market_data.get('deposit_rate')  #
        swap_rates_df = self._market_data.get('swap_rates_df')  #

        if deposit_rate is None or swap_rates_df is None or swap_rates_df.empty:
            raise ValueError("Insufficient market data (deposit or swap rates) to build Swap curve helpers.")

        # Deposit Rate Helper
        helpers.append(ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(deposit_rate / 100)),  # Rate in decimal
            self._fixing_tenor_ql,  # Tenor from conventions
            self._settlement_days,  #
            self.calendar,  #
            get_ql_business_day_convention(self._floating_leg_convention['business_day_convention']),  #
            False,  # endOfMonth
            get_ql_day_counter(self._floating_leg_convention['day_count'])  #
        ))

        # Additional instruments (FRA or Futures)
        if self._add_on_instruments.get('type') == 'FRA':  #
            fra_rates_df = self._market_data.get('fra_rates_df')  #
            if fra_rates_df is not None:  #
                for _, row in fra_rates_df.iterrows():  #
                    helpers.append(ql.FraRateHelper(
                        ql.QuoteHandle(ql.SimpleQuote(row['Rate'] / 100)),  # Rate in decimal
                        row['StartStub'],  # Months to start
                        self._ql_currency.name,  # Index name (e.g., USD, EUR, GBP)
                        self._fixing_tenor_ql  #
                    ))
        elif self._add_on_instruments.get('type') == 'FUT':  #
            futures_rates_df = self._market_data.get('futures_rates_df')  #
            if futures_rates_df is not None:  #
                for _, row in futures_rates_df.iterrows():  #
                    helpers.append(ql.FuturesRateHelper(
                        ql.QuoteHandle(ql.SimpleQuote(row['CCAdjPx'])),  # Adjusted price
                        row['StartDate'],  # Start date (tradeable date + settlement)
                        3,  # IMM quarters from start date (common for 3M futures)
                        self.calendar,  #
                        get_ql_business_day_convention(self._floating_leg_convention['business_day_convention']),  #
                        True,  # endOfMonth
                        get_ql_day_counter(self._floating_leg_convention['day_count']),  #
                        ql.QuoteHandle(ql.SimpleQuote(0.0)),  # Price conversion correction
                        getattr(ql, self._add_on_instruments.get('fut_type', 'Futures'))()  # e.g., ql.Futures.IMM
                    ))

        # Swap Rate Helpers
        for _, row in swap_rates_df.iterrows():  #
            rate = row['SwapRate']  #
            tenor_str = str(row['Tenor'])  #
            tenor_num = int(tenor_str[:-1])  #
            tenor_unit = tenor_str[-1].upper()  #

            if tenor_unit == 'D':
                ql_period_unit = ql.Days
            elif tenor_unit == 'M':
                ql_period_unit = ql.Months
            elif tenor_unit == 'Y':
                ql_period_unit = ql.Years
            else:
                raise ValueError(f"Unknown tenor unit: {tenor_unit} in {tenor_str}")

            # Special handling for MXN_TIIE's 28-day periods
            if self.currency_code == 'MXN_TIIE' and tenor_unit == 'D':  #
                # MXN_TIIE logic from original implies conversion of Y/M to D (28*13*Y or 28*M)
                # The 'SwapTenor' calculation would be done here for each tenor.
                # Assuming tenor_num here is already in days for 'D' unit, or needs conversion from M/Y.
                # The original `x4['SwapTenor']` was `28*int(x4['Tenor'][i][:-1])` or `28*13*int(x4['Tenor'][i][:-1])`
                # This suggests tenor_num is the number of 28-day periods or months/years.
                # Reconfirming: if 'Tenor' is '1M', then tenor_num=1, tenor_unit='M'.
                # For MXN, that '1M' means 28 days. '1Y' means 13*28 days.
                # So if tenor_unit is Y or M, we convert to Days for the period object.
                if tenor_unit == 'Y':  #
                    tenor_num = tenor_num * 13 * 28  #
                    ql_period_unit = ql.Days  #
                elif tenor_unit == 'M':  #
                    tenor_num = tenor_num * 28  #
                    ql_period_unit = ql.Days  #

            helpers.append(ql.SwapRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(rate / 100)),  # Rate in decimal
                ql.Period(tenor_num, ql_period_unit),  #
                self.calendar,  #
                get_ql_frequency(self._fixed_leg_convention['frequency']),  #
                get_ql_business_day_convention(self._fixed_leg_convention['business_day_convention']),  #
                get_ql_day_counter(self._fixed_leg_convention['day_count']),  #
                ql.IborIndex(  # Floating leg index
                    self._fixing_ticker,  # Index name
                    self._fixing_tenor_ql,  # Fixing tenor
                    self._settlement_days,  #
                    self.currency,  #
                    self.calendar,  #
                    get_ql_business_day_convention(self._floating_leg_convention['business_day_convention']),  #
                    False,  # endOfMonth
                    get_ql_day_counter(self._floating_leg_convention['day_count'])  #
                ),
                ql.QuoteHandle(),  # Spread on floating leg (not used here)
                ql.Period(0, ql.Days),  # Fwd Start (not used here)
                self._discount_curve  # Discounting curve handle
            ))
        return helpers

    def build_curve(self) -> None:
        """
        Builds the QuantLib PiecewiseLogCubicDiscount curve for the swap rates.
        Initializes the OIS discount curve if required by conventions.
        """
        if self._historical_data_loaded:
            print(f"Swap curve for {self.currency_code} already loaded from historical data.")
            return

        # Build OIS discount curve if required
        if self._ois_trigger and self._dc_index and self._discount_curve is None:  #
            try:
                ois_builder = OISCurveBuilder(self._dc_index, self.reference_date)  #
                ois_builder.build_curve()  # Build the OIS curve
                self._discount_curve = ql.YieldTermStructureHandle(ois_builder.curve)  #
                print(f"OIS discount curve ({self._dc_index}) built for {self.currency_code} swap curve.")
            except Exception as e:
                print(
                    f"Warning: Could not build OIS discount curve for {self.currency_code}: {e}. Proceeding without explicit OIS discount curve.")
                # Fallback: if OIS curve fails, use the main curve for discounting itself.
                # This is a common practice when a separate discount curve is not available.
                self._discount_curve = ql.RelinkableYieldTermStructureHandle()  # Create an empty handle to be linked later

        helpers = self._build_curve_helpers()

        try:
            self._ql_curve = ql.PiecewiseLogCubicDiscount(
                self._settlement_days,  #
                self.calendar,  #
                helpers,  #
                get_ql_day_counter(self._floating_leg_convention['day_count'])  #
            )
            self._ql_curve.enableExtrapolation()  #
            self._nodes = self._ql_curve.nodes()  #

            # If OIS curve failed to build, link the main curve to the discount curve handle
            if self._ois_trigger and self._dc_index and self._discount_curve and self._discount_curve.empty():  #
                self._discount_curve.linkTo(self._ql_curve)  #

            print(
                f"Swap curve for {self.currency_code} built successfully on {ql_to_datetime(self.reference_date).strftime('%Y-%m-%d')}.")
        except Exception as e:
            raise RuntimeError(f"Failed to build Swap curve for {self.currency_code}: {e}")

    @property
    def discount_curve(self) -> Optional[ql.YieldTermStructureHandle]:
        """
        Returns the QuantLib discount curve handle used for pricing (can be OIS or itself).
        """
        return self._discount_curve