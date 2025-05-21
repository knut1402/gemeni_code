# src/curves/inflation_curve_builder.py

import QuantLib as ql
import pandas as pd
import datetime
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.curves.base_curve import BaseCurve
from src.data_manager import b_con, data_loader
from src.date_utils import to_bbg_date_str, datetime_to_ql, ql_to_datetime
from src.quantlib_utils import setup_ql_settings
from src.curves.ois_curve_builder import OISCurveBuilder  # For nominal discounting


class InflationCurveBuilder(BaseCurve):
    """
    Builds an inflation zero-coupon (ZC) swap curve using QuantLib,
    projecting future inflation indices based on market data and seasonality.
    """

    def __init__(self, inflation_index_name: str,
                 reference_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None,
                 base_month_offset: int = 0):
        """
        Initializes the Inflation curve builder.

        Args:
            inflation_index_name (str): The name of the inflation index (e.g., "HICPxT", "UKRPI", "USCPI").
            reference_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the curve is built. Can be a string, int (days offset),
                datetime, or ql.Date. If None, uses today's date.
            base_month_offset (int): Offset in months to adjust the base month for inflation index projection.
        """
        # For inflation, currency_code in BaseCurve will be the inflation index name itself
        super().__init__(inflation_index_name, reference_date_input)

        from src.config import config  # Import here to avoid circular dependency on first load
        self._convention_config = config.get_inflation_config(inflation_index_name)  #

        self._inflation_index_name = inflation_index_name  #
        self._ql_inflation_index_type = self._convention_config.get('ql_inflation_index')  #
        self._bbg_ticker_root = self._convention_config.get('bbg_ticker_root')  #
        self._bbg_fixing_ticker_root = self._convention_config.get('bbg_fixing_ticker_root')  #
        self._print_dates_str = self._convention_config.get('print_dates', [])  #
        self._dc_curve_index = self._convention_config.get('dc_curve_index')  #
        self._lag_months = self._convention_config.get('lag_months', 0)  #
        self._interp_method = self._convention_config.get('interp_method', 0)  #
        self._fixing_hist_file = self._convention_config.get('fixing_hist_file')  #
        self._seasonality_file = self._convention_config.get('seasonality_file')  #
        self._base_month_input = self._convention_config.get('base_month')  # # Base month as string

        self._base_month_offset = base_month_offset  #

        self._historical_fixings: pd.DataFrame = self._load_historical_fixings()  #
        self._seasonality_adjustments: pd.DataFrame = self._load_seasonality_adjustments()  #
        self._nominal_discount_curve: Optional[ql.YieldTermStructureHandle] = None
        self._inflation_zc_rates: pd.DataFrame = pd.DataFrame()  # Store raw inflation swap rates from market

        # Calculate dynamic dates based on reference date and conventions
        self._print_dates: pd.Series = self._parse_print_dates()  #
        self._base_month_ql: ql.Date = self._get_adjusted_base_month()  #
        self._last_market_fixing_month: ql.Date = self._get_last_market_fixing_month()  #

    def _parse_print_dates(self) -> pd.Series:
        """Parses print dates strings from conventions to datetime objects."""
        return pd.Series([datetime.datetime.strptime(d, '%Y-%m-%d') for d in self._print_dates_str])  #

    def _load_historical_fixings(self) -> pd.DataFrame:
        """Loads historical inflation index fixings from the data lake."""
        try:
            #
            return data_loader.load_pickle(self._fixing_hist_file)  #
        except FileNotFoundError:
            raise FileNotFoundError(f"Inflation historical fixings file '{self._fixing_hist_file}' not found.")  #
        except Exception as e:
            raise RuntimeError(f"Error loading historical fixings for {self._inflation_index_name}: {e}")  #

    def _load_seasonality_adjustments(self) -> pd.DataFrame:
        """Loads seasonality adjustments from the data lake."""
        try:
            #
            return data_loader.load_pickle(self._seasonality_file)  #
        except FileNotFoundError:
            raise FileNotFoundError(f"Inflation seasonality file '{self._seasonality_file}' not found.")  #
        except Exception as e:
            raise RuntimeError(f"Error loading seasonality for {self._inflation_index_name}: {e}")  #

    def _get_adjusted_base_month(self) -> ql.Date:
        """
        Calculates the adjusted base month for index projection based on the reference date
        and any specified offset.
        """
        if self._base_month_input:  #
            # Convert string base month to ql.Date, then apply offset.
            base_month_ql = datetime_to_ql(datetime.datetime.strptime(self._base_month_input, '%Y-%m-%d'))  #
            return base_month_ql + ql.Period(self._base_month_offset, ql.Months)  #
        else:
            # Fallback if no specific base month is provided, use reference date start of month.
            return self.reference_date - (self.reference_date.dayOfMonth() - 1)  #

    def _get_last_market_fixing_month(self) -> ql.Date:
        """
        Determines the last available inflation fixing month from historical data,
        adjusted for any relevant lag.
        """
        # Inferred from `c.last_fix_month` and usage in `Infl_ZC_Pricer`
        # This typically means the last actual published fixing.
        # Assuming `_historical_fixings` DataFrame has a 'months' column with ql.Date.
        if not self._historical_fixings.empty:  #
            return self._historical_fixings['months'].max()  #
        return self.reference_date  # Fallback

    def _fetch_market_data(self) -> None:
        """
        Fetches live inflation swap rates from Bloomberg.
        """
        ref_date_bbg = to_bbg_date_str(self.reference_date, ql_date=1)  #

        # Determine relevant tickers for inflation swaps based on conventions.
        # Example: 'USSWIT10 Curncy' for USCPI 10Y.
        # The naming convention is not perfectly standardized across all indices.
        #
        # NOTE: Verify the Bloomberg ticker construction for your inflation swaps.
        # The `bbg_ticker_root` from conventions.yaml is used here.
        # It's assumed the conventions file lists the full set of tickers or enough info to build them.
        # For 'HICPxT', original code implies `inf_tab['ticker'] = c.ticker`
        # and `inf_tab['maturity'] = [inf_tab['ticker'][i].split(' ')[0][len(inf_tab['ticker'][0].split(' ')[0])-1:]+'Y'`
        # This suggests `bbg_ticker_root` might be `USSWIT` or `UKRPI` etc. and tenors are appended.
        #
        inflation_swap_tickers_base = [f"{self._bbg_ticker_root}{i} Curncy" for i in
                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25,
                                        30]]  # Placeholder for common maturities
        # The original code adds a 31Y point for USCPI if needed. This is specific to the convention `a == 'USCPI'`.
        if self._inflation_index_name == 'USCPI':  #
            # This is hardcoded for USCPI in original, verify if 31Y is always needed.
            inflation_swap_tickers_base.append("USSWIT31 Curncy")  #

        try:
            # Fetch current inflation swap rates
            inflation_swap_data = b_con.bdh(inflation_swap_tickers_base, 'PX_LAST', ref_date_bbg, ref_date_bbg,
                                            longdata=True)  #

            # Pivot data to get rates per ticker
            inflation_swap_prices = inflation_swap_data.pivot(index='date', columns='ticker', values='value').iloc[
                0].reset_index()  #
            inflation_swap_prices.columns = ['Ticker', 'PX_LAST']  #

            # Prepare DataFrame for inflation rates with maturity info
            self._inflation_zc_rates = pd.DataFrame(columns=['ticker', 'maturity', 'px'])  #
            self._inflation_zc_rates['ticker'] = inflation_swap_prices['Ticker']  #
            self._inflation_zc_rates['px'] = inflation_swap_prices['PX_LAST']  #

            # Extract maturity from ticker (e.g., 'USSWIT10 Curncy' -> '10Y')
            self._inflation_zc_rates['maturity'] = self._inflation_zc_rates['ticker'].apply(
                lambda x: f"{int(re.search(r'(\d+)Y', x).group(1))}Y" if re.search(r'(\d+)Y', x) else None
            )  #
            self._inflation_zc_rates = self._inflation_zc_rates.dropna(subset=['maturity'])  #
            self._inflation_zc_rates['maturity_years'] = self._inflation_zc_rates['maturity'].apply(
                lambda x: int(x[:-1]))  #
            self._inflation_zc_rates = self._inflation_zc_rates.sort_values('maturity_years').reset_index(drop=True)  #

            self._market_data['inflation_swap_rates'] = self._inflation_zc_rates  # Store for later use

        except Exception as e:
            raise RuntimeError(f"Failed to fetch live inflation swap data for {self._inflation_index_name}: {e}")  #

    def _build_curve_helpers(self) -> List[ql.RateHelper]:
        """
        Constructs QuantLib rate helpers for the inflation curve.
        For inflation curves, QuantLib helpers are typically built from the inflation
        swap rates and a nominal zero-coupon curve (for discounting).
        """
        self._fetch_market_data()  # Ensure market data is loaded

        if self._inflation_zc_rates.empty:  #
            raise ValueError("No inflation swap rates available to build helpers.")  #

        # Ensure nominal discount curve is built (e.g., from OIS)
        if self._nominal_discount_curve is None:  #
            try:
                #
                ois_builder = OISCurveBuilder(self._dc_curve_index, self.reference_date)  #
                ois_builder.build_curve()  #
                self._nominal_discount_curve = ql.YieldTermStructureHandle(ois_builder.curve)  #
                print(f"Nominal discount curve ({self._dc_curve_index}) built for inflation curve.")  #
            except Exception as e:
                raise RuntimeError(f"Failed to build nominal discount curve for inflation curve: {e}")  #

        helpers: List[ql.InflationRateHelper] = []  #

        # Define QuantLib's inflation index object
        # This requires historical fixings, so use the `HistoricalSource` for the index.
        # This is a key part of QuantLib's inflation curve.
        # `base_month_ql` used for index projection.

        # Find the last historical fixing date and its value
        # The original code uses `dated_last_fix` which seems to be the month of the last fixing.
        last_hist_fixing_date = datetime_to_ql(
            self._print_dates[self._print_dates <= ql_to_datetime(self.reference_date)].tolist()[-1])  #
        last_hist_fixing_date_adj = self.calendar.advance(last_hist_fixing_date, ql.Period(-1, ql.Months))  #
        last_hist_fixing_date_adj = ql.Date(1, last_hist_fixing_date_adj.month(), last_hist_fixing_date_adj.year())  #

        initial_fixing_value = \
        self._historical_fixings[self._historical_fixings['months'] == last_hist_fixing_date_adj]['index'].iloc[0]  #
        #
        # The original code used a complex way to find the base index.
        # `base_index = inf_index_hist[inf_index_hist['months'] == base_month]['index'].tolist()[0]`
        # if `c.interp == 0`, else `get_infl_index`.
        # Here we use the actual base_month_ql which might not be an exact fixing date.

        # Use initial_fixing_value as the last known index value for the inflation term structure.
        # The nominal term structure (ois_discount_curve) is needed for inflation helpers.
        #
        # NOTE: `ql.CPI.EURHICP`, `ql.CPI.UKRPI`, `ql.CPI.USCPI` for actual index types.
        # The convention `ql_inflation_index` should provide the correct type.
        #
        try:  #
            ql_inflation_index_obj = getattr(ql.CPI, self._ql_inflation_index_type)(
                True,  # Interpolated/not
                self._nominal_discount_curve,  # Nominal term structure
                self._convention_config.get('interp_method') == 1  # Interpolated
            )
        except AttributeError:  #
            # Fallback for generic inflation index if not specific CPI
            ql_inflation_index_obj = ql.ZeroInflationIndex(
                self._inflation_index_name,  # Name
                self.currency,  # Currency
                self._interp_method == 1,  # Interpolated
                ql.Period(self._lag_months, ql.Months),  # Lag
                self.calendar,  # Calendar
                get_ql_business_day_convention(self._fixed_leg_convention['business_day_convention']),  #
                get_ql_day_counter(self._fixed_leg_convention['day_count'])  #
            )  #

        # Set historical fixings for the QuantLib inflation index
        # This is crucial for pricing and projecting.
        for _, row in self._historical_fixings.iterrows():  #
            ql_inflation_index_obj.addFixing(row['months'], row['index'])  #

        # Add initial fixing for the projection
        # This is where the curve is "anchored" to the last known inflation index value.
        ql_inflation_index_obj.addFixing(last_hist_fixing_date_adj, initial_fixing_value)  #

        # Loop through inflation swap rates to create helpers
        for _, row in self._inflation_zc_rates.iterrows():  #
            maturity_years = row['maturity_years']  #
            rate = row['px'] / 100  # Rate in decimal

            helpers.append(
                ql.ZeroInflationSwapHelper(
                    ql.QuoteHandle(ql.SimpleQuote(rate)),  # Quote for the inflation rate
                    ql.Period(maturity_years, ql.Years),  # Swap tenor
                    self.calendar,  # Calendar
                    get_ql_business_day_convention(self._fixed_leg_convention['business_day_convention']),
                    # Business day convention
                    get_ql_day_counter(self._fixed_leg_convention['day_count']),  # Day count
                    self._nominal_discount_curve,  # Nominal discount curve
                    ql_inflation_index_obj,  # Inflation index
                    get_ql_business_day_convention(self._fixed_leg_convention['business_day_convention']),
                    # Inflation business day convention (same as fixed leg for simplicity)
                    get_ql_day_counter(self._fixed_leg_convention['day_count']),
                    # Inflation day count (same as fixed leg for simplicity)
                    ql.Period(self._lag_months, ql.Months)  # Lag
                )
            )
        return helpers

    def build_curve(self) -> None:
        """
        Builds the QuantLib ZeroInflationTermStructure from the rate helpers.
        """
        helpers = self._build_curve_helpers()  #

        # Find the evaluation date (from QuantLib settings) and the calendar.
        evaluation_date = ql.Settings.instance().evaluationDate  #
        calendar = self.calendar  #

        try:
            # Build the ZeroInflationTermStructure.
            # Use `PiecewiseZeroInflationCurve` for flexible interpolation.
            # The base inflation index is implicitly taken from the helpers.
            self._ql_curve = ql.PiecewiseZeroInflationCurve(
                evaluation_date,  #
                calendar,  #
                get_ql_day_counter(self._fixed_leg_convention['day_count']),  #
                ql.Period(self._lag_months, ql.Months),  # Observation lag
                self._convention_config.get('ql_inflation_index_obj'),  # The inflation index object
                self._nominal_discount_curve,  # Nominal term structure
                helpers  # Rate helpers
            )
            self._ql_curve.enableExtrapolation()  #
            self._nodes = self._ql_curve.nodes()  #
            print(
                f"Inflation ZC curve for {self._inflation_index_name} built successfully on {ql_to_datetime(self.reference_date).strftime('%Y-%m-%d')}.")  #
        except Exception as e:
            raise RuntimeError(f"Failed to build inflation ZC curve for {self._inflation_index_name}: {e}")  #

    @property
    def inflation_zc_rates(self) -> pd.DataFrame:
        """
        Returns the raw inflation zero-coupon swap rates used for curve building.
        """
        return self._inflation_zc_rates  #

    @property
    def historical_fixings(self) -> pd.DataFrame:
        """
        Returns the loaded historical inflation index fixings.
        """
        return self._historical_fixings  #

    @property
    def seasonality_adjustments(self) -> pd.DataFrame:
        """
        Returns the loaded seasonality adjustments.
        """
        return self._seasonality_adjustments  #

    @property
    def base_month_ql(self) -> ql.Date:
        """
        Returns the QuantLib base month for index projection.
        """
        return self._base_month_ql  #

    @property
    def nominal_discount_curve(self) -> ql.YieldTermStructureHandle:
        """
        Returns the nominal discount curve used for inflation curve building.
        """
        return self._nominal_discount_curve  #