# src/pricers/linker_carry_calculator.py

import QuantLib as ql
import pandas as pd
import datetime
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.instruments.bond import Bond  #
from src.curves.inflation_curve_builder import InflationCurveBuilder  #
from src.data_manager import b_con  #
from src.date_utils import to_bbg_date_str, datetime_to_ql  #
from src.financial_math_utils import get_infl_index  #


class LinkerCarryCalculator:
    """
    Calculates the carry components for inflation-linked bonds (linkers).
    """

    def __init__(self):
        """
        Initializes the LinkerCarryCalculator.
        """
        pass  # No specific state to initialize

    def calculate_linker_carry(self,
                               isin: str,
                               inflation_index_name: str,
                               currency_code: str,
                               repo_rate: float,
                               fixing_curve_type: str = 'Market',  # 'BARX', 'Market', 'Seasonals'
                               forward_dates: Optional[List[Union[str, datetime.datetime, ql.Date]]] = None
                               ) -> Dict[str, Any]:
        """
        Calculates the carry components for a single inflation-linked bond.

        Args:
            isin (str): The ISIN of the inflation-linked bond.
            inflation_index_name (str): The name of the inflation index associated with the bond.
            currency_code (str): The currency code of the bond.
            repo_rate (float): The repo rate (as a decimal, e.g., 0.005 for 0.5%).
            fixing_curve_type (str): Specifies which inflation curve to use for index projection:
                                     'BARX' (forecast curve index 2), 'Market' (market curve index 1),
                                     or 'Seasonals' (seasonality curve index 0).
            forward_dates (Optional[List[Union[str, datetime.datetime, ql.Date]]]):
                A list of future dates for which to calculate forward clean prices and yields.
                If None, generates 3 month-end dates from the current reference date.

        Returns:
            Dict[str, Any]: A dictionary containing carry components and forward yields.
        """
        # Determine which inflation curve to use based on fixing_curve_type
        if fixing_curve_type == 'BARX':  #
            forecast_index = 2  #
            use_forecast = True  #
            use_market_fixing = False  #
        elif fixing_curve_type == 'Market':  #
            forecast_index = 1  #
            use_forecast = False  #
            use_market_fixing = True  #
        elif fixing_curve_type == 'Seasonals':  #
            forecast_index = 0  #
            use_forecast = False  #
            use_market_fixing = False  #
        else:
            raise ValueError(
                f"Unknown fixing_curve_type: {fixing_curve_type}. Choose 'BARX', 'Market', or 'Seasonals'.")  #

        # Build the inflation curve
        # Use a historical reference date (e.g., -1 day) for `fix_c` as per original logic
        inflation_builder = InflationCurveBuilder(inflation_index_name, reference_date_input=-1)  #
        inflation_builder.build_curve()  #

        # Get the appropriate inflation fixings dataframe
        inf_fixings_curve_df = inflation_builder.curve[forecast_index]  #

        # Determine forward dates
        if forward_dates is None:  #
            fwd_dates_ql = [ql.Date.endOfMonth(inflation_builder.reference_date) + ql.Period(i, ql.Months) for i in
                            range(3)]  #
        else:
            fwd_dates_ql = [get_ql_date(d) for d in forward_dates]  #

        # Instantiate Bond instrument to fetch its attributes
        bond = Bond(isin, currency_code, valuation_date_input=inflation_builder.reference_date)  #

        # --- Fetch bond attributes (already done by Bond.__init__) ---
        # The following attributes are accessed from `bond.get_attribute()`:
        # 'MATURITY', 'CPN', 'CPN_FREQ', 'DAYS_ACC', 'DAYS_TO_NEXT_COUPON',
        # 'PX_LAST', 'PX_CLOSE_1D', 'SETTLE_DT', 'PX_CLOSE_DT', 'DAYS_TO_SETTLE', 'BASE_CPI'

        # --- Calculate Spot and Close Dirty Prices ---
        spot_settle_dt_ql = datetime_to_ql(bond.get_attribute('SETTLE_DT')) if isinstance(
            bond.get_attribute('SETTLE_DT'), datetime.datetime) else bond.get_attribute('SETTLE_DT')  #
        close_date_ql = datetime_to_ql(bond.get_attribute('PX_CLOSE_DT')) if isinstance(
            bond.get_attribute('PX_CLOSE_DT'), datetime.datetime) else bond.get_attribute('PX_CLOSE_DT')  #

        # Use `DAYS_TO_SETTLE` from Bloomberg to get close settlement date
        close_settle_dt_ql = inflation_builder.calendar.advance(close_date_ql, bond.get_attribute('DAYS_TO_SETTLE'),
                                                                ql.Days)  #

        # Calculate accrued coupon for close and spot dates
        coupon_rate = bond.get_attribute('CPN')  #
        coupon_freq = bond.get_attribute('CPN_FREQ')  #
        days_acc_close_ref = bond.get_attribute('DAYS_ACC') - int(
            spot_settle_dt_ql - close_settle_dt_ql)  # Adjusted days accrued for close
        days_acc_spot_ref = bond.get_attribute('DAYS_ACC')  #
        total_coupon_days = bond.get_attribute('DAYS_ACC') + bond.get_attribute('DAYS_TO_NEXT_COUPON')  #

        coupon_acc_close = (coupon_rate * (1 / coupon_freq) * days_acc_close_ref) / total_coupon_days  #
        coupon_acc_spot = (coupon_rate * (1 / coupon_freq) * days_acc_spot_ref) / total_coupon_days  #

        # Get inflation index ratios for close and spot settlement dates
        base_cpi = bond.get_attribute('BASE_CPI')  #
        ref_index_close = get_infl_index(inf_fixings_curve_df, close_settle_dt_ql)  #
        idx_ratio_close = round(ref_index_close / base_cpi, 5)  #

        ref_index_spot = get_infl_index(inf_fixings_curve_df, spot_settle_dt_ql)  #
        idx_ratio_spot = round(ref_index_spot / base_cpi, 5)  #

        # Calculate dirty prices for close and spot
        close_clean_px = bond.get_attribute('PX_CLOSE_1D')  #
        spot_clean_px = bond.get_attribute('PX_LAST')  #

        dp_cls = (close_clean_px + coupon_acc_close) * idx_ratio_close  # Dirty price at close
        dp_spot = (spot_clean_px + coupon_acc_spot) * idx_ratio_spot  # Dirty price at spot

        # --- Calculate Repo Carry from Close to Spot ---
        repo_close = (dp_cls * repo_rate) * int(spot_settle_dt_ql - close_settle_dt_ql) / 36000  #
        spot_clean_px_from_close = ((dp_cls + repo_close) / idx_ratio_spot) - coupon_acc_spot  #

        # --- Calculate Forward Clean Prices and Yields ---
        fwd_index_ratios = []
        fwd_cpn_accs = []
        repos = []
        fwd_clean_pxs = []

        for fwd_dt_ql in fwd_dates_ql:  #
            fwd_index = get_infl_index(inf_fixings_curve_df, fwd_dt_ql)  #
            fwd_index_ratios.append(round(fwd_index / base_cpi, 5))  #

            days_acc_fwd_ref = days_acc_spot_ref + int(fwd_dt_ql - spot_settle_dt_ql)  #
            fwd_cpn_accs.append((coupon_rate * (1 / coupon_freq) * days_acc_fwd_ref) / total_coupon_days)  #

            repos.append((dp_spot * repo_rate) * int(fwd_dt_ql - spot_settle_dt_ql) / 36000)  #
            fwd_clean_pxs.append(((dp_spot + repos[-1]) / fwd_index_ratios[-1]) - fwd_cpn_accs[-1])  #

        # Calculate yields for various points using Bloomberg overrides
        close_yield = bond.calculate_yield_from_price(close_clean_px, settlement_date=close_settle_dt_ql)  #
        spot_yield_from_close = bond.calculate_yield_from_price(spot_clean_px_from_close,
                                                                settlement_date=spot_settle_dt_ql)  #
        spot_yield = bond.calculate_yield_from_price(spot_clean_px, settlement_date=spot_settle_dt_ql)  #
        fwd_yields = [bond.calculate_yield_from_price(px, settlement_date=fwd_dt_ql) for px, fwd_dt_ql in
                      zip(fwd_clean_pxs, fwd_dates_ql)]  #

        # --- Calculate Carry Components ---
        carry_components = {  #
            "Carry": 100 * (spot_yield_from_close - close_yield)  # Original `linker_carry_calc` carry calculation
        }  #

        # Add forward carry components
        if fwd_yields:  #
            carry_components[f"Carry_{ql_to_datetime(fwd_dates_ql[0]).strftime('%b%y')}"] = 100 * (
                        fwd_yields[0] - spot_yield)  #
            for i in range(1, len(fwd_yields)):  #
                carry_components[f"Carry_{ql_to_datetime(fwd_dates_ql[i]).strftime('%b%y')}"] = 100 * (
                            fwd_yields[i] - fwd_yields[i - 1])  #

        return carry_components