# src/pricers/inflation_pricer.py

import QuantLib as ql
import pandas as pd
import datetime
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.curves.inflation_curve_builder import InflationCurveBuilder  #
from src.curves.ois_curve_builder import OISCurveBuilder  # For nominal discount curve to calculate PV
from src.quantlib_utils import get_ql_date  #
from src.date_utils import ql_to_datetime  #


class InflationPricer:
    """
    Provides pricing and analytics for inflation zero-coupon (ZC) swaps.

    This class consolidates functionalities to calculate ZC rates, sensitivities (Inf01, Gamma01),
    and cross-sensitivities to nominal rates.
    """

    def __init__(self):
        """
        Initializes the InflationPricer.
        """
        pass  # No specific state to initialize for the pricer itself

    def calculate_zc_metrics(self,
                             inflation_curve_builder: InflationCurveBuilder,
                             start_date_input: Union[str, int, datetime.datetime, ql.Date],
                             tenor_years: int,
                             lag_months: Optional[int] = None,
                             notional: float = 1_000_000.0,
                             use_forecast_curve: bool = False,  # Corresponds to `use_forecast = 1`
                             use_market_fixing_curve: bool = True,  # Corresponds to `use_mkt_fixing = 1`
                             trade_date_input: Optional[Union[str, datetime.datetime, ql.Date]] = None,
                             # Corresponds to `trade_dt`
                             trade_zc_rate: float = 0.0  # Corresponds to `zc_rt`
                             ) -> Dict[str, Any]:
        """
        Calculates zero-coupon inflation rate and various sensitivities for an inflation swap.

        Args:
            inflation_curve_builder (InflationCurveBuilder): An already built InflationCurveBuilder object.
            start_date_input (Union[str, int, datetime.datetime, ql.Date]):
                The start date of the inflation swap. Can be 'QM' (Quarterly Month), '0D' (Spot),
                or a specific date string/object.
            tenor_years (int): The tenor of the inflation swap in years.
            lag_months (Optional[int]): The observation lag in months. If None, uses value from convention.
            notional (float): The notional amount of the inflation swap.
            use_forecast_curve (bool): If True, uses the forecast curve for projection (index 2).
            use_market_fixing_curve (bool): If True, uses the market fixing curve for projection (index 1).
            trade_date_input (Optional[Union[str, datetime.datetime, ql.Date]]):
                If provided, calculate PV/P&L metrics against this historical trade date.
            trade_zc_rate (float): The ZC rate at the trade date, for P&L calculation.

        Returns:
            Dict[str, Any]: A dictionary containing calculated metrics (ZC rate, Inf01, Gamma01, etc.).
        """
        inflation_curve_builder.build_curve()  # Ensure the inflation curve is built

        # Select the appropriate inflation curve based on flags
        inf_fixings_curves = inflation_curve_builder.curve  # A tuple of (seasonals, market_fixings, forecast)
        inf_fixings: pd.DataFrame = None
        last_fixing_month: ql.Date = None

        if use_forecast_curve:  #
            inf_fixings = inf_fixings_curves[2]  # Corresponds to `inf_curve.curve[2]`
            # `last_fixing_month` from original code for forecast curve was `inf_curve.fixing_hist['months'][-1:].tolist()[0]`
            last_fixing_month = inflation_curve_builder.historical_fixings['months'].max()  #
        elif use_market_fixing_curve:  #
            inf_fixings = inf_fixings_curves[1]  # Corresponds to `inf_curve.curve[1]`
            last_fixing_month = inflation_curve_builder.last_market_fixing_month  # Corresponds to `inf_curve.last_pm`
        else:  # Default to seasonals if neither forecast nor market fixing is chosen
            inf_fixings = inf_fixings_curves[0]  # Corresponds to `inf_curve.curve[0]`
            last_fixing_month = inflation_curve_builder.last_market_fixing_month  # Corresponds to `inf_curve.last_pm`

        if inf_fixings.empty:
            raise ValueError("Selected inflation fixing curve is empty. Cannot calculate ZC metrics.")

        # Determine start and end dates of the swap
        start_date_ql: ql.Date
        end_date_ql: ql.Date

        # Handle `trade_dt` (trade_date_input)
        if trade_date_input:  #
            trade_date_ql = get_ql_date(trade_date_input)  #
            # Assume trade settlement is T+1 Day after trade date for swap pricing
            start_date_ql = inflation_curve_builder.calendar.advance(trade_date_ql, 1, ql.Days)  #
            end_date_ql = inflation_curve_builder.calendar.advance(start_date_ql, tenor_years, ql.Years)  #
            # If trade_dt provided, nominal discount curve for PV needs to reflect that date
            trade_dc_builder = OISCurveBuilder(inflation_curve_builder.nominal_discount_curve.currentLink().name(),
                                               trade_date_input)  #
            trade_dc_builder.build_curve()  #
            nominal_discount_curve_for_pv = ql.YieldTermStructureHandle(trade_dc_builder.curve)  #
        else:
            # Handle start_date_input formats
            if isinstance(start_date_input, ql.Date):  #
                start_date_ql = start_date_input  #
            elif isinstance(start_date_input, str):  #
                # Try parsing as 'DD-MM-YYYY' or 'YYYY-MM-DD' date string
                try:
                    start_date_ql = get_ql_date(start_date_input)  #
                except ValueError:
                    # Handle tenor strings like '6M', '1Y' from settlement date
                    if start_date_input[-1].lower() in ('d', 'w', 'm', 'y'):  #
                        start_date_ql = inflation_curve_builder.nominal_discount_curve.currentLink().referenceDate() + ql.Period(
                            start_date_input)  #
                    else:
                        raise ValueError(f"Unsupported start_date_input string format: {start_date_input}")
            else:  # Interpret as years offset from settlement date (default to 0 if not specified)
                start_date_ql = inflation_curve_builder.nominal_discount_curve.currentLink().referenceDate() + ql.Period(
                    start_date_input, ql.Years)  #

            end_date_ql = inflation_curve_builder.calendar.advance(start_date_ql, tenor_years, ql.Years)  #
            nominal_discount_curve_for_pv = inflation_curve_builder.nominal_discount_curve  #

        # Calculate lagged start/end months for ZC rate calculation
        actual_lag_months = lag_months if lag_months is not None else inflation_curve_builder._lag_months  #

        # If `interp_method == 0`, use start of month. If `interp_method == 1`, use actual date.
        if inflation_curve_builder._interp_method == 0:  #
            start_month_ql = inflation_curve_builder.calendar.advance(start_date_ql, -actual_lag_months, ql.Months)  #
            start_month_ql = ql.Date(1, start_month_ql.month(), start_month_ql.year())  #
            end_month_ql = inflation_curve_builder.calendar.advance(start_month_ql, tenor_years, ql.Years)  #
            end_month_ql = ql.Date(1, end_month_ql.month(), end_month_ql.year())  #
        else:  # Interpolated case
            start_month_ql = start_date_ql - ql.Period(3, ql.Months)  # Inferred 3M lag for interpolation
            end_month_ql = start_month_ql + ql.Period(tenor_years, ql.Years)  #

        # Retrieve base fixing and index ratio from the selected inflation fixings curve
        base_month_fix = inf_fixings[inf_fixings['months'] == start_month_ql]['index'].iloc[0]  #
        index_ratio = (inf_fixings[inf_fixings['months'] == end_month_ql]['index'].iloc[0] / base_month_fix)  #

        zc_rate = 100 * ((index_ratio ** (1 / tenor_years)) - 1)  #

        # Calculate Inf01 (Risk01)
        try:  #
            df1_disc_factor = nominal_discount_curve_for_pv.discount(end_date_ql)  #
        except:  #
            df1_disc_factor = 1.0  # Fallback if discount curve fails

        inf01 = tenor_years * ((1 + (zc_rate / 100)) ** (tenor_years - 1)) * df1_disc_factor  #
        risk01 = inf01 * notional * 100  # Original `risk01` definition

        # Calculate Conv01 (Gamma01)
        conv01 = tenor_years * (tenor_years - 1) * (
                    (1 + (zc_rate / 100)) ** (tenor_years - 2)) * df1_disc_factor / 10000  #
        gamma01 = conv01 * notional * 100  # Original `gamma01` definition

        # Calculate Cross Gamma (sensitivity to nominal rates)
        # Bumping nominal curve by 1bp
        sh_curve = ql.ZeroSpreadedTermStructure(
            ql.YieldTermStructureHandle(nominal_discount_curve_for_pv.currentLink()),  #
            ql.QuoteHandle(ql.SimpleQuote(0.01 / 100))  # 1 bp spread
        )  #
        try:  #
            df2_disc_factor = sh_curve.discount(end_date_ql)  #
        except:  #
            df2_disc_factor = 1.0  # Fallback

        cross_g1 = (df2_disc_factor - df1_disc_factor) * (
                    inf01 / df1_disc_factor) * notional * 100  # Original `cross_g1` definition
        cross_g2 = (df2_disc_factor - df1_disc_factor) * (
                    conv01 / df1_disc_factor) * notional * 100  # Original `cross_g2` definition

        ir_delta = 0.0  # Placeholder, as `ir_delta` was only calculated in the `if zc_rt != 0` block

        pv_df = pd.DataFrame()  # Default empty DataFrame if no trade_zc_rate
        if trade_zc_rate != 0:  #
            # Calculate PV and P&L components if trade_zc_rate is provided
            pv = notional * (((1 + (zc_rate / 100)) ** tenor_years) - (
                        (1 + (trade_zc_rate / 100)) ** tenor_years)) * df1_disc_factor * 1000000  # Original PV formula

            inf01_at_trade = tenor_years * ((1 + (trade_zc_rate / 100)) ** (tenor_years - 1)) * df1_disc_factor  #
            conv01_at_trade = tenor_years * (tenor_years - 1) * (
                        (1 + (trade_zc_rate / 100)) ** (tenor_years - 2)) * df1_disc_factor / 10000  #

            zc_chg = 100 * (zc_rate - trade_zc_rate)  #
            inf_pnl = (inf01_at_trade * notional * 100 * (zc_chg)) + (
                        conv01_at_trade * notional * 100 * 0.5 * (zc_chg ** 2))  #

            # Need actual historical swap rate to calculate rate_chg, which is not readily available
            # This part of the original code relied on `Swap_Pricer` with historical curve,
            # which is complex to replicate precisely without more info.
            # `rate_chg = 100*(Swap_Pricer([[inf_curve.dc,0,tenor]] , fixed_leg_freq = 0).rate[0] - Swap_Pricer([[trade_curve, trade_dt, tenor]], fixed_leg_freq=0).rate[0])`
            # For simplicity, calculate `rate_chg` based on nominal curve shift here.

            # The exact `rate_chg` logic from original `SWAP_PRICER` is complex.
            # Assuming `rate_chg` is a 1bp change in nominal rates for cross gamma calculation.
            # This needs to be carefully aligned with `SwapPricer` if it represents actual change.
            rate_chg = 0.01  # Placeholder for a 1bp shift if needed

            rates_pnl = (cross_g1 * rate_chg * zc_chg) + (cross_g2 * rate_chg * (zc_chg ** 2) * 0.5)  # Original formula

            ir_delta = (pv / df1_disc_factor) * (df1_disc_factor - df2_disc_factor)  #

            pv_df = pd.DataFrame(columns=['value', 'chg', 'risk (incp.)', 'delta', 'gamma'],
                                 index=['PV', 'Inf', 'Rates', 'Residual'])  #
            pv_df['value'] = [pv, inf_pnl, rates_pnl, pv - (inf_pnl + rates_pnl)]  #
            pv_df['chg'] = ["", np.round(zc_chg, 1), np.round(rate_chg, 1), ""]  #
            pv_df['risk (incp.)'] = ["", np.round(inf01_at_trade * notional * 100, 1), np.round(cross_g1, 1), ""]  #
            pv_df['delta'] = ["", np.round(risk01, 1), np.round(ir_delta, 1), ""]  #
            pv_df['gamma'] = ["", np.round(gamma01, 1), np.round(cross_g1, 1), ""]  #
            pv_df = pv_df.round(1)  #
        else:
            pv_df = pd.DataFrame(columns=['delta', 'gamma'], index=['Inf', 'Rates'])  #
            pv_df['delta'] = [np.round(risk01, 0), np.round(ir_delta, 0)]  #
            pv_df['gamma'] = [np.round(gamma01, 0), np.round(cross_g1, 0)]  #

        return {
            "base_month": start_month_ql,  #
            "base_fixing": base_month_fix,  #
            "index_name": inflation_curve_builder._inflation_index_name,  #
            "zc_rate": round(zc_rate, 3),  #
            "interp_method": inflation_curve_builder._interp_method,  #
            "last_fixing_month": last_fixing_month,  #
            "inf01": round(inf01, 1),  #
            "conv01": round(conv01, 1),  #
            "inf_risk": round(risk01, 1),  #
            "inf_gamma": round(gamma01, 1),  #
            "cross_gamma": round(cross_g1, 1),  #
            "cross_gamma_inf_conv": round(cross_g2, 1),  #
            "rates_risk": round(ir_delta, 1),  #
            "pv_table": pv_df  #
        }