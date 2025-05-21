# src/market_tables/inflation_tables.py

import pandas as pd
import numpy as np
import QuantLib as ql
from typing import Any, Dict, List, Optional, Union
from src.curves.inflation_curve_builder import InflationCurveBuilder  #
from src.pricers.inflation_pricer import InflationPricer  #
from src.date_utils import ql_to_datetime  #
from src.financial_math_utils import flat_lst  #


class InflationTableGenerator:
    """
    Generates various tables for inflation swap rates, including outrights, forwards,
    curves, and butterflies, and their changes over time.
    """

    def __init__(self):
        """
        Initializes the InflationTableGenerator.
        """
        self.inflation_pricer = InflationPricer()  #

    def generate_inflation_swap_table(self,
                                      inflation_curve_builders: List[InflationCurveBuilder],  # Original `crvs`
                                      lags_months: List[int],  # Original `lag`
                                      outright_tenors_years: List[int],  # Original `outright_rates`
                                      forward_rates_tenors: List[List[int]],
                                      # Original `fwd_rates` as [[start_yr, tenor_yr]]
                                      curve_tenors: List[List[int]],  # Original `curve_rates` as [[short_yr, long_yr]]
                                      fly_tenors: List[List[int]],
                                      # Original `fly_rates` as [[short_yr, mid_yr, long_yr]]
                                      shift_offsets: List[Union[int, str]] = [0, '1M', '2M', '3M'],  # Original `shift`
                                      price_nodes: bool = False,  # Original `price_nodes`
                                      use_forecast: bool = False,  # Original `use_forecast`
                                      use_market_fixing: bool = True  # Original `use_mkt_fixing`
                                      ) -> Dict[str, pd.DataFrame]:
        """
        Generates a comprehensive set of inflation swap tables.

        Args:
            inflation_curve_builders (List[InflationCurveBuilder]): List of built inflation curve builders
                                                                    (e.g., for current and historical dates).
            lags_months (List[int]): List of observation lags (in months) corresponding to each curve builder.
            outright_tenors_years (List[int]): List of outright maturities in years (e.g., [1, 5, 10]).
            forward_rates_tenors (List[List[int]]): List of forward swap tenors,
                                                    e.g., [[1, 1], [2, 1]] for 1y1y, 2y1y.
            curve_tenors (List[List[int]]): List of curve spreads,
                                            e.g., [[2, 10], [5, 30]] for 2s10s, 5s30s.
            fly_tenors (List[List[int]]): List of butterfly spreads,
                                          e.g., [[2, 5, 10]] for 2s5s10s.
            shift_offsets (List[Union[int, str]]): List of date shifts for historical comparisons (e.g., [0, -1, '1M']).
            price_nodes (bool): If True, includes monthly fixing price nodes table.
            use_forecast (bool): If True, inflation pricer uses forecast curve.
            use_market_fixing (bool): If True, inflation pricer uses market fixing curve.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing various generated tables:
                                     'outright_zc_rates', 'forward_zc_rates',
                                     'curve_rates', 'fly_rates', 'fixing_nodes' (if price_nodes=True).
        """
        n_curves = len(inflation_curve_builders)

        # --- Outright ZC Rates ---
        zc_outright_data = []
        for tenor in outright_tenors_years:
            rates_for_tenor = []
            for i in range(n_curves):
                metrics = self.inflation_pricer.calculate_zc_metrics(
                    inflation_curve_builders[i],
                    start_date_input=0,  # Spot starting
                    tenor_years=tenor,
                    lag_months=lags_months[i],
                    use_forecast_curve=use_forecast,
                    use_market_fixing_curve=use_market_fixing
                )
                rates_for_tenor.append(metrics['zc_rate'])
            zc_outright_data.append(rates_for_tenor)

        # Flatten and format for DataFrame
        outright_zc_df = pd.DataFrame({'Tenor': outright_tenors_years})
        outright_zc_df['ZC'] = [data[0] for data in zc_outright_data]
        for i in range(1, n_curves):
            outright_zc_df[f'Δ{i}'] = [100 * (data[0] - data[i]) for data in zc_outright_data]
            outright_zc_df[f'Δ{i}'] = outright_zc_df[f'Δ{i}'].round(1)

        # --- Forward ZC Rates ---
        fwd_zc_data = []
        for start_yr, tenor_yr in forward_rates_tenors:
            rates_for_fwd = []
            for i in range(n_curves):
                metrics = self.inflation_pricer.calculate_zc_metrics(
                    inflation_curve_builders[i],
                    start_date_input=start_yr,  # Start from start_yr years forward
                    tenor_years=tenor_yr,
                    lag_months=lags_months[i],
                    use_forecast_curve=use_forecast,
                    use_market_fixing_curve=use_market_fixing
                )
                rates_for_fwd.append(metrics['zc_rate'])
            fwd_zc_data.append(rates_for_fwd)

        fwd_zc_df = pd.DataFrame({'Fwds': [f"{s} x {t}" for s, t in forward_rates_tenors]})
        fwd_zc_df['Fwd.ZC'] = [data[0] for data in fwd_zc_data]
        for i in range(1, n_curves):
            fwd_zc_df[f'Δ.{i}'] = [100 * (data[0] - data[i]) for data in fwd_zc_data]
            fwd_zc_df[f'Δ.{i}'] = fwd_zc_df[f'Δ.{i}'].round(1)

        # --- Curve Rates (Spreads) ---
        curve_data = []
        for offset_val in shift_offsets:
            for short_yr, long_yr in curve_tenors:
                rates_for_curve = []
                for i in range(n_curves):
                    short_rate_metrics = self.inflation_pricer.calculate_zc_metrics(
                        inflation_curve_builders[i],
                        start_date_input=offset_val,
                        tenor_years=short_yr,
                        lag_months=lags_months[i],
                        use_forecast_curve=use_forecast,
                        use_market_fixing_curve=use_market_fixing
                    )
                    long_rate_metrics = self.inflation_pricer.calculate_zc_metrics(
                        inflation_curve_builders[i],
                        start_date_input=offset_val,
                        tenor_years=long_yr,
                        lag_months=lags_months[i],
                        use_forecast_curve=use_forecast,
                        use_market_fixing_curve=use_market_fixing
                    )
                    rates_for_curve.append(
                        100 * (long_rate_metrics['zc_rate'] - short_rate_metrics['zc_rate']))  # Long minus Short
                curve_data.append(rates_for_curve)

        curve_df = pd.DataFrame({'Curves': [f"{s} - {l}" for s, l in curve_tenors]})
        curve_df['Rate'] = [data[0] for data in curve_data[::len(shift_offsets)]]  # Rates for the first shift offset
        curve_df['Rate'] = curve_df['Rate'].round(1)
        for i in range(1, n_curves):
            curve_df[f'Δ~{i}'] = [data[i] for data in curve_data[::len(shift_offsets)]]
            curve_df[f'Δ~{i}'] = curve_df[f'Δ~{i}'].round(1)
        for i, offset_val in enumerate(shift_offsets[1:]):  # Shifts relative to first shift_offset
            curve_df[str(offset_val)] = [data[0] for data in curve_data[i + 1::len(shift_offsets)]]
            curve_df[str(offset_val)] = (curve_df[str(offset_val)] - curve_df['Rate']).round(
                1)  # Difference from "current" rate

        # --- Fly Rates (Butterflies) ---
        fly_data = []
        for offset_val in shift_offsets:
            for short_yr, mid_yr, long_yr in fly_tenors:
                rates_for_fly = []
                for i in range(n_curves):
                    short_rate_metrics = self.inflation_pricer.calculate_zc_metrics(
                        inflation_curve_builders[i],
                        start_date_input=offset_val,
                        tenor_years=short_yr,
                        lag_months=lags_months[i],
                        use_forecast_curve=use_forecast,
                        use_market_fixing_curve=use_market_fixing
                    )
                    mid_rate_metrics = self.inflation_pricer.calculate_zc_metrics(
                        inflation_curve_builders[i],
                        start_date_input=offset_val,
                        tenor_years=mid_yr,
                        lag_months=lags_months[i],
                        use_forecast_curve=use_forecast,
                        use_market_fixing_curve=use_market_fixing
                    )
                    long_rate_metrics = self.inflation_pricer.calculate_zc_metrics(
                        inflation_curve_builders[i],
                        start_date_input=offset_val,
                        tenor_years=long_yr,
                        lag_months=lags_months[i],
                        use_forecast_curve=use_forecast,
                        use_market_fixing_curve=use_market_fixing
                    )
                    # Fly formula: (2 * Mid - Short - Long)
                    rates_for_fly.append(100 * (
                                (2 * mid_rate_metrics['zc_rate']) - short_rate_metrics['zc_rate'] - long_rate_metrics[
                            'zc_rate']))
                fly_data.append(rates_for_fly)

        fly_df = pd.DataFrame({'Curves': [f"{s}.{m}.{l}" for s, m, l in fly_tenors]})
        fly_df['Rate'] = [data[0] for data in fly_data[::len(shift_offsets)]]  # Rates for the first shift offset
        fly_df['Rate'] = fly_df['Rate'].round(1)
        for i in range(1, n_curves):
            fly_df[f'Δ~{i}'] = [data[i] for data in fly_data[::len(shift_offsets)]]
            fly_df[f'Δ~{i}'] = fly_df[f'Δ~{i}'].round(1)
        for i, offset_val in enumerate(shift_offsets[1:]):
            fly_df[str(offset_val)] = [data[0] for data in fly_data[i + 1::len(shift_offsets)]]
            fly_df[str(offset_val)] = (fly_df[str(offset_val)] - fly_df['Rate']).round(
                1)  # Difference from "current" rate

        # --- Fixing Nodes Table (if price_nodes is True) ---
        fixing_nodes_df = pd.DataFrame()
        if price_nodes:
            # Assumes the first curve builder is the "current" one
            current_curve_builder = inflation_curve_builders[0]

            # This logic replicates the `x_fixing` and related calculations from original `inf_swap_table`
            # which computes monthly fixings from the `last_pm` (last market fixing month) and then compares.
            #
            # NOTE: Ticker construction for inflation fixings (e.g., CPINEMU, UKRPI, CPI)
            # is based on `bbg_fixing_ticker_root` in conventions.yaml.
            # The original code had special handling for USCPI (e.g., using 'F' and 'H' tickers).
            # This logic needs to be verified based on your Bloomberg exact usage.
            #
            last_pm = current_curve_builder.last_market_fixing_month
            x_fixing_ql_dates = [last_pm + ql.Period(i, ql.Months) for i in range(13)]  # 13 months for 1 year ahead

            x2_mkt_data = []
            x2_barx_data = []  # Forecast curve (index 2) is "BARX" in original

            for fix_ql_date in x_fixing_ql_dates:
                # Market fixings (use_market_fixing = True)
                mkt_metrics = self.inflation_pricer.calculate_zc_metrics(
                    current_curve_builder,
                    start_date_input=fix_ql_date - ql.Period(1, ql.Years),  # 1Y rate ending at `fix_ql_date`
                    tenor_years=1,  # 1-year rate
                    lag_months=current_curve_builder._lag_months,
                    use_forecast_curve=False,
                    use_market_fixing_curve=True
                )
                x2_mkt_data.append(mkt_metrics['zc_rate'])

                # Barx fixings (use_forecast = True)
                barx_metrics = self.inflation_pricer.calculate_zc_metrics(
                    current_curve_builder,
                    start_date_input=fix_ql_date - ql.Period(1, ql.Years),
                    tenor_years=1,
                    lag_months=current_curve_builder._lag_months,
                    use_forecast_curve=True,
                    use_market_fixing_curve=False
                )
                x2_barx_data.append(barx_metrics['zc_rate'])

            fixing_nodes_df['Months'] = [ql_to_datetime(d).strftime('%b-%y') for d in x_fixing_ql_dates]
            fixing_nodes_df['Barcap'] = x2_barx_data
            fixing_nodes_df['Mkt'] = x2_mkt_data
            if n_curves > 1:  # Calculate change if more than one historical curve is provided
                # Assuming `inflation_curve_builders[1]` is the historical comparison curve
                prev_curve_builder = inflation_curve_builders[1]
                x2_mkt_prev_data = []
                for fix_ql_date in x_fixing_ql_dates:
                    mkt_metrics_prev = self.inflation_pricer.calculate_zc_metrics(
                        prev_curve_builder,
                        start_date_input=fix_ql_date - ql.Period(1, ql.Years),
                        tenor_years=1,
                        lag_months=prev_curve_builder._lag_months,
                        use_forecast_curve=False,
                        use_market_fixing_curve=True
                    )
                    x2_mkt_prev_data.append(mkt_metrics_prev['zc_rate'])
                fixing_nodes_df['Chg'] = [100 * (mkt - prev_mkt) for mkt, prev_mkt in
                                          zip(x2_mkt_data, x2_mkt_prev_data)]
                fixing_nodes_df['Chg'] = fixing_nodes_df['Chg'].round(1)
            else:
                fixing_nodes_df['Chg'] = np.nan  # No change if only one curve

        # Combine results into a dictionary
        return {
            "outright_zc_rates": outright_zc_df,
            "forward_zc_rates": fwd_zc_df,
            "curve_rates": curve_df,
            "fly_rates": fly_df,
            "fixing_nodes": fixing_nodes_df if price_nodes else pd.DataFrame()
        }