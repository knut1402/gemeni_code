# src/pricers/swap_pricer.py

import QuantLib as ql
import pandas as pd
import datetime
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.instruments.swap import Swap, SwapParameters  #
from src.curves.swap_curve_builder import SwapCurveBuilder  #
from src.curves.ois_curve_builder import OISCurveBuilder  # For quick_swap, if not using a builder directly
from src.quantlib_utils import get_ql_date, get_ql_business_day_convention, get_ql_day_counter, get_ql_frequency  #
from src.date_utils import ql_to_datetime  #


class SwapPricer:
    """
    Provides comprehensive pricing and analytics for interest rate swaps.

    This class consolidates functionalities to calculate fair rates, NPV, DV01,
    as well as derive forward swap curves, spreads, and butterflies.
    """

    def __init__(self):
        """
        Initializes the SwapPricer.
        """
        pass  # No specific state to initialize for the pricer itself

    def calculate_swap_metrics(self, swap_params_list: List[SwapParameters], fixed_leg_freq: Optional[str] = None) -> \
    Dict[str, Any]:
        """
        Calculates various metrics (fair rate, DV01, NPV, risk) for a list of swaps.
        Can also compute spreads and butterflies if multiple swaps are provided.

        Args:
            swap_params_list (List[SwapParameters]): A list of SwapParameters dataclass objects.
            fixed_leg_freq (Optional[str]): Optional override for the fixed leg frequency (e.g., 'Semiannual').

        Returns:
            Dict[str, Any]: A dictionary containing calculated metrics,
                            including a summary DataFrame.
        """
        output_name = []
        output_rate = []
        output_dv01 = []
        output_risk = []  # Corresponds to original 'output_01'
        output_npv_total = 0.0
        output_start_date = []
        output_end_date = []

        summary_table = pd.DataFrame()

        for params in swap_params_list:
            try:
                # Instantiate Swap instrument
                swap_instrument = Swap(params, valuation_date_input=ql.Settings.instance().evaluationDate)

                # Apply fixed leg frequency override if provided
                if fixed_leg_freq:
                    swap_instrument.params.fixed_leg_freq = fixed_leg_freq

                fair_rate = swap_instrument.calculate_fair_rate()
                npv = swap_instrument.calculate_npv()
                dv01 = swap_instrument.calculate_dv01()  # Assumes default 0.5 bps bump

                output_name.append(f"{params.start_tenor} x {params.maturity_tenor}")
                output_rate.append(fair_rate)
                output_dv01.append(dv01)
                output_npv_total += npv
                output_risk.append(
                    round(dv01 * params.notional / 10000, 0))  # Original 'output_01' was DV01 * N / 10000

                # Retrieve dates after swap instrument setup
                # These attributes are not directly available in the original `swap_pricer_output`
                # but are implied by `output_dates`.
                start_date = swap_instrument.ql_instrument.startDate()  # Inferred
                end_date = swap_instrument.ql_instrument.maturityDate()  # Inferred
                output_start_date.append(start_date)
                output_end_date.append(end_date)

            except Exception as e:
                print(
                    f"Error calculating metrics for swap {params.currency_code} {params.start_tenor}x{params.maturity_tenor}: {e}")
                # Append NaN or appropriate placeholder for failed calculations
                output_name.append(f"{params.start_tenor} x {params.maturity_tenor} (Error)")
                output_rate.append(np.nan)
                output_dv01.append(np.nan)
                output_risk.append(np.nan)
                output_start_date.append(None)
                output_end_date.append(None)

        summary_table['Rate'] = output_rate
        summary_table['dv01'] = output_dv01
        summary_table['risk'] = output_risk
        summary_table.index = output_name

        # Calculate spread and fly if applicable
        if len(output_rate) >= 2:
            summary_table['Spread'] = summary_table['Rate'].diff() * 100
        if len(output_rate) >= 3:
            # Original formula for fly: `100*(2*self.rate[1] - (self.rate[0] + self.rate[2]))`
            # This is `(Rate_Mid - Rate_Short) - (Rate_Long - Rate_Mid)`
            # The original `summary_table['Fly'] = output_table['Spread'].diff()*-1`
            # implies `(Spread_Long - Spread_Short) * -1`.
            # Spread_Long is Rate_Long - Rate_Mid. Spread_Short is Rate_Mid - Rate_Short.
            # So `(Rate_Long - Rate_Mid) - (Rate_Mid - Rate_Short) = Rate_Long - 2*Rate_Mid + Rate_Short`.
            # Multiplied by -1 for original code: `-(Rate_Long - 2*Rate_Mid + Rate_Short) = 2*Rate_Mid - Rate_Long - Rate_Short`.
            # This matches `2*self.rate[1] - (self.rate[0] + self.rate[2])`.
            summary_table['Fly'] = summary_table['Spread'].diff() * -1  #

        summary_table = summary_table.fillna("")  #

        output_dates_df = pd.DataFrame({
            'Start': output_start_date,
            'End': output_end_date
        }, index=output_name)

        return {
            "name": output_name,
            "dates": output_dates_df,
            "rate": output_rate,
            "dv01": output_dv01,
            "risk": output_risk,
            "npv": round(output_npv_total, 0),
            "table": summary_table,
            "spread": summary_table['Spread'].iloc[-1] if 'Spread' in summary_table.columns and not summary_table[
                'Spread'].empty else np.nan,  # Inferred from original `self.spread`
            "fly": summary_table['Fly'].iloc[-1] if 'Fly' in summary_table.columns and not summary_table[
                'Fly'].empty else np.nan  # Inferred from original `self.fly`
        }

    def calculate_forward_curve(self,
                                currency_code: str,
                                instruments: List[List[Union[int, str]]],
                                # [[start_yr, tenor_yr], ...] or [[tenor_yr], ...]
                                ratios: List[float],  # For spreads/butterflies
                                end_fwd_start_years: int = 10,
                                interval_years: float = 1.0,
                                fixed_leg_freq: Optional[str] = None
                                ) -> Dict[str, Any]:
        """
        Calculates a forward swap curve based on specified instruments and ratios.

        Args:
            currency_code (str): The currency code for the curve.
            instruments (List[List[Union[int, str]]]): List of instrument definitions.
                Each inner list can be `[start_tenor_in_years, maturity_tenor_in_years]`
                for forward-starting swaps, or `[maturity_tenor_in_years]` for spot-starting.
            ratios (List[float]): Ratios for combining the rates (e.g., [1, -2, 1] for a butterfly).
            end_fwd_start_years (int): The end year for the forward start dates.
            interval_years (float): The interval in years for generating forward start dates.
            fixed_leg_freq (Optional[str]): Optional override for the fixed leg frequency.

        Returns:
            Dict[str, Any]: A dictionary containing the forward rates and their carry.
        """
        ql_eval_date = ql.Settings.instance().evaluationDate  #

        start_tenors_for_curve = np.arange(0, end_fwd_start_years + interval_years, interval_years)

        rates_per_instrument: Dict[int, List[float]] = {}

        # Build a curve builder for the underlying curve once
        # For simplicity, assuming a SwapCurveBuilder is appropriate here.
        # Original code used `crv` (a curve object, potentially OIS or Swap).
        # We need to explicitly build the curve to pass to Swap.
        curve_builder = SwapCurveBuilder(currency_code, ql_eval_date)  #
        curve_builder.build_curve()  #

        for k, inst_def in enumerate(instruments):
            rates_per_instrument[k] = []

            for start_yr_offset in start_tenors_for_curve:
                # Define start and maturity for each forward swap
                if len(inst_def) == 1:
                    # Spot starting swap, e.g., 0x10Y (tenor is maturity_tenor_in_years)
                    start_tenor = start_yr_offset  # Offset from current valuation date
                    maturity_tenor = inst_def[0]  # The outright maturity in years
                elif len(inst_def) == 2:
                    # Forward starting swap, e.g., 5Yx10Y
                    start_tenor = f"{int(start_yr_offset)}Y" if start_yr_offset > 0 else 0  # Start tenor from spot
                    maturity_tenor = inst_def[1]  # Maturity of the forward leg (e.g., 10Y leg in a 5Yx10Y)

                    # If inst_def is like [start_yr_offset, tenor_length]
                    # The original `sw.mt` was `inst[k][0]` or `inst[k][1]`.
                    # Here, assuming `inst_def[0]` is the start tenor for the forward, and `inst_def[1]` is the tenor length.
                    # The example `inst = [[2],[5],[10]]` in `Swap_curve_fwd` implies `inst_def[0]` is maturity.
                    # Re-evaluating `Swap_curve_fwd` for `inst = [[2],[5],[10]]`: `sw.mt` takes `inst[k][0]`.
                    # So `inst_def` is `[tenor_length]` for spot-starting.
                    # If `inst_def` is `[start_fwd_yr, tenor_length]`, then `sw.st` is `start_fwd_yr`.
                    # The original `Swap_curve_fwd(crv, inst, ...)` takes `inst = [[2],[5],[10]]`
                    # or `inst = [[2,3], [5,10]]` etc.
                    # So `inst_def` is either `[Maturity]` or `[ForwardStartYear, MaturityLength]`

                    # Re-aligning with `Swap_Pricer`'s `SwapParameters` and original `Swap_curve_fwd` usage:
                    # `sw = swap_class(custom_sw_class_index, 0, inst[k][0], ...)` for spot-starting
                    # `sw = swap_class(custom_sw_class_index, inst[k][0], inst[k][1], ...)` for forward-starting
                    # `inst_def` is `[maturity_tenor]` or `[start_tenor, maturity_tenor_length]`.
                    # If `inst_def` contains `start_tenor_length`, then `maturity_tenor` is `start_tenor_length + tenor_length`.
                    # Let's assume `instruments` is a list of [maturity_years] for spot, or [fwd_start_years, fwd_maturity_years]

                    # For `inst = [[2],[5],[10]]` in `Swap_curve_fwd`: `inst_def` is `[maturity_years]`.
                    # `sw.st = 0`, `sw.mt = inst[k][0]`.
                    # For `inst = [[2,3], [5,10]]`: `inst_def` is `[start_years, maturity_years]`.
                    # `sw.st = inst[k][0]`, `sw.mt = inst[k][1]`.

                    # The interpretation of `instruments` in `Swap_curve_fwd` is:
                    # `inst = [[2],[5],[10]]` -> `start=0`, `end=2`, `start=0`, `end=5`, `start=0`, `end=10`
                    # `inst = [[2,3], [5,10]]` -> `start=2`, `end=3`, `start=5`, `end=10`
                    # So `inst_def[0]` is the *start* of the period (or 0 for spot), `inst_def[1]` is the *end* of the period.
                    # This means `maturity_tenor` refers to the *absolute maturity year* from start date of instrument.
                    #
                    # Corrected interpretation based on example usage of `inst` in `Swap_curve_fwd`:
                    # `instruments` elements are either `[maturity_in_years]` (for spot-starting)
                    # or `[start_in_years, end_in_years]` (for forward-starting).
                    #
                    # For `start_tenor_for_curve` application:
                    # If `inst_def` is `[maturity_in_years]`:
                    #   `start_tenor` for `SwapParameters` is `start_yr_offset`.
                    #   `maturity_tenor` for `SwapParameters` is `inst_def[0]` (length of spot swap).
                    # If `inst_def` is `[fwd_start_in_years, fwd_end_in_years]`:
                    #   `start_tenor` for `SwapParameters` is `start_yr_offset + inst_def[0]`.
                    #   `maturity_tenor` for `SwapParameters` is `inst_def[1]`.

                    # This structure is not ideal. Let's make `instruments` explicit:
                    # `List[Tuple[int, int]]` where `(start_offset_from_curve_date, length_of_swap_in_years)`
                    # For spot-starting 10Y: `(0, 10)`
                    # For 5Yx10Y: `(5, 10)`

                    # Given `inst = [[2],[5],[10]]` and `inst = [[2,3],[5,10]]`, the original seems
                    # to use `inst[k][0]` as start and `inst[k][1]` as end if two elements.
                    # And `0` as start and `inst[k][0]` as end if one element.
                    #
                    # Let's adjust `instruments` input to be:
                    # `List[Dict[str, Union[int, str]]]` with keys like `start_offset` and `length`.
                    # Or simpler, let `instruments` be `List[Tuple[int, int]]` where first is offset, second is length.

                    # Assuming `instruments` specifies `(period_start_from_spot, period_length_in_years)`
                    # E.g., for `2Y` spot: `(0, 2)`. For `5Yx10Y`: `(5, 10)`.
                    #
                    # If `len(inst_def) == 1`: means `[maturity_length_from_spot]`.
                    #   `start_tenor_param = start_yr_offset`
                    #   `maturity_tenor_param = inst_def[0]`
                    # If `len(inst_def) == 2`: means `[forward_start_length, forward_maturity_length]`.
                    #   `start_tenor_param = start_yr_offset + inst_def[0]`
                    #   `maturity_tenor_param = inst_def[1]`
                    #
                    # This implies:
                    fwd_start_offset = 0  # Offset for the instrument itself from the curve's base
                    fwd_instrument_length = inst_def[0]  # Length of the instrument itself
                    if len(inst_def) == 2:
                        fwd_start_offset = inst_def[0]
                        fwd_instrument_length = inst_def[1]

                    # Calculate actual start and end tenors for the SwapParameters
                    swap_start_tenor_for_params = f"{int(start_yr_offset + fwd_start_offset)}Y"
                    swap_maturity_tenor_for_params = fwd_instrument_length  # This is the length of the swap

                swap_params = SwapParameters(
                    currency_code=currency_code,  #
                    start_tenor=swap_start_tenor_for_params,  #
                    maturity_tenor=fwd_instrument_length,  # This is the length of the swap (e.g., 10Y in 5Yx10Y)
                    notional=1_000_000.0,  # Default notional
                    fixed_leg_freq=fixed_leg_freq  #
                )

                swap_instrument = Swap(swap_params, pricing_curve_builder=curve_builder,
                                       valuation_date_input=ql_eval_date)
                rates_per_instrument[k].append(swap_instrument.calculate_fair_rate())

        # Combine rates based on ratios
        curve_fwds_output = np.array(
            [-100 * np.sum([rates_per_instrument[i][j] * ratios[i] for i in range(len(ratios))]) for j in
             range(len(start_tenors_for_curve))])
        curve_fwds_carry = -1 * np.diff(curve_fwds_output)

        return {
            "forward_rates": curve_fwds_output,
            "carry": curve_fwds_carry,
            "tenors": start_tenors_for_curve
        }

    def quick_swap_rate(self,
                        curve_builder: Union[SwapCurveBuilder, OISCurveBuilder],
                        start_period_value: int,
                        start_period_unit: ql.TimeUnit,
                        swap_tenor_value: int,
                        swap_tenor_unit: ql.TimeUnit,
                        fixed_leg_freq: Optional[str] = None
                        ) -> float:
        """
        Calculates the fair rate of a swap using QuantLib's MakeVanillaSwap utility.
        This function is intended for quick, simple swap rate calculations.

        Args:
            curve_builder (Union[SwapCurveBuilder, OISCurveBuilder]):
                An instantiated and built curve builder (e.g., SwapCurveBuilder, OISCurveBuilder).
            start_period_value (int): Value for the swap's forward start period (e.g., 1 for 1Y).
            start_period_unit (ql.TimeUnit): Unit for the swap's forward start period (e.g., ql.Years, ql.Months).
            swap_tenor_value (int): Value for the swap's maturity tenor (e.g., 10 for 10Y).
            swap_tenor_unit (ql.TimeUnit): Unit for the swap's maturity tenor (e.g., ql.Years, ql.Months).
            fixed_leg_freq (Optional[str]): Optional override for the fixed leg frequency.

        Returns:
            float: The fair fixed rate of the swap in percentage.
        """
        curve_builder.build_curve()  # Ensure the curve is built
        term_structure_handle = ql.YieldTermStructureHandle(curve_builder.curve)

        # Get currency conventions for index
        from src.config import config  #
        currency_config = config.get_currency_config(curve_builder.currency_code)  #
        floating_leg_config = currency_config['floating_leg']  #
        settlement_days = currency_config['settlement_days']  #
        ql_calendar = currency_config['ql_calendar_obj']  #
        ql_currency = currency_config['ql_currency_obj']  #

        # Create IborIndex
        ibor_index = ql.IborIndex(  #
            floating_leg_config['fixing_index'],  #
            ql.Period(floating_leg_config['fixing_tenor']),  #
            settlement_days,  #
            ql_currency,  #
            ql_calendar,  #
            get_ql_business_day_convention(floating_leg_config['business_day_convention']),  #
            True,  # endOfMonth
            get_ql_day_counter(floating_leg_config['day_count']),  #
            term_structure_handle  #
        )  #

        # Set pricing engine
        pricing_engine = ql.DiscountingSwapEngine(curve_builder.discount_curve if hasattr(curve_builder,
                                                                                          'discount_curve') and curve_builder.discount_curve else term_structure_handle)

        # Define start period and swap tenor using QuantLib.Period
        start_period_ql = ql.Period(start_period_value, start_period_unit)
        swap_tenor_ql = ql.Period(swap_tenor_value, swap_tenor_unit)

        # Build swap using MakeVanillaSwap helper
        swap = ql.MakeVanillaSwap(swap_tenor_ql, ibor_index, 0.0, start_period_ql,
                                  pricingEngine=pricing_engine,
                                  settlementDays=settlement_days)

        # Apply fixed leg frequency override if provided
        if fixed_leg_freq:
            ql_fixed_freq = get_ql_frequency(fixed_leg_freq)
            # MakeVanillaSwap does not directly support fixed leg frequency override in constructor
            # This would require manually building the fixed leg schedule or adjusting the MakeVanillaSwap helper
            # For simplicity, will not apply override here if MakeVanillaSwap is used.
            # If this is critical, a full Swap object should be built instead of MakeVanillaSwap.
            print(
                f"Warning: Fixed leg frequency override '{fixed_leg_freq}' is not directly supported by MakeVanillaSwap. Use full Swap instrument for custom frequency.")

        return swap.fairRate() * 100