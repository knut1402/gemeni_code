# src/instruments/swap.py

from dataclasses import dataclass, field
import QuantLib as ql
import datetime
from typing import Optional, Union, Dict, Any, List
from src.instruments.base_instrument import BaseInstrument
from src.curves.swap_curve_builder import SwapCurveBuilder
from src.curves.ois_curve_builder import \
    OISCurveBuilder  # For explicit OIS discounting curve if not handled by SwapCurveBuilder
from src.quantlib_utils import get_ql_date, get_ql_business_day_convention, get_ql_day_counter, get_ql_frequency  #


@dataclass
class SwapParameters:
    """
    Defines the parameters for a swap instrument.

    This dataclass replaces the inferred `swap_class` from the original Utilities.py
    for a clearer, type-hinted approach to defining swap characteristics.
    """
    currency_code: str  # e.g., 'USD', 'EUR', 'GBP'
    start_tenor: Union[int, str]  # e.g., 0 for spot, '3M' for 3 months forward
    maturity_tenor: Union[int, str]  # e.g., 10 for 10 years
    notional: float = 1_000_000.0  # Default notional
    fixed_rate_x: float = 0.0  # Reference fixed rate, often 0.0 or 2.0 for pricing
    fixed_leg_freq: Optional[Union[str, ql.Frequency]] = None  # Optional fixed leg frequency override


class Swap(BaseInstrument):
    """
    Represents a Vanilla Interest Rate Swap instrument.

    Handles the construction of QuantLib swap objects, calculates fair rates, NPV,
    and provides methods for sensitivity analysis.
    """

    def __init__(self, swap_params: SwapParameters,
                 pricing_curve_builder: Optional[Union[SwapCurveBuilder, OISCurveBuilder]] = None,
                 valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the Swap instrument.

        Args:
            swap_params (SwapParameters): Dataclass holding the swap's defining parameters.
            pricing_curve_builder (Optional[Union[SwapCurveBuilder, OISCurveBuilder]]):
                An already built QuantLib curve builder object (e.g., SwapCurveBuilder, OISCurveBuilder)
                to be used for pricing. If None, a new SwapCurveBuilder will be instantiated.
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the instrument is valued. If None, uses today's date.
        """
        super().__init__(swap_params.currency_code, valuation_date_input)
        self.params = swap_params
        self._pricing_curve_builder = pricing_curve_builder
        self._index_name = f"{self.currency_code}_{self._convention_config['floating_leg']['fixing_tenor']}"  #

        # Ensure pricing curve is built/available
        self._setup_pricing_curves()

        # Setup the QuantLib instrument
        self._setup_ql_instrument()

    def _setup_pricing_curves(self) -> None:
        """
        Sets up the QuantLib pricing curve(s) for the swap.
        Prioritizes provided curve builder, otherwise builds a new one.
        Handles single-curve and multi-curve (OIS discounting) frameworks.
        """
        if self._pricing_curve_builder:
            # If a curve builder is provided, use its curve(s)
            self._pricing_curve_builder.build_curve()  # Ensure curve is built
            self._ql_pricing_curve = self._pricing_curve_builder.curve  #
            # Get discount curve. If SwapCurveBuilder has a discount_curve property, use it.
            # Else, if OIS, use its main curve. If neither, use the pricing curve itself.
            if hasattr(self._pricing_curve_builder, 'discount_curve') and self._pricing_curve_builder.discount_curve:  #
                self._ql_discount_curve_handle = self._pricing_curve_builder.discount_curve  #
            else:
                self._ql_discount_curve_handle = ql.YieldTermStructureHandle(self._ql_pricing_curve)  #
        else:
            # If no builder provided, create a new SwapCurveBuilder
            print(f"No pricing curve builder provided. Building default SwapCurveBuilder for {self.currency_code}...")
            swap_builder = SwapCurveBuilder(self.currency_code, self.valuation_date)  #
            swap_builder.build_curve()  #
            self._ql_pricing_curve = swap_builder.curve  #
            self._ql_discount_curve_handle = swap_builder.discount_curve if swap_builder.discount_curve else ql.YieldTermStructureHandle(
                self._ql_pricing_curve)  #

        self._ql_pricing_curve_handle = ql.YieldTermStructureHandle(self._ql_pricing_curve)  #

    def _fetch_market_data(self) -> None:
        """
        For a Swap instrument, market data is typically embodied in the pricing curves.
        This method is included for consistency with BaseInstrument but may not fetch
        additional raw data for a standard swap.
        """
        pass  # Market data is handled by the associated curve builders

    def _setup_ql_instrument(self) -> None:
        """
        Sets up the QuantLib VanillaSwap object and its pricing engine.
        """
        fixed_leg_config = self._convention_config['fixed_leg']  #
        floating_leg_config = self._convention_config['floating_leg']  #

        # Convert fixed_leg_freq override if provided
        fixed_freq = get_ql_frequency(fixed_leg_config['frequency'])  #
        if self.params.fixed_leg_freq:  #
            if isinstance(self.params.fixed_leg_freq, str):
                fixed_freq = get_ql_frequency(self.params.fixed_leg_freq)  #
            elif isinstance(self.params.fixed_leg_freq, ql.Frequency):
                fixed_freq = self.params.fixed_leg_freq  #

        # Determine start and end dates based on tenors
        start_date = self.valuation_date  # Default spot start
        if isinstance(self.params.start_tenor, str):  #
            # Handle period strings like '3M', '1Y'
            start_date = self.calendar.advance(start_date, ql.Period(self.params.start_tenor))  #
        elif isinstance(self.params.start_tenor, int) and self.params.start_tenor > 0:  #
            # Interpret as years offset for simplicity, could be days offset if context requires
            start_date = self.calendar.advance(start_date, ql.Period(self.params.start_tenor, ql.Years))  #

        end_date = self.calendar.advance(start_date, ql.Period(self.params.maturity_tenor,
                                                               ql.Years))  # Assuming maturity_tenor in years
        if isinstance(self.params.maturity_tenor, str):  #
            end_date = self.calendar.advance(start_date, ql.Period(self.params.maturity_tenor))  #

        # Fixed Leg Schedule
        fixed_schedule = ql.Schedule(
            start_date, end_date, fixed_freq,  #
            self.calendar,  #
            get_ql_business_day_convention(fixed_leg_config['business_day_convention']),  #
            get_ql_business_day_convention(fixed_leg_config['business_day_convention']),  #
            ql.DateGeneration.Forward,  #
            False  # End of month
        )

        # Floating Leg Schedule
        floating_schedule = ql.Schedule(
            start_date, end_date, get_ql_frequency(floating_leg_config['frequency']),  #
            self.calendar,  #
            get_ql_business_day_convention(floating_leg_config['business_day_convention']),  #
            get_ql_business_day_convention(floating_leg_config['business_day_convention']),  #
            ql.DateGeneration.Forward,  #
            False  # End of month
        )

        # Floating Leg Index
        # The original code's index logic (from SWAP_PRICER.py) had index_custom and ois_trigger flags.
        # Here, we directly use the `IborIndex` if applicable, assuming conventions define it.
        #
        # NOTE: `index_custom` flag from Conventions.py was complex.
        # Assuming for now that `floating_leg_config['fixing_index']` directly maps to an IborIndex name.
        # If the index is truly custom (e.g., requiring special build), this will need review.
        floating_ibor_index = ql.IborIndex(
            self._convention_config['floating_leg']['fixing_index'],  #
            ql.Period(self._convention_config['floating_leg']['fixing_tenor']),  #
            self._convention_config['settlement_days'],  #
            self.currency,  #
            self.calendar,  #
            get_ql_business_day_convention(self._convention_config['floating_leg']['business_day_convention']),  #
            True,  # endOfMonth, typically true for Ibor
            get_ql_day_counter(self._convention_config['floating_leg']['day_count']),  #
            self._ql_pricing_curve_handle  # Uses the main pricing curve for forecasting
        )

        # Create the QuantLib VanillaSwap object
        self._ql_instrument = ql.VanillaSwap(
            ql.VanillaSwap.Payer,  # Payer of fixed leg
            self.params.notional,  #
            fixed_schedule,  #
            self.params.fixed_rate_x / 100,  # Fixed rate as decimal
            get_ql_day_counter(fixed_leg_config['day_count']),  #
            floating_schedule,  #
            floating_ibor_index,  #
            0.0,  # Floating spread
            get_ql_day_counter(floating_leg_config['day_count'])  #
        )

        # Set the pricing engine
        self._ql_engine = ql.DiscountingSwapEngine(self._ql_discount_curve_handle)  #
        self._ql_instrument.setPricingEngine(self._ql_engine)  #

    def calculate_fair_rate(self) -> float:
        """
        Calculates the fair fixed rate of the swap.

        Returns:
            float: The fair fixed rate in percentage.
        """
        if self._ql_instrument is None:
            self._setup_ql_instrument()
        return self._ql_instrument.fairRate() * 100  #

    def calculate_npv(self) -> float:
        """
        Calculates the Net Present Value (NPV) of the swap.

        Returns:
            float: The NPV of the swap.
        """
        if self._ql_instrument is None:
            self._setup_ql_instrument()
        return self._ql_instrument.NPV()  #

    def calculate_dv01(self, bump_size_bps: float = 0.5) -> float:
        """
        Calculates the DV01 (Dollar Value of an 01) of the swap using bump-and-reprice.

        Args:
            bump_size_bps (float): The size of the bump in basis points (e.g., 0.5 for 0.5 bps).

        Returns:
            float: The DV01 of the swap.
        """
        if self._ql_instrument is None:
            self._setup_ql_instrument()

        original_fair_rate = self._ql_instrument.fairRate()  #
        original_npv = self._ql_instrument.NPV()  #

        bump_amount = bump_size_bps / 10000.0  # Convert bps to decimal

        # Bump up
        bumped_up_rate_handle = ql.QuoteHandle(ql.SimpleQuote(original_fair_rate + bump_amount))  #
        temp_engine_up = ql.DiscountingSwapEngine(self._ql_discount_curve_handle)  #
        temp_instrument_up = ql.VanillaSwap(  #
            self._ql_instrument.type(), self.params.notional, self._ql_instrument.fixedSchedule(),  #
            bumped_up_rate_handle.value(), self._ql_instrument.fixedDayCount(),  #
            self._ql_instrument.floatingSchedule(), self._ql_instrument.floatingLegIndex(),  #
            self._ql_instrument.spread(), self._ql_instrument.floatingDayCount()  #
        )  #
        temp_instrument_up.setPricingEngine(temp_engine_up)  #
        npv_up = temp_instrument_up.NPV()  #

        # Bump down
        bumped_down_rate_handle = ql.QuoteHandle(ql.SimpleQuote(original_fair_rate - bump_amount))  #
        temp_engine_down = ql.DiscountingSwapEngine(self._ql_discount_curve_handle)  #
        temp_instrument_down = ql.VanillaSwap(  #
            self._ql_instrument.type(), self.params.notional, self._ql_instrument.fixedSchedule(),  #
            bumped_down_rate_handle.value(), self._ql_instrument.fixedDayCount(),  #
            self._ql_instrument.floatingSchedule(), self._ql_instrument.floatingLegIndex(),  #
            self._ql_instrument.spread(), self._ql_instrument.floatingDayCount()  #
        )  #
        temp_instrument_down.setPricingEngine(temp_engine_down)  #
        npv_down = temp_instrument_down.NPV()  #

        # DV01 is the change in NPV for a 1 basis point change in rate.
        # (NPV_down - NPV_up) / (2 * bump_amount * self.params.notional / 100)
        # Original formula was `(a2-b2)/ ((a1-b1)*sw.n/100)`
        # (NPV_bump_down - NPV_bump_up) / (2 * bump_amount) is the change in NPV per unit of rate change.
        # Then scale by notional and 100 for percentage basis points.
        # DV01 often implies per 100 units of notional for 1 bp change.
        # So it's (NPV_down - NPV_up) / (2 * bump_amount) / self.params.notional * 10000 (if per 100 of notional, per basis point)
        # Or, just (NPV_down - NPV_up) / (2 * bump_amount) gives NPV sensitivity per unit of rate change.
        # If DV01 is defined as "change in NPV for 1bp parallel shift in rate" in absolute terms
        # then (NPV_down - NPV_up) / (2 * bump_amount / 10000) for a 1bp shift.

        # Replicating original `(a2-b2)/ ((a1-b1)*sw.n/100)`
        # `a1` and `b1` are fair rates after bump, `a2` and `b2` are NPVs.
        # This is `Delta_NPV / (Delta_Fair_Rate * Notional_Factor)`
        # The original `ind_01` was `(a2-b2)/ ((a1-b1)*sw.n/100)`
        # where `a1-b1` is `2*bump_amount*100` if fair rate changes by bump.
        # So it's `(npv_up - npv_down) / (2 * bump_amount * self.params.notional / 100)` from original usage context.
        # `ind_01 = round((a2-b2)/ ((a1-b1)*sw.n/100),2)`
        # `a2-b2` is `NPV_up - NPV_down` with positive rate change meaning `npv_up - npv_down`.
        # `(a1-b1)` is `(Original_Rate + Bump) - (Original_Rate - Bump)` -> `2*Bump_Rate_Change`.
        # The term `sw.n/100` seems to normalize by notional in hundreds.

        # Let's return the basic sensitivity per 1 unit change in rate
        # then scale it as needed by the caller to match original output `ind_01`.
        # The common DV01 is change in NPV for 1bp shift.

        # DV01: Change in NPV for a 1 basis point (0.01%) parallel shift in the fair rate.
        # = (NPV_down - NPV_up) / (2 * bump_amount) * (0.0001 / self.params.notional) * self.params.notional * 10000
        # = (NPV_down - NPV_up) / (2 * bump_amount) * 10000 (to get per 1% change)
        # Then scaled by notional as `sw.n / 100`.

        # Original logic: `ind_01 = round((a2-b2)/ ((a1-b1)*sw.n/100),2)`
        # where `a1, b1` are fair rates (percentages), `a2, b2` are NPVs.
        # If `a1 = original_fair_rate_plus_bump` and `b1 = original_fair_rate_minus_bump`,
        # then `(a1-b1)` is `2 * bump_size_bps`.
        # So `DV01_original_style = (npv_up - npv_down) / (2 * bump_size_bps * self.params.notional / 100)`
        # Or, assuming `a2` is NPV for rate `+bump` and `b2` for `-bump`:
        # `(npv_up - npv_down) / (2 * bump_size_bps)` gives NPV change per percentage point change.
        # DV01 is per 1bp, so divide by 100.
        # Final `DV01` logic based on what `ind_01` would represent:
        # It means `Delta_NPV / Delta_Rate_Percent / (Notional / 100)`.
        # So, `Delta_NPV` is `(npv_up - npv_down)`.
        # `Delta_Rate_Percent` is `2 * bump_size_bps / 100`.
        # `Notional / 100` is `self.params.notional / 100`.

        # Let's calculate the DV01 as: change in NPV for a 1bp (0.0001) shift in rate.
        # It's (NPV_up - NPV_down) / (2 * bump_amount * 10000)
        # Original: `round((a2-b2)/ ((a1-b1)*sw.n/100),2)`
        # Let `Rate1` be `fair_rate + bump`, `Rate2` be `fair_rate - bump`. `NPV1, NPV2` their NPVs.
        # `(NPV1 - NPV2) / ((Rate1 - Rate2) * Notional / 100)`
        # `(NPV1 - NPV2) / (2 * bump_amount * 100 * Notional / 100)`
        # `(NPV1 - NPV2) / (2 * bump_amount * Notional)`
        # So:
        dv01_val = (npv_up - npv_down) / (2 * bump_amount * self.params.notional)  #
        # The original code had a final division by `self.params.notional / 100` after `(a1-b1)`.
        # This suggests `ind_01` is "per 1 unit of notional" * 100 (for % of notional).

        # Let's use the definition: change in NPV per 1% change in rate, per 100 units of notional.
        # (NPV_up - NPV_down) / (2 * bump_amount * 100) * (100 / self.params.notional)
        # This simplifies to (NPV_up - NPV_down) / (2 * bump_amount * self.params.notional)
        # The `round(..., 2)` suggests two decimal places.
        return round(dv01_val * 10000, 2)  # DV01 per 100 notional for 1bp shift (common usage)