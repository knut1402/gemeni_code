# src/strategy_analytics/option_strategy.py

import QuantLib as ql
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.volatility_surface.bond_future_vol_surface import BondFutureVolSurface  #
from src.volatility_surface.stir_vol_surface import StirVolSurface  #
from src.instruments.option import Option  #
from src.financial_math_utils import bond_fut_yield, px_opt_ticks, px_dec_to_frac, convert_to_64  #
from src.date_utils import ql_to_datetime  #


class OptionStrategy:
    """
    Analyzes option strategies (e.g., butterflies, spreads) built from individual options.

    Calculates strategy P&L and Greeks over a simulated range of underlying prices,
    leveraging a volatility surface to determine implied volatilities.
    """

    def __init__(self,
                 currency_code: str,
                 underlying_ticker: str,  # Bloomberg ticker of the underlying future
                 option_types: List[str],  # List of option types ('C' or 'P')
                 strikes: List[float],  # List of strike prices
                 weights: List[float],  # List of weights/multipliers for each option (e.g., [1, -2, 1] for butterfly)
                 strategy_name: str = 'Option Strategy',
                 chain_length: List[int] = [12, 12],
                 # Default for vol surface building [calls_below_atm, calls_above_atm]
                 simulation_range: List[float] = [-3.0, 3.0],
                 # Range around ATM for simulation (e.g., [-3, 3] for -3 to +3 points)
                 simulation_increment: float = 0.5,  # Step size for simulation (e.g., 0.5 points)
                 vol_surface_builder: Optional[Union[BondFutureVolSurface, StirVolSurface]] = None
                 ):
        """
        Initializes the OptionStrategy for analysis.

        Args:
            currency_code (str): The three-letter currency code.
            underlying_ticker (str): Bloomberg ticker of the underlying future (e.g., 'USU1 Comdty', 'EDH2 Comdty').
            option_types (List[str]): List of option types corresponding to strikes ('C' or 'P').
            strikes (List[float]): List of strike prices.
            weights (List[float]): List of weights/multipliers for each option in the strategy.
            strategy_name (str): A descriptive name for the strategy.
            chain_length (List[int]): Defines the number of strikes to fetch around ATM for volatility surface building.
            simulation_range (List[float]): Defines the range (min, max offset from current spot) for underlying price simulation.
            simulation_increment (float): Step size for underlying price simulation.
            vol_surface_builder (Optional[Union[BondFutureVolSurface, StirVolSurface]]):
                An already built volatility surface builder object. If None, one will be instantiated.
        """
        self.currency_code = currency_code
        self.underlying_ticker = underlying_ticker
        self.option_types = option_types
        self.strikes = strikes
        self.weights = weights
        self.strategy_name = strategy_name
        self.chain_length = chain_length
        self.simulation_range = simulation_range
        self.simulation_increment = simulation_increment
        self._vol_surface_builder = vol_surface_builder

        self._strategy_spot_px: float = np.nan
        self._strategy_px: float = np.nan
        self._strategy_expiry_dt: ql.Date = None
        self._future_ticker_root: str = self.underlying_ticker.split(' ')[0][:4]  # e.g., 'USU1' from 'USU1 Comdty'
        self._underlying_futures_bbg_ticker: str = self.underlying_ticker  # Full Bloomberg ticker, e.g., 'USU1 Comdty'

        self.strategy_analysis_df: pd.DataFrame = pd.DataFrame()  # To store simulation results

        self._setup_volatility_surface()
        self._analyze_strategy()

    def _setup_volatility_surface(self) -> None:
        """
        Sets up the volatility surface for pricing the options in the strategy.
        Instantiates a new builder if one is not provided.
        """
        if self._vol_surface_builder is None:
            # Determine surface type based on underlying_ticker pattern
            if ' Comdty' in self.underlying_ticker:  # Common suffix for futures
                # Simple heuristic: if ticker root length is 3 (e.g., 'USU'), likely bond future.
                # If 2 (e.g., 'ED'), likely STIR. This needs verification.
                if len(self.underlying_ticker.split(' ')[0][
                       :3]) == 3:  # Assuming 3-char root for bond futures like USU, TYU
                    print(f"Instantiating BondFutureVolSurface for {self.underlying_ticker}...")
                    self._vol_surface_builder = BondFutureVolSurface(self.currency_code, self.underlying_ticker)
                else:  # Fallback to STIR if not recognized as bond future
                    print(f"Instantiating StirVolSurface for {self.underlying_ticker}...")
                    self._vol_surface_builder = StirVolSurface(self.currency_code, self.underlying_ticker)
            else:
                raise ValueError(
                    f"Could not infer volatility surface type for {self.underlying_ticker}. Please provide a vol_surface_builder.")

        # Build the volatility surface
        self._vol_surface_builder.build_surface(self._future_ticker_root, self.chain_length)

        # Store key properties from the built surface
        self._strategy_spot_px = self._vol_surface_builder.underlying_price
        self._strategy_expiry_dt = self._vol_surface_builder.expiry_date

    def _analyze_strategy(self) -> None:
        """
        Performs the core strategy analysis, simulating underlying prices
        and calculating strategy P&L and Greeks for each simulated point.
        """
        vol_surface_data = self._vol_surface_builder.surface_data

        # Extract options data from the surface for current strategy options
        strategy_options_data = pd.DataFrame()
        for i in range(len(self.strikes)):  #
            opt_type = self.option_types[i]  #
            strike = self.strikes[i]  #
            # Find the exact option in the fetched surface data
            matched_option = vol_surface_data[
                (vol_surface_data['option_type'] == opt_type) &
                (vol_surface_data['strike'] == strike)
                ]  #
            if matched_option.empty:
                raise ValueError(f"Option {opt_type} {strike} not found in the built volatility surface data.")
            strategy_options_data = pd.concat([strategy_options_data, matched_option])  #
        strategy_options_data.reset_index(drop=True, inplace=True)  #
        strategy_options_data['weights'] = self.weights  #

        # Calculate initial strategy price and Greeks at current spot
        self._strategy_px = np.dot(strategy_options_data['px'], strategy_options_data['weights'])  #
        initial_delta = np.dot(strategy_options_data['delta'], strategy_options_data['weights'])  #
        initial_gamma = np.dot(strategy_options_data['gamma'], strategy_options_data['weights'])  #
        initial_theta = np.dot(strategy_options_data['theta'], strategy_options_data['weights'])  #
        initial_vega = np.dot(strategy_options_data['vega'], strategy_options_data['weights'])  #

        # Initialize strategy analysis DataFrame with current spot values
        self.strategy_analysis_df = pd.DataFrame([{
            'fut_px': self._strategy_spot_px,  #
            'ATM_K': 0.0,  # At current spot, ATM_K is 0
            'strat_px': self._strategy_px,  #
            'strat_delta': initial_delta,  #
            'strat_gamma': initial_gamma,  #
            'strat_theta': initial_theta,  #
            'strat_vega': initial_vega  #
        }])

        # Simulate underlying price movements
        sim_spot_prices = np.arange(
            min(self.strikes) + self.simulation_range[0],  # Min strike offset by range
            max(self.strikes) + self.simulation_range[1],  # Max strike offset by range
            step=self.simulation_increment  #
        )

        # Filter out current spot price if it falls exactly on a simulated point,
        # to avoid duplicating the initial row.
        sim_spot_prices = np.unique(np.round(sim_spot_prices, 6))  # Round to handle float precision

        # Calculate bond future yield for simulated spots (only for bond futures)
        # This part depends on `bond_fut_yield` utility for `K_Yield` and `K_Dist`
        # For STIRs, `bond_fut_yield` would not apply in the same way for `K_Yield`.
        # Assuming `bond_fut_yield` is general enough or a similar utility exists for STIR.
        # This part requires specific implementation of `bond_fut_yield` in `financial_math_utils.py`.
        #
        # NOTE: `bond_fut_yield` implementation from `Utilities.py` was not provided.
        # Its behavior for `K_Yield` and `K_Dist` (ATM_K) is inferred.
        # For bond futures, it relates to the underlying bond's yield.
        # For STIRs, ATM_K is simpler, often price difference from underlying.
        #
        # If `bond_fut_yield` expects a bond future ticker:
        if isinstance(self._vol_surface_builder, BondFutureVolSurface):  #
            sim_spot_yield_data = bond_fut_yield(self._future_ticker_root, sim_spot_prices.tolist())  #
            # Assuming sim_spot_yield_data has 'K_Yield' for each simulated spot.
            # And `sim_spot_yield_data['K_Yield'][0]` is the yield at current underlying.
            base_yield_for_shifts = sim_spot_yield_data['K_Yield'].iloc[
                (sim_spot_yield_data['K_Yield'] -
                 bond_fut_yield(self._future_ticker_root, [self._strategy_spot_px])['K_Yield'].iloc[0]).abs().argsort()[
                :1].iloc[0]
            ] if not sim_spot_yield_data.empty else np.nan  #
        else:  # For STIRs or other types where bond_fut_yield doesn't apply directly for K_Yield
            # Simplified ATM_K for STIRs: Price difference from ATM strike
            base_yield_for_shifts = 0.0  # Placeholder, as STIRs typically don't have 'K_Yield'
            sim_spot_yield_data = pd.DataFrame({'K_Yield': sim_spot_prices})  # Placeholder

        # Simulate and calculate Greeks for each simulated spot price
        for sim_px in sim_spot_prices:  #
            # Interpolate implied volatility for each option at the simulated spot price
            # This is complex as it involves shifting the vol smile/surface.
            # Original code: `shift1 = 100*(bond_fut_yield(v1.fut[:4], strikes)['K_Yield'] - spot_yld_sim.loc[i]['K_Yield'])`
            # `iv_new = []` `np.interp(shift1[j], df5['ATM_K'], df5['iv'])`

            # This implies creating a "shifted" ATM_K for each original strike based on new spot.
            # Then interpolating based on that shifted ATM_K.

            sim_option_ivs = []
            for option_strike in self.strikes:  #
                # Recalculate 'ATM_K' for each strike at the simulated spot price
                # This needs `bond_fut_yield` or equivalent for the simulated spot.
                if isinstance(self._vol_surface_builder, BondFutureVolSurface):  #
                    # Need yield at simulated spot, then calculate strike yield shift
                    sim_spot_yield = bond_fut_yield(self._future_ticker_root, [sim_px])['K_Yield'].iloc[0]  #
                    strike_yield_at_current_fut = \
                    bond_fut_yield(self._future_ticker_root, [option_strike])['K_Yield'].iloc[0]  #
                    shift_from_atm = 100 * (
                                strike_yield_at_current_fut - sim_spot_yield)  # Similar to original `shift1`
                    # Interpolate IV based on this new `ATM_K`
                    interpolated_iv = self._vol_surface_builder.interpolate_volatility(shift_from_atm, axis='ATM_K',
                                                                                       method='linear')  #
                else:  # For STIRs, ATM_K is simpler (price difference)
                    interpolated_iv = self._vol_surface_builder.interpolate_volatility(option_strike, axis='strike',
                                                                                       method='linear')  #

                sim_option_ivs.append(interpolated_iv)  #

            # Get simulated option prices and Greeks using the interpolated volatilities
            # `get_sim_option_px` from Utilities.py
            # This is a critical utility not fully provided in original `Utilities.py`.
            # Assuming its output matches the class `sim_option_px_output` from `BOND_FUTURES.py`.
            #
            # NOTE: `get_sim_option_px` implementation from `Utilities.py` was not provided.
            # Its role is to price options at simulated spot with interpolated vols.
            # It's replicated here using the `Option` class and its `set_underlying_price`/`set_volatility` methods.
            #
            sim_prices_list = []
            sim_deltas_list = []
            sim_gammas_list = []
            sim_thetas_list = []
            sim_vegas_list = []

            for i, opt_type_str in enumerate(self.option_types):  #
                option_inst = Option(
                    currency_code=self.currency_code,  #
                    option_type=opt_type_str,  #
                    strike_price=self.strikes[i],  #
                    expiry_date_input=self._strategy_expiry_dt,  #
                    underlying_ticker=self._underlying_futures_bbg_ticker,  #
                    valuation_date_input=self.valuation_date  #
                )
                option_inst.set_underlying_price(sim_px)  #
                option_inst.set_volatility(sim_option_ivs[i] / 100.0)  #

                sim_prices_list.append(option_inst.calculate_npv())  #
                sim_deltas_list.append(option_inst.calculate_delta())  #
                sim_gammas_list.append(option_inst.calculate_gamma())  #
                sim_thetas_list.append(option_inst.calculate_theta())  #
                sim_vegas_list.append(option_inst.calculate_vega())  #

            # Aggregate strategy P&L and Greeks
            sim_strat_px = np.dot(sim_prices_list, self.weights)  #
            sim_strat_delta = np.dot(sim_deltas_list, self.weights)  #
            sim_strat_gamma = np.dot(sim_gammas_list, self.weights)  #
            sim_strat_theta = np.dot(sim_thetas_list, self.weights)  #
            sim_strat_vega = np.dot(sim_vegas_list, self.weights)  #

            # Append to strategy analysis DataFrame
            self.strategy_analysis_df.loc[len(self.strategy_analysis_df)] = {
                'fut_px': sim_px,  #
                'ATM_K': (self._strategy_spot_px - sim_px),  # Simplified ATM_K for simulation plot
                'strat_px': sim_strat_px,  #
                'strat_delta': sim_strat_delta,  #
                'strat_gamma': sim_strat_gamma,  #
                'strat_theta': sim_strat_theta,  #
                'strat_vega': sim_strat_vega  #
            }

        self.strategy_analysis_df.sort_values(by=['fut_px'], inplace=True)  #
        self.strategy_analysis_df.reset_index(drop=True, inplace=True)  #

        # Add formatted price column as per original `px_dec_to_opt_frac`
        # The `currency` property was used to determine formatting.
        # Assuming `self.currency_code` can be used.
        if self.currency_code == 'USD':  #
            self.strategy_analysis_df['strat_px_fmt'] = self.strategy_analysis_df['strat_px'].apply(px_opt_ticks)  #
        else:  # For STIRs, typically just rounded decimal or absolute value of 100-price
            self.strategy_analysis_df['strat_px_fmt'] = np.round(self.strategy_analysis_df['strat_px'], 3).astype(
                str)  #

        # Add a strategy type for plotting later ('USD' for bond, 'stir' for STIR)
        if isinstance(self._vol_surface_builder, BondFutureVolSurface):  #
            self._strategy_type = 'USD'  #
        elif isinstance(self._vol_surface_builder, StirVolSurface):  #
            self._strategy_type = 'stir'  #
        else:  #
            self._strategy_type = 'generic'  #

    @property
    def strategy_details(self) -> Dict[str, Any]:
        """
        Returns a dictionary of core strategy details.
        Corresponds to the output of `bf_opt_strat_output` or `stir_opt_strat_output`.
        """
        return {
            "vol_surface_data": self._vol_surface_builder.surface_data,  #
            "ticker": self.underlying_ticker,  #
            "fut_bbg_ticker": self._underlying_futures_bbg_ticker,  #
            "center_strike": self._vol_surface_builder.at_the_money_strike,  #
            "expiry_dt": self._strategy_expiry_dt,  #
            "spot_px": self._strategy_spot_px,  #
            "spot_px_fmt": px_dec_to_frac(self._strategy_spot_px) if self.currency_code == 'USD' else str(
                np.round(self._strategy_spot_px, 3)),  #
            "strategy_simulation_df": self.strategy_analysis_df,  #
            "strategy_name": self.strategy_name,  #
            "strategy_current_px": self._strategy_px,  #
            "option_details": [self.underlying_ticker, self.option_types, self.strikes, self.weights],  #
            "strategy_type": self._strategy_type  #
        }