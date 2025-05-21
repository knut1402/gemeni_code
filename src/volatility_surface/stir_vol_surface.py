# src/volatility_surface/stir_vol_surface.py

import QuantLib as ql
import pandas as pd
import datetime
import numpy as np
import re  #
from typing import Any, Dict, List, Optional, Union
from src.volatility_surface.base_vol_surface import BaseVolatilitySurface
from src.data_manager import b_con  #
from src.date_utils import to_bbg_date_str, datetime_to_ql  #
from src.quantlib_utils import get_ql_yield_term_structure_handle, get_ql_black_vol_term_structure_handle  #
from src.instruments.option import Option  # To price individual options and get Greeks
from src.curves.ois_curve_builder import OISCurveBuilder  # For STIR pricing (discounting)


class StirVolSurface(BaseVolatilitySurface):
    """
    Builds a volatility surface for STIR (Short Term Interest Rate) options.

    This builder fetches option chain data, market prices, calculates implied volatilities
    and Greeks using QuantLib's Black-Scholes-Merton model, adapted for STIR futures.
    """

    def __init__(self, currency_code: str,
                 underlying_ticker: str,  # Bloomberg ticker of the underlying STIR future (e.g., 'EDH2 Comdty')
                 valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the StirVolSurface builder.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "EUR").
            underlying_ticker (str): Bloomberg ticker of the underlying STIR future.
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the surface is built.
        """
        super().__init__(currency_code, underlying_ticker, valuation_date_input)

        # STIR future specific conventions
        self._fut_conv = self._convention_config.get('futures_conventions', {}).get('stir_futures', {})  #

        # Common option pricing parameters from general settings
        self._risk_free_rate_ticker = self._convention_config.get('general_settings', {}).get(
            'default_risk_free_rate_ticker')  #
        self._dividend_rate_for_options = self._convention_config.get('general_settings', {}).get('default_div_rate')  #
        self._default_option_max_vol = self._convention_config.get('general_settings', {}).get(
            'default_option_max_vol')  #

        # For STIR options, the discount curve is critical and usually OIS
        self._ois_discount_curve: Optional[ql.YieldTermStructureHandle] = None
        self._setup_ois_discount_curve()

    def _setup_ois_discount_curve(self) -> None:
        """
        Sets up the OIS discount curve needed for STIR option pricing.
        """
        try:
            # Assumes the discount curve index is defined in currency conventions
            dc_index = self._convention_config.get('dc_index')  #
            if not dc_index:
                raise ValueError(f"Discount curve index not specified for {self.currency_code} in conventions.")

            ois_builder = OISCurveBuilder(dc_index, self.valuation_date)  #
            ois_builder.build_curve()  #
            self._ois_discount_curve = ql.YieldTermStructureHandle(ois_builder.curve)  #
            print(f"OIS discount curve ({dc_index}) built for STIR volatility surface.")
        except Exception as e:
            raise RuntimeError(f"Failed to build OIS discount curve for STIR options: {e}")

    def _fetch_market_data(self, chain_ticker_base: str, chain_length: List[int]) -> None:
        """
        Fetches underlying STIR future price, option chain data, and option market prices.

        Args:
            chain_ticker_base (str): The base ticker for the option chain, e.g., 'EDM1'.
                                     (Note: Original code `ticker+option_type+strike` for chain lookup,
                                     then inferred `fut`.)
            chain_length (List[int]): [calls_below_atm, calls_above_atm] around ATM strike.
        """
        ref_date_bbg = to_bbg_date_str(self.valuation_date, ql_date=1)  #

        try:
            # 1. Fetch underlying STIR future price
            underlying_price_ref = b_con.ref(self.underlying_ticker, 'PX_LAST')  #
            self.market_data['underlying_price'] = underlying_price_ref['value'].iloc[0]  #

            # 2. Get option chain for the underlying STIR future
            # The underlying ticker is often the future itself, so use that for OPT_CHAIN.
            option_chain_raw = b_con.bulkref(self.underlying_ticker, 'OPT_CHAIN')  #

            # Parse option chain data
            df_chain = pd.DataFrame(option_chain_raw['value'])  #
            df_chain.columns = ['FullTicker']  #
            df_chain['ticker_root'] = df_chain['FullTicker'].str[:4]  # e.g., 'EDH2'
            df_chain['option_type'] = df_chain['FullTicker'].str[4:5]  # 'C' or 'P'
            df_chain['strike'] = df_chain['FullTicker'].apply(lambda x: float(x[5:-6].strip()))  #

            # Filter chain by the specific `chain_ticker_base` (e.g., EDH2) and sort by strike
            df_chain = df_chain[df_chain['ticker_root'] == chain_ticker_base].sort_values(by='strike').reset_index(
                drop=True)  #

            if df_chain.empty:
                raise ValueError(f"No option chain found for {chain_ticker_base}.")

            # Determine the ATM strike for STIRs (often 100 - Rate)
            # Original code `centre_strike = np.round(spot*2)/2` or `np.round(spot)`
            # For STIRs, typically `np.round(spot * 4) / 4` for 0.25 tick sizes, or `np.round(spot * 8) / 8` for 0.125.
            # Assuming typical 0.25 or 0.125 increments for STIR options.
            # Example: 98.00, 98.25, 98.50.
            atm_strike_estimate = round(
                self.market_data['underlying_price'] * 4) / 4  # # Assumed 0.25 tick increment for ATM

            # Find closest strikes to ATM for calls and puts
            call_chain = df_chain[df_chain['option_type'] == 'C']  #
            put_chain = df_chain[df_chain['option_type'] == 'P']  #

            if call_chain.empty or put_chain.empty:
                raise ValueError("Option chain must contain both Calls and Puts.")

            # Select strikes around ATM based on chain_length
            # `chain_length` from config: [calls_below_atm, calls_above_atm]
            call_atm_idx_candidates = call_chain[(call_chain['strike'] - atm_strike_estimate).abs() == (
                        call_chain['strike'] - atm_strike_estimate).abs().min()].index  # Find actual ATM
            put_atm_idx_candidates = put_chain[(put_chain['strike'] - atm_strike_estimate).abs() == (
                        put_chain['strike'] - atm_strike_estimate).abs().min()].index  # Find actual ATM

            if call_atm_idx_candidates.empty or put_atm_idx_candidates.empty:
                # Fallback if exact ATM strike isn't in chain (e.g., if ATM is 98.125 and strikes are 98.00, 98.25)
                call_atm_idx = (call_chain['strike'] - atm_strike_estimate).abs().argsort()[:1].iloc[0]
                put_atm_idx = (put_chain['strike'] - atm_strike_estimate).abs().argsort()[:1].iloc[0]
            else:
                call_atm_idx = call_atm_idx_candidates[0]  # Pick first if multiple
                put_atm_idx = put_atm_idx_candidates[0]  # Pick first if multiple

            call_selected_strikes_df = call_chain.iloc[
                                       max(0, call_atm_idx - chain_length[0]): min(len(call_chain),
                                                                                   call_atm_idx + chain_length[1] + 1)
                                       ]  #
            put_selected_strikes_df = put_chain.iloc[
                                      max(0, put_atm_idx - chain_length[0]): min(len(put_chain),
                                                                                 put_atm_idx + chain_length[1] + 1)
                                      ]  #

            selected_options_df = pd.concat([call_selected_strikes_df, put_selected_strikes_df]).reset_index(
                drop=True)  #

            if selected_options_df.empty:
                raise ValueError("No options selected after filtering by chain_length around ATM.")

            # 3. Fetch market prices (PX_BID, PX_ASK) for selected options
            options_market_data = b_con.bdh(
                selected_options_df['FullTicker'].tolist(),  #
                ['PX_BID', 'PX_ASK'],  #
                ref_date_bbg, ref_date_bbg,  #
                longdata=True  #
            )

            # Calculate mid price and merge with selected options data
            options_market_data_mid = options_market_data.groupby(['ticker', 'date']).value.mean().reset_index()  #
            options_market_data_mid.rename(columns={'value': 'PX_MID'}, inplace=True)  #

            full_options_data = pd.merge(selected_options_df, options_market_data_mid[['ticker', 'PX_MID']],
                                         left_on='FullTicker', right_on='ticker', how='inner')  #
            full_options_data.drop(columns=['ticker'], inplace=True)  #

            # Filter out options with zero price (as per original code `opt_px[opt_px < 1/64] = 0.5/64`)
            full_options_data['PX_MID'].mask(full_options_data['PX_MID'] < (1 / 64), (0.5 / 64), inplace=True)  #

            # 4. Fetch expiry date (assuming all options in chain have same expiry)
            # Original code `expiry_dt = con.ref(t2[0], ['LAST_TRADEABLE_DT'])['value'][0]`
            expiry_date_ref = b_con.ref(full_options_data['FullTicker'].iloc[0], ['LAST_TRADEABLE_DT'])  #
            self.market_data['expiry_date'] = datetime_to_ql(expiry_date_ref['value'].iloc[0])  #

            self.market_data['option_chain_data'] = full_options_data  # Store the processed option chain data
            self.market_data['atm_strike'] = atm_strike_estimate  #

        except Exception as e:
            raise RuntimeError(f"Failed to fetch market data for STIR vol surface for {self.underlying_ticker}: {e}")

    def build_surface(self, chain_ticker_base: str, chain_length: List[int]) -> None:
        """
        Builds the volatility surface by calculating implied volatilities and Greeks
        for each option in the selected chain.

        Args:
            chain_ticker_base (str): A base ticker for the option chain lookup.
            chain_length (List[int]): Defines the number of strikes to fetch around ATM.
        """
        self._fetch_market_data(chain_ticker_base, chain_length)  #

        option_chain_data = self.market_data.get('option_chain_data')  #
        underlying_price = self.market_data.get('underlying_price')  #
        expiry_date = self.market_data.get('expiry_date')  #

        if option_chain_data.empty or underlying_price is None or expiry_date is None:
            raise ValueError("Market data not sufficient to build volatility surface.")

        # Initialize lists for implied vols and Greeks
        implied_vols = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        bs_prices = []

        # Fetch common risk-free rate and dividend yield (0.0 for futures)
        risk_free_rate_ref = b_con.ref(self._risk_free_rate_ticker, 'PX_LAST')  #
        risk_free_rate = risk_free_rate_ref['value'].iloc[0] / 100.0  #
        dividend_rate = self._dividend_rate_for_options  #

        # For STIRs, 'ATM_K' often represents 100 - Strike, or 100 - Futures Price.
        # Original code uses `bond_fut_yield` for `ATM_K`, which calculates yield distance.
        # For STIRs, ATM distance is typically simpler, based on price or implied rate.
        # Assuming `ATM_K` here is a simple price difference from underlying.
        option_chain_data['ATM_K'] = (
                    self.market_data['underlying_price'] - option_chain_data['strike'])  # # Simplified ATM_K for STIRs

        # Iterate through each option to calculate implied volatility and Greeks
        for idx, row in option_chain_data.iterrows():  #
            option_price = row['PX_MID']  #
            strike = row['strike']  #
            option_type_str = row['option_type']  #

            # Create an Option instrument for current option
            option_inst = Option(
                currency_code=self.currency_code,  #
                option_type=option_type_str,  #
                strike_price=strike,  #
                expiry_date_input=expiry_date,  #
                underlying_ticker=self.underlying_ticker,  #
                valuation_date_input=self.valuation_date  #
            )

            # Set underlying price (if different from default fetched by Option.__init__)
            option_inst.set_underlying_price(underlying_price)  #
            option_inst.set_risk_free_rate(risk_free_rate)  #

            # Calculate implied volatility
            initial_vol_guess = 0.05  #
            implied_vol = option_inst.calculate_implied_volatility(option_price, guess=initial_vol_guess)  #
            implied_vols.append(implied_vol)  #

            # Set the calculated implied volatility to price the option and get Greeks
            option_inst.set_volatility(implied_vol / 100.0)  # Convert back to decimal for QuantLib

            # Calculate Greeks
            bs_prices.append(option_inst.calculate_npv())  #
            deltas.append(option_inst.calculate_delta())  #
            gammas.append(option_inst.calculate_gamma())  #
            thetas.append(option_inst.calculate_theta())  #
            vegas.append(option_inst.calculate_vega())  #

        # Populate the surface_data DataFrame
        self._surface_data = option_chain_data.copy()  #
        self._surface_data['px'] = self._surface_data['PX_MID']  # Original 'px' column was mid-price
        self._surface_data['bs_px'] = bs_prices  #
        self._surface_data['iv'] = implied_vols  #
        self._surface_data['delta'] = deltas  #
        self._surface_data['gamma'] = gammas  #
        self._surface_data['theta'] = thetas  #
        self._surface_data['vega'] = vegas  #

        # Add price formatting as per original
        from src.financial_math_utils import px_opt_ticks  #
        self._surface_data['px_64'] = self._surface_data['px'].apply(px_opt_ticks)  #

        print(f"Volatility surface for {self.underlying_ticker} built successfully.")

    def interpolate_volatility(self, target_value: float, axis: str = 'ATM_K', method: str = 'linear') -> float:
        """
        Interpolates volatility from the built surface for a given value (strike or delta or ATM_K).

        Args:
            target_value (float): The target value for interpolation.
            axis (str): The axis to interpolate on ('ATM_K', 'strike', or 'delta').
            method (str): Interpolation method ('linear' or 'spline').

        Returns:
            float: The interpolated implied volatility.

        Raises:
            ValueError: If the surface is not built or interpolation fails.
        """
        if self._surface_data.empty:
            raise RuntimeError("Volatility surface not built. Call build_surface() first.")  #

        if axis not in self._surface_data.columns:
            raise ValueError(f"Interpolation axis '{axis}' not found in surface data.")  #

        df_for_interp = self._surface_data.sort_values(by=axis)  #

        if df_for_interp.empty:
            raise ValueError("Volatility surface data is empty for interpolation.")  #

        x_values = df_for_interp[axis].values  #
        y_values = df_for_interp['iv'].values  #

        # Handle cases where interpolation value is outside range
        if target_value < x_values.min():
            print(
                f"Warning: Target value {target_value} is below the minimum {axis} in the surface. Using min volatility.")
            return y_values[x_values.argmin()]
        elif target_value > x_values.max():
            print(
                f"Warning: Target value {target_value} is above the maximum {axis} in the surface. Using max volatility.")
            return y_values[x_values.argmax()]

        if method == 'linear':  #
            return float(np.interp(target_value, x_values, y_values))  #
        elif method == 'spline':  #
            from scipy.interpolate import CubicSpline  #
            spl = CubicSpline(x_values, y_values)  #
            return float(spl(target_value))  #
        else:
            raise ValueError(f"Unsupported interpolation method: {method}.")  #