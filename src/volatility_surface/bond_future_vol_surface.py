# src/volatility_surface/bond_future_vol_surface.py

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
from src.financial_math_utils import bond_fut_yield  # Inferred utility: `bond_fut_yield` is crucial here.


class BondFutureVolSurface(BaseVolatilitySurface):
    """
    Builds a volatility surface for bond futures options.

    This builder fetches option chain data, market prices, calculates implied volatilities
    and Greeks using QuantLib's Black-Scholes-Merton model, and processes the data
    to form a comprehensive volatility surface.
    """

    def __init__(self, currency_code: str,
                 underlying_ticker: str,  # Bloomberg ticker of the underlying bond future (e.g., 'USU1 Comdty')
                 valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the BondFutureVolSurface builder.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "GBP").
            underlying_ticker (str): Bloomberg ticker of the underlying bond future.
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the surface is built.
        """
        super().__init__(currency_code, underlying_ticker, valuation_date_input)

        # Bond future specific conventions
        self._fut_conv = self._convention_config.get('futures_conventions', {}).get('bond_futures', {})  #

        # Common option pricing parameters from general settings
        self._risk_free_rate_ticker = self._convention_config.get('general_settings', {}).get(
            'default_risk_free_rate_ticker')  #
        self._dividend_rate_for_options = self._convention_config.get('general_settings', {}).get('default_div_rate')  #
        self._default_option_max_vol = self._convention_config.get('general_settings', {}).get(
            'default_option_max_vol')  #

    def _fetch_market_data(self, chain_ticker_base: str, chain_length: List[int]) -> None:
        """
        Fetches underlying future price, option chain data, and option market prices.

        Args:
            chain_ticker_base (str): The base ticker for the option chain, e.g., 'USU1'.
                                     (Note: Original code passes `ticker+option_type+strike` for chain lookup
                                     and then infers `fut` and `centre_strike`.)
                                     Here we simplify `chain_ticker_base` to be the root future ticker.
            chain_length (List[int]): [calls_below_atm, calls_above_atm] around ATM strike.
                                      For puts, symmetric range usually used.
        """
        ref_date_bbg = to_bbg_date_str(self.valuation_date, ql_date=1)  #

        try:
            # 1. Fetch underlying future price
            underlying_price_ref = b_con.ref(self.underlying_ticker, 'PX_LAST')  #
            self.market_data['underlying_price'] = underlying_price_ref['value'].iloc[0]  #

            # 2. Get option chain for the underlying future
            # The original code used `con.bulkref(fut, ['OPT_CHAIN'])`.
            # `fut` was derived from the first option ticker: `con.ref(t2[0], ['OPT_UNDL_TICKER'])['value'][0]+' Comdty'`
            # Here, we assume `self.underlying_ticker` is the correct `fut` for chain lookup.
            option_chain_raw = b_con.bulkref(self.underlying_ticker, 'OPT_CHAIN')  #

            # Parse option chain data
            df_chain = pd.DataFrame(option_chain_raw['value'])  #
            df_chain.columns = ['FullTicker']  #
            df_chain['ticker_root'] = df_chain['FullTicker'].str[:4]  # e.g., 'USU1'
            df_chain['option_type'] = df_chain['FullTicker'].str[4:5]  # 'C' or 'P'
            df_chain['strike'] = df_chain['FullTicker'].apply(lambda x: float(x[5:-6].strip()))  #

            # Filter chain by the specific `chain_ticker_base` (e.g., USU1) and sort by strike
            df_chain = df_chain[df_chain['ticker_root'] == chain_ticker_base].sort_values(by='strike').reset_index(
                drop=True)  #

            if df_chain.empty:
                raise ValueError(f"No option chain found for {chain_ticker_base}.")  #

            # Determine the ATM strike
            # Original code `centre_strike = np.round(spot*2)/2` or `np.round(spot)` if first fails.
            atm_strike_estimate = round(self.market_data['underlying_price'] * 2) / 2  #

            # Find closest strikes to ATM for calls and puts
            call_chain = df_chain[df_chain['option_type'] == 'C']  #
            put_chain = df_chain[df_chain['option_type'] == 'P']  #

            if call_chain.empty or put_chain.empty:  #
                raise ValueError("Option chain must contain both Calls and Puts.")  #

            # Find closest call and put strikes to the estimated ATM
            call_atm_idx = (call_chain['strike'] - atm_strike_estimate).abs().argsort()[:1].iloc[0]  #
            put_atm_idx = (put_chain['strike'] - atm_strike_estimate).abs().argsort()[:1].iloc[0]  #

            # Select strikes around ATM based on chain_length
            call_selected_strikes_df = call_chain.iloc[
                                       max(0, call_atm_idx - chain_length[0]): min(len(call_chain),
                                                                                   call_atm_idx + chain_length[1] + 1)
                                       ]  #
            put_selected_strikes_df = put_chain.iloc[
                                      max(0, put_atm_idx - chain_length[0]): min(len(put_chain),
                                                                                 put_atm_idx + chain_length[1] + 1)
                                      ]  #

            # Concatenate selected calls and puts
            selected_options_df = pd.concat([call_selected_strikes_df, put_selected_strikes_df]).reset_index(
                drop=True)  #

            if selected_options_df.empty:
                raise ValueError("No options selected after filtering by chain_length around ATM.")  #

            # 3. Fetch market prices (PX_BID, PX_ASK) for selected options
            options_market_data = b_con.bdh(
                selected_options_df['FullTicker'].tolist(),  #
                ['PX_BID', 'PX_ASK'],  #
                ref_date_bbg, ref_date_bbg,  #
                longdata=True  #
            )  #

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
            raise RuntimeError(
                f"Failed to fetch market data for bond future vol surface for {self.underlying_ticker}: {e}")  #

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
            raise ValueError("Market data not sufficient to build volatility surface.")  #

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

        # Calculate bond future yield and ATM distance for each strike
        # `bond_fut_yield` is a utility, its definition is crucial.
        # Assuming `bond_fut_yield` returns a DataFrame with 'K_Yield' and 'K_Dist'
        # based on future ticker and strikes.
        #
        # NOTE: The actual `bond_fut_yield` function from `Utilities.py` was not provided.
        # The implementation of `bond_fut_yield` is crucial for 'K_Yield' and 'K_Dist'.
        # For now, a placeholder or a simple direct calculation is used.
        # If `bond_fut_yield` involves Bloomberg calls for bond yield, it should be noted.
        #
        bond_fut_yield_data = bond_fut_yield(self.underlying_ticker,
                                             option_chain_data['strike'].tolist())  # Placeholder/Inferred
        # Assuming `bond_fut_yield` returns a DataFrame with 'K_Yield' and 'K_Dist' columns.
        option_chain_data['K_Yield'] = bond_fut_yield_data['K_Yield'] if not bond_fut_yield_data.empty else np.nan  #
        option_chain_data['ATM_K'] = bond_fut_yield_data['K_Dist'] if not bond_fut_yield_data.empty else np.nan  #

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
            # Use 0.05 as an initial guess for implied volatility (common practice)
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
        # `px_opt_ticks` from Utilities.py
        # `px_dec_to_frac` from Utilities.py
        from src.financial_math_utils import px_opt_ticks  #
        self._surface_data['px_64'] = self._surface_data['px'].apply(px_opt_ticks)  #

        print(f"Volatility surface for {self.underlying_ticker} built successfully.")  #

    def interpolate_volatility(self, target_strike_or_delta: float, method: str = 'linear') -> float:
        """
        Interpolates volatility from the built surface for a given strike or delta.

        Args:
            target_strike_or_delta (float): The target value (strike or delta).
            method (str): Interpolation method ('linear' or 'spline').

        Returns:
            float: The interpolated implied volatility.

        Raises:
            ValueError: If the surface is not built or interpolation fails.
        """
        if self._surface_data.empty:
            raise RuntimeError("Volatility surface not built. Call build_surface() first.")  #

        # Determine if interpolating by strike or delta
        # Original `build_vol_spline` uses `ATM_K` for interpolation.
        # `ATM_K` represents the distance from ATM in yield terms, or a standardized strike.
        # `shift1` from `BOND_FUTURES.py` was `100*(bond_fut_yield(v1.fut[:4], strikes)['K_Yield'] - spot_yld_sim.loc[i]['K_Yield'])`
        # This means `ATM_K` is essentially a yield difference.

        # For simplicity, if `target_strike_or_delta` is close to a delta, interpret as delta.
        # Otherwise, interpret as strike. More robust would be explicit flag.

        # Let's assume `target_strike_or_delta` directly refers to `ATM_K` or `strike` for now.
        # Original `build_vol_spline` uses `ATM_K`.
        # Its `df3['ATM_K'] = np.array(a.tab_call['ATM_K'] - a.tab_call[a.tab_call['strikes'] == sim_spot]['ATM_K'].tolist())`
        # implies interpolating on a *shifted* ATM_K axis.

        # Let's assume `target_strike_or_delta` is a value for 'ATM_K' or 'strike'.
        # For a basic linear interpolation, it needs to be sorted.

        # Original code used `np.interp(shift1[j], df5['ATM_K'], df5['iv'])`
        # where `df5` was the 2 closest points to the `shift1[j]`.

        # This is a general interpolation function.
        # It needs to know WHICH axis to interpolate on (strike, delta, ATM_K).
        # Let's default to interpolating on `ATM_K` as per original `build_vol_spline` usage.
        #
        # NOTE: The implementation of `interpolate_volatility` is crucial.
        # It needs to correctly handle interpolation axis (`ATM_K` or `strikes` or `delta`)
        # and how to combine call/put sides if symmetric interpolation is desired.

        interp_axis = 'ATM_K'  # Default as per `build_vol_spline` usage

        # Filter for relevant options (e.g., calls for positive ATM_K, puts for negative)
        # Or just use the whole surface sorted by `interp_axis`.
        df_for_interp = self._surface_data.sort_values(by=interp_axis)  #

        if df_for_interp.empty:
            raise ValueError("Volatility surface data is empty for interpolation.")  #

        x_values = df_for_interp[interp_axis].values  #
        y_values = df_for_interp['iv'].values  #

        if method == 'linear':  #
            return float(np.interp(target_strike_or_delta, x_values, y_values))  #
        elif method == 'spline':  #
            # For spline, more robust interpolation (e.g., scipy.interpolate.CubicSpline) would be used.
            # QuantLib also has interpolation methods for term structures.
            # Placeholder for actual spline implementation.
            from scipy.interpolate import CubicSpline  #
            spl = CubicSpline(x_values, y_values)  #
            return float(spl(target_strike_or_delta))  #
        else:
            raise ValueError(f"Unsupported interpolation method: {method}.")  #