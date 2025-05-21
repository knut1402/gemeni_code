# src/pricers/option_pricer.py

import QuantLib as ql
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.instruments.option import Option  #
from src.volatility_surface.base_vol_surface import BaseVolatilitySurface  #
from src.volatility_surface.bond_future_vol_surface import BondFutureVolSurface  #
from src.volatility_surface.stir_vol_surface import StirVolSurface  #


class OptionPricer:
    """
    Provides pricing functionalities for financial options.

    This class can price individual options or a collection of options,
    leveraging the Option instrument class and a volatility surface.
    """

    def __init__(self):
        """
        Initializes the OptionPricer.
        """
        pass

    def get_option_price_and_greeks(self,
                                    currency_code: str,
                                    option_type: str,  # 'C' or 'P'
                                    strike_price: float,
                                    expiry_date_input: Union[str, datetime.datetime, ql.Date],
                                    underlying_ticker: str,
                                    implied_volatility_pct: float,  # Implied volatility in percentage (e.g., 15.0)
                                    valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None
                                    ) -> Dict[str, float]:
        """
        Calculates the price (NPV) and Greeks (Delta, Gamma, Theta, Vega) for a single option
        given an implied volatility.

        Args:
            currency_code (str): The three-letter currency code.
            option_type (str): 'C' for Call, 'P' for Put.
            strike_price (float): The strike price of the option.
            expiry_date_input (Union[str, datetime.datetime, ql.Date]): The option's expiry date.
            underlying_ticker (str): Bloomberg ticker of the underlying future.
            implied_volatility_pct (float): The implied volatility in percentage (e.g., 15.0 for 15%).
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]): The valuation date.

        Returns:
            Dict[str, float]: A dictionary containing 'NPV', 'Delta', 'Gamma', 'Theta', 'Vega'.
        """
        try:
            # Instantiate Option instrument
            option_inst = Option(
                currency_code=currency_code,  #
                option_type=option_type,  #
                strike_price=strike_price,  #
                expiry_date_input=expiry_date_input,  #
                underlying_ticker=underlying_ticker,  #
                valuation_date_input=valuation_date_input  #
            )

            # Set the implied volatility for pricing
            option_inst.set_volatility(implied_volatility_pct / 100.0)  # Convert to decimal for QuantLib

            # Calculate and return metrics
            return {
                'NPV': option_inst.calculate_npv(),  #
                'Delta': option_inst.calculate_delta(),  #
                'Gamma': option_inst.calculate_gamma(),  #
                'Theta': option_inst.calculate_theta(),  #
                'Vega': option_inst.calculate_vega()  #
            }
        except Exception as e:
            print(f"Error pricing option {underlying_ticker} {option_type} {strike_price}: {e}")
            return {'NPV': np.nan, 'Delta': np.nan, 'Gamma': np.nan, 'Theta': np.nan, 'Vega': np.nan}

    def price_options_from_vol_surface(self,
                                       currency_code: str,
                                       underlying_ticker: str,
                                       option_types: List[str],
                                       strikes: List[float],
                                       implied_vols_pct: List[float],  # List of implied volatilities (percentage)
                                       expiry_date_input: Union[str, datetime.datetime, ql.Date],
                                       valuation_date_input: Optional[
                                           Union[str, int, datetime.datetime, ql.Date]] = None
                                       ) -> pd.DataFrame:
        """
        Prices a list of options (e.g., a strategy's components) given their strikes, types,
        and corresponding implied volatilities. This is similar to the `get_bsm_px_from_vol`
        logic in the original BOND_FUTURES.py.

        Args:
            currency_code (str): The three-letter currency code.
            underlying_ticker (str): Bloomberg ticker of the underlying future.
            option_types (List[str]): List of option types ('C' or 'P').
            strikes (List[float]): List of strike prices.
            implied_vols_pct (List[float]): List of implied volatilities (percentage) for each option.
            expiry_date_input (Union[str, datetime.datetime, ql.Date]): The option's expiry date.
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]): The valuation date.

        Returns:
            pd.DataFrame: A DataFrame with calculated 'NPV', 'Delta', 'Gamma', 'Theta', 'Vega' for each option.
        """
        results = []
        for i in range(len(option_types)):
            metrics = self.get_option_price_and_greeks(
                currency_code=currency_code,
                option_type=option_types[i],
                strike_price=strikes[i],
                expiry_date_input=expiry_date_input,
                underlying_ticker=underlying_ticker,
                implied_volatility_pct=implied_vols_pct[i],
                valuation_date_input=valuation_date_input
            )
            results.append(metrics)
        return pd.DataFrame(results)