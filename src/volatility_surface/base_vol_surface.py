# src/volatility_surface/base_vol_surface.py

from abc import ABC, abstractmethod
import QuantLib as ql
import pandas as pd
import datetime
from typing import Any, Dict, List, Optional, Union
from src.quantlib_utils import setup_ql_settings, get_ql_date  #
from src.date_utils import ql_to_datetime  #


class BaseVolatilitySurface(ABC):
    """
    Abstract base class for all volatility surface builders (e.g., Bond Futures, STIR).

    Defines the common interface for surface construction, properties, and methods.
    """

    def __init__(self, currency_code: str,
                 underlying_ticker: str,  # Bloomberg ticker of the underlying future
                 valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the base volatility surface with a currency, underlying, and sets up QuantLib.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "GBP").
            underlying_ticker (str): Bloomberg ticker of the underlying future (e.g., 'USU1 Comdty').
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the surface is built. Can be a string, int (days offset),
                datetime, or ql.Date. If None, uses today's date.
        """
        self.currency_code = currency_code
        self.underlying_ticker = underlying_ticker
        self._valuation_date_input = valuation_date_input
        self._valuation_date: ql.Date = None
        self._ql_calendar: Optional[ql.Calendar] = None
        self._ql_currency: Optional[ql.Currency] = None
        self._surface_data: pd.DataFrame = pd.DataFrame()  # Stores implied vols, Greeks, etc.
        self._market_data: Dict[str, Any] = {}  # Raw market data used for construction
        self._convention_config: Dict[str, Any] = {}

        self._setup_initial_ql_settings()

    def _setup_initial_ql_settings(self) -> None:
        """
        Sets up the QuantLib evaluation date and retrieves currency/calendar from config.
        """
        setup_ql_settings(self._valuation_date_input)
        self._valuation_date = ql.Settings.instance().evaluationDate

        from src.config import config  # Import here to avoid circular dependency
        self._convention_config = config.get_currency_config(self.currency_code)

        self._ql_calendar = self._convention_config.get('ql_calendar_obj')
        self._ql_currency = self._convention_config.get('ql_currency_obj')

        if not self._ql_calendar:
            raise ValueError(f"QuantLib calendar not configured for {self.currency_code} in conventions.")
        if not self._ql_currency:
            raise ValueError(f"QuantLib currency object not configured for {self.currency_code} in conventions.")

    @abstractmethod
    def _fetch_market_data(self, chain_ticker_base: str, chain_length: List[int]) -> None:
        """
        Abstract method to fetch necessary market data for the volatility surface.
        This typically involves retrieving option chain data (strikes, types, expiry)
        and their market prices from Bloomberg.

        Args:
            chain_ticker_base (str): A base ticker for the option chain lookup
                                     (e.g., 'USU1' for bond futures, 'EDH2' for Eurodollar).
                                     Often derived from the underlying future.
            chain_length (List[int]): Defines the number of strikes to fetch around ATM.
                                      E.g., [12, 12] means 12 strikes below and 12 above ATM for calls/puts.
        """
        pass

    @abstractmethod
    def build_surface(self, chain_ticker_base: str, chain_length: List[int]) -> None:
        """
        Abstract method to build the volatility surface.
        This involves:
        1. Fetching option chain data and market prices.
        2. Calculating implied volatilities for each option.
        3. Calculating option Greeks (Delta, Gamma, Theta, Vega).
        4. Storing the results in a structured format (e.g., self._surface_data).

        Args:
            chain_ticker_base (str): A base ticker for the option chain lookup.
            chain_length (List[int]): Defines the number of strikes to fetch around ATM.
                                      E.g., [12, 12] means 12 strikes below and 12 above ATM for calls/puts.
        """
        pass

    @abstractmethod
    def interpolate_volatility(self, target_strike_or_delta: float, method: str = 'linear') -> float:
        """
        Abstract method to interpolate volatility from the surface for a given
        strike or delta.

        Args:
            target_strike_or_delta (float): The strike or delta value for which to interpolate volatility.
            method (str): Interpolation method (e.g., 'linear', 'cubic_spline').

        Returns:
            float: The interpolated implied volatility.
        """
        pass

    @property
    def surface_data(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing the built volatility surface data
        (implied vols, Greeks, strikes, deltas, etc.).
        """
        if self._surface_data.empty:
            raise RuntimeError("Volatility surface has not been built yet. Call build_surface() first.")
        return self._surface_data

    @property
    def market_data(self) -> Dict[str, Any]:
        """
        Returns the raw market data used to build the surface.
        """
        return self._market_data

    @property
    def reference_date(self) -> ql.Date:
        """
        Returns the QuantLib reference date for the surface.
        """
        return self._valuation_date

    @property
    def underlying_price(self) -> float:
        """
        Returns the underlying future's spot price used for surface construction.
        """
        return self.market_data.get('underlying_price', np.nan)

    @property
    def expiry_date(self) -> ql.Date:
        """
        Returns the expiry date of the option chain used for the surface.
        Assumes a single expiry for the built surface.
        """
        return self.market_data.get('expiry_date', None)

    @property
    def at_the_money_strike(self) -> float:
        """
        Returns the calculated at-the-money (ATM) strike for the surface.
        """
        return self.market_data.get('atm_strike', np.nan)