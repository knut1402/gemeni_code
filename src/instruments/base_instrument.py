# src/instruments/base_instrument.py

from abc import ABC, abstractmethod
import QuantLib as ql
import datetime
from typing import Any, Dict, Optional, Union
from src.quantlib_utils import setup_ql_settings, get_ql_date
from src.date_utils import ql_to_datetime


class BaseInstrument(ABC):
    """
    Abstract base class for all financial instruments.

    Defines common properties and abstract methods for instrument valuation.
    """

    def __init__(self, currency_code: str,
                 valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the base instrument with a currency and sets up QuantLib settings.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "GBP").
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the instrument is valued. Can be a string, int (days offset),
                datetime, or ql.Date. If None, uses today's date.
        """
        self.currency_code = currency_code
        self._valuation_date_input = valuation_date_input
        self._valuation_date: ql.Date = None
        self._ql_engine: Optional[ql.PricingEngine] = None
        self._ql_instrument: Optional[ql.Instrument] = None
        self._market_data: Dict[str, Any] = {}
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

        # Calendars and currencies are often defined at the currency level in config
        self._ql_calendar: ql.Calendar = self._convention_config.get('ql_calendar_obj')
        self._ql_currency: ql.Currency = self._convention_config.get('ql_currency_obj')

        if not self._ql_calendar:
            raise ValueError(f"QuantLib calendar not configured for {self.currency_code} in conventions.")
        if not self._ql_currency:
            raise ValueError(f"QuantLib currency object not configured for {self.currency_code} in conventions.")

    @abstractmethod
    def _fetch_market_data(self) -> None:
        """
        Abstract method to fetch necessary market data for the instrument.
        Populates self._market_data.
        """
        pass

    @abstractmethod
    def _setup_ql_instrument(self) -> None:
        """
        Abstract method to set up the QuantLib instrument object and its pricing engine.
        Populates self._ql_instrument and self._ql_engine.
        """
        pass

    @abstractmethod
    def calculate_npv(self) -> float:
        """
        Abstract method to calculate the Net Present Value (NPV) of the instrument.
        """
        pass

    @property
    def valuation_date(self) -> ql.Date:
        """
        Returns the QuantLib valuation date for the instrument.
        """
        return self._valuation_date

    @property
    def calendar(self) -> ql.Calendar:
        """
        Returns the QuantLib calendar used for the instrument.
        """
        return self._ql_calendar

    @property
    def currency(self) -> ql.Currency:
        """
        Returns the QuantLib currency object for the instrument.
        """
        return self._ql_currency

    @property
    def ql_instrument(self) -> Optional[ql.Instrument]:
        """
        Returns the QuantLib instrument object after setup.
        """
        return self._ql_instrument

    @property
    def ql_engine(self) -> Optional[ql.PricingEngine]:
        """
        Returns the QuantLib pricing engine for the instrument.
        """
        return self._ql_engine

    @property
    def market_data(self) -> Dict[str, Any]:
        """
        Returns the raw market data used for the instrument.
        """
        return self._market_data