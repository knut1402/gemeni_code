# src/curves/base_curve.py

from abc import ABC, abstractmethod
import QuantLib as ql
import datetime
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from src.quantlib_utils import setup_ql_settings, get_ql_date
from src.date_utils import ql_to_datetime


class BaseCurve(ABC):
    """
    Abstract base class for all financial curve builders (e.g., OIS, Swap, Inflation).

    Defines the common interface for curve construction, properties, and methods.
    """

    def __init__(self, currency_code: str,
                 reference_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the base curve with a currency and sets up QuantLib settings.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "GBP").
            reference_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the curve is built. Can be a string, int (days offset),
                datetime, or ql.Date. If None, uses today's date.
        """
        self.currency_code = currency_code
        self._reference_date_input = reference_date_input
        self._reference_date: ql.Date = None
        self._ql_curve: Optional[ql.YieldTermStructure] = None
        self._ql_calendar: Optional[ql.Calendar] = None
        self._ql_currency: Optional[ql.Currency] = None
        self._market_data: Dict[str, Any] = {}
        self._nodes: List[Any] = []  # QuantLib nodes for inspection
        self._convention_config: Dict[str, Any] = {}  # Configuration for the specific currency/index

        self._setup_initial_ql_settings()

    def _setup_initial_ql_settings(self) -> None:
        """
        Sets up the QuantLib evaluation date and retrieves currency/calendar from config.
        """
        setup_ql_settings(self._reference_date_input)
        self._reference_date = ql.Settings.instance().evaluationDate

        from src.config import config  # Import here to avoid circular dependency on first load
        self._convention_config = config.get_currency_config(self.currency_code)

        self._ql_calendar = self._convention_config.get('ql_calendar_obj')
        self._ql_currency = self._convention_config.get('ql_currency_obj')

        if not self._ql_calendar:
            raise ValueError(f"QuantLib calendar not configured for {self.currency_code} in conventions.")
        if not self._ql_currency:
            raise ValueError(f"QuantLib currency object not configured for {self.currency_code} in conventions.")

    @abstractmethod
    def _fetch_market_data(self) -> None:
        """
        Abstract method to fetch necessary market data from Bloomberg or other sources.
        Populates self._market_data.
        """
        pass

    @abstractmethod
    def _build_curve_helpers(self) -> List[ql.RateHelper]:
        """
        Abstract method to construct QuantLib rate helpers from market data.

        Returns:
            List[ql.RateHelper]: A list of QuantLib rate helper objects.
        """
        pass

    @abstractmethod
    def build_curve(self) -> None:
        """
        Abstract method to build the QuantLib yield curve.
        Populates self._ql_curve.
        """
        pass

    @property
    def curve(self) -> ql.YieldTermStructure:
        """
        Returns the built QuantLib yield term structure.
        """
        if self._ql_curve is None:
            raise RuntimeError("Curve has not been built yet. Call build_curve() first.")
        return self._ql_curve

    @property
    def reference_date(self) -> ql.Date:
        """
        Returns the QuantLib reference date for the curve.
        """
        return self._reference_date

    @property
    def calendar(self) -> ql.Calendar:
        """
        Returns the QuantLib calendar used for the curve.
        """
        return self._ql_calendar

    @property
    def currency(self) -> ql.Currency:
        """
        Returns the QuantLib currency object for the curve.
        """
        return self._ql_currency

    @property
    def market_data(self) -> Dict[str, Any]:
        """
        Returns the raw market data used to build the curve.
        """
        return self._market_data

    @property
    def nodes(self) -> List[Any]:
        """
        Returns the QuantLib nodes of the built curve.
        """
        if self._ql_curve and hasattr(self._ql_curve, 'nodes'):
            return self._ql_curve.nodes()
        return self._nodes  # Fallback if specific curve type doesn't have .nodes()

    def get_curve_rates(self, tenors_in_years: List[float], day_count: ql.DayCounter,
                        frequency: ql.Frequency) -> pd.Series:
        """
        Retrieves forward rates from the built curve for specified tenors.

        Args:
            tenors_in_years (List[float]): List of tenors in years (e.g., [1.0, 5.0, 10.0]).
            day_count (ql.DayCounter): Day count convention for the rates.
            frequency (ql.Frequency): Frequency for the rates.

        Returns:
            pd.Series: A Series of rates indexed by tenor.
        """
        if self._ql_curve is None:
            raise RuntimeError("Curve has not been built yet. Call build_curve() first.")

        rates = []
        for tenor in tenors_in_years:
            end_date = self.calendar.advance(self.reference_date, ql.Period(int(tenor * 365), ql.Days))
            # Use `ql.Simple` for simple interest, `ql.Compounded` for compounding
            rate = self._ql_curve.forwardRate(self.reference_date, end_date, day_count, ql.Simple).rate()
            rates.append(rate * 100)  # Convert to percentage
        return pd.Series(rates, index=[f"{t}Y" for t in tenors_in_years], name="Rates")