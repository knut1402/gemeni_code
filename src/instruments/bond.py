# src/instruments/bond.py

import QuantLib as ql
import pandas as pd
import datetime
from typing import Any, Dict, List, Optional, Union
from src.instruments.base_instrument import BaseInstrument
from src.data_manager import b_con  #
from src.date_utils import to_bbg_date_str  #


class Bond(BaseInstrument):
    """
    Represents a generic bond instrument.

    This class fetches bond-specific market data and attributes from Bloomberg.
    It serves as a base for more specialized bond types or as a data container
    for bond-related analytics (e.g., linker carry calculations).
    """

    def __init__(self, isin: str, currency_code: str,
                 valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the Bond instrument.

        Args:
            isin (str): The ISIN (International Securities Identification Number) of the bond.
            currency_code (str): The three-letter currency code (e.g., "USD", "GBP").
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the bond is valued. If None, uses today's date.
        """
        super().__init__(currency_code, valuation_date_input)
        self.isin = isin
        self._bond_attributes: Dict[str, Any] = {}  # Stores general bond attributes

        self._fetch_market_data()  # Fetch bond attributes and prices

    def _fetch_market_data(self) -> None:
        """
        Fetches necessary bond attributes and prices from Bloomberg.
        Attributes typically needed for bond calculations include:
        Maturity, Coupon, Coupon Frequency, Days Accrued, Days to Next Coupon,
        Last Price, Close Price (1D), Settlement Date, Close Date, Days to Settle, Base CPI.
        """
        ref_date_bbg = to_bbg_date_str(self.valuation_date, ql_date=1)

        bond_fields = [
            'MATURITY', 'CPN', 'CPN_FREQ', 'DAYS_ACC', 'DAYS_TO_NEXT_COUPON',
            'PX_LAST', 'PX_CLOSE_1D', 'SETTLE_DT', 'PX_CLOSE_DT', 'DAYS_TO_SETTLE', 'BASE_CPI'
        ]  #

        try:
            # Fetch general bond attributes
            bond_data = b_con.ref(self.isin, bond_fields)
            # Convert to a dictionary for easier access
            for _, row in bond_data.iterrows():  #
                self._bond_attributes[row['field']] = row['value']  #

            # Convert specific date strings from Bloomberg to datetime objects if present
            if 'SETTLE_DT' in self._bond_attributes and isinstance(self._bond_attributes['SETTLE_DT'], datetime.date):
                self._bond_attributes['SETTLE_DT'] = datetime.datetime.combine(self._bond_attributes['SETTLE_DT'],
                                                                               datetime.time())
            if 'PX_CLOSE_DT' in self._bond_attributes and isinstance(self._bond_attributes['PX_CLOSE_DT'],
                                                                     datetime.date):
                self._bond_attributes['PX_CLOSE_DT'] = datetime.datetime.combine(self._bond_attributes['PX_CLOSE_DT'],
                                                                                 datetime.time())

        except Exception as e:
            raise RuntimeError(f"Failed to fetch market data for bond {self.isin}: {e}")

    def _setup_ql_instrument(self) -> None:
        """
        For a generic Bond class, a specific QuantLib instrument object (like FixedRateBond)
        might be set up here if pricing details are available. For now, this remains a placeholder
        as the original code primarily fetched data and calculated yields externally.
        """
        pass  # Placeholder for actual QuantLib bond instrument setup

    def calculate_npv(self) -> float:
        """
        Calculates the Net Present Value (NPV) of the bond.
        This method needs a QuantLib pricing engine and a yield curve.
        Currently, the original code calculates yield externally via Bloomberg.
        This would be implemented once a proper bond pricing engine is integrated.
        """
        raise NotImplementedError("NPV calculation for generic bond not yet implemented. Requires a yield curve.")

    def get_attribute(self, attribute_name: str) -> Any:
        """
        Retrieves a specific attribute of the bond.

        Args:
            attribute_name (str): The name of the attribute (e.g., 'CPN', 'PX_LAST').

        Returns:
            Any: The value of the attribute.

        Raises:
            KeyError: If the attribute is not found.
        """
        if attribute_name not in self._bond_attributes:
            raise KeyError(f"Bond attribute '{attribute_name}' not found for ISIN {self.isin}.")
        return self._bond_attributes[attribute_name]

    def calculate_yield_from_price(self, price: float,
                                   settlement_date: Optional[Union[str, datetime.datetime, ql.Date]] = None,
                                   bond_field: str = 'YLD_YTM_BID') -> float:
        """
        Calculates the yield-to-maturity of the bond given a clean price and settlement date.
        Uses Bloomberg's YTM calculation via overrides.

        Args:
            price (float): The clean price of the bond.
            settlement_date (Optional[Union[str, datetime.datetime, ql.Date]]):
                The settlement date for the yield calculation. If None, uses the bond's default
                settlement date from fetched attributes, or falls back to valuation date + settlement days.
            bond_field (str): The Bloomberg field for yield calculation (e.g., 'YLD_YTM_BID', 'YAS_BOND_YLD').

        Returns:
            float: The yield-to-maturity in decimal.
        """
        ovrds = [('PX_BID', str(price))]  # Use PX_BID as override for price

        final_settlement_date_ql = None
        if settlement_date:
            final_settlement_date_ql = get_ql_date(settlement_date)
        elif 'SETTLE_DT' in self._bond_attributes and self._bond_attributes['SETTLE_DT']:
            # Use the fetched settlement date if available
            final_settlement_date_ql = datetime_to_ql(self._bond_attributes['SETTLE_DT'])  #
        else:
            # Fallback to valuation date + settlement days from currency config
            sett_days = self._convention_config.get('settlement_days', 2)  #
            final_settlement_date_ql = self.calendar.advance(self.valuation_date, sett_days, ql.Days)  #

        if final_settlement_date_ql:
            ovrds.append(('SETTLE_DT', to_bbg_date_str(final_settlement_date_ql, ql_date=1)))  #
        else:
            raise ValueError("Could not determine a settlement date for yield calculation.")  #

        try:
            yield_data = b_con.ref(self.isin, [bond_field], ovrds=ovrds)  #
            return yield_data['value'].iloc[0]  #
        except Exception as e:
            raise RuntimeError(
                f"Failed to calculate yield for bond {self.isin} with price {price} and settlement date {final_settlement_date_ql}: {e}")  #