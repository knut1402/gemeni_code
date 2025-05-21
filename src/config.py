# src/config.py

import yaml
from pathlib import Path
import QuantLib as ql
import pandas as pd
from typing import Dict, Any, List, Optional

# Define the path to the conventions YAML file
CONVENTIONS_FILE = Path(__file__).parent.parent / "data" / "conventions.yaml"


class AppConfig:
    """
    Manages application-wide configurations loaded from a YAML file.

    This class provides structured access to financial market conventions
    (currencies, inflation indices, futures), Bloomberg settings, and
    general application parameters. It uses QuantLib objects for calendars
    and currencies where appropriate.

    Attributes:
        _data (Dict[str, Any]): The raw configuration data loaded from the YAML.
        currencies (Dict[str, Any]): Dictionary of currency-specific conventions.
        inflation_indices (Dict[str, Any]): Dictionary of inflation index-specific conventions.
        futures_conventions (Dict[str, Any]): Dictionary of futures-specific conventions.
        general_settings (Dict[str, Any]): Dictionary of general application settings.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """
        Ensures a single instance of AppConfig is used (Singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_config()
            cls._instance._set_pandas_options()
        return cls._instance

    def _load_config(self) -> None:
        """
        Loads configuration data from the YAML file.
        Initializes QuantLib objects for calendars and currencies.
        """
        if not CONVENTIONS_FILE.exists():
            raise FileNotFoundError(f"Conventions file not found at: {CONVENTIONS_FILE}")

        with open(CONVENTIONS_FILE, 'r') as file:
            self._data = yaml.safe_load(file)

        self.currencies = self._data.get('currencies', {})
        self.inflation_indices = self._data.get('inflation_indices', {})
        self.futures_conventions = self._data.get('futures_conventions', {})
        self.general_settings = self._data.get('general_settings', {})

        # Initialize QuantLib objects for currencies
        for ccy_code, ccy_data in self.currencies.items():
            ql_currency_str = ccy_data.get('ql_currency')
            if ql_currency_str:
                try:
                    # Dynamically get QuantLib currency object
                    ccy_data['ql_currency_obj'] = getattr(ql, ql_currency_str)()
                except AttributeError:
                    print(f"Warning: QuantLib currency '{ql_currency_str}' not found for {ccy_code}.")
                    ccy_data['ql_currency_obj'] = None

            ql_calendar_str = ccy_data.get('calendar')
            if ql_calendar_str:
                try:
                    ccy_data['ql_calendar_obj'] = getattr(ql, ql_calendar_str)()
                except AttributeError:
                    print(f"Warning: QuantLib calendar '{ql_calendar_str}' not found for {ccy_code}.")
                    ccy_data['ql_calendar_obj'] = None

    def _set_pandas_options(self) -> None:
        """
        Sets global pandas display options based on configurations.
        """
        pd.set_option('display.max_columns', self.general_settings.get('default_pandas_max_columns', None))
        pd.set_option('display.width', self.general_settings.get('default_pandas_display_width', None))

    def get_currency_config(self, ccy_code: str) -> Dict[str, Any]:
        """
        Retrieves configuration for a specific currency.

        Args:
            ccy_code (str): The currency code (e.g., "USD", "GBP").

        Returns:
            Dict[str, Any]: A dictionary containing the currency's conventions.

        Raises:
            ValueError: If the currency code is not found in the configuration.
        """
        if ccy_code not in self.currencies:
            raise ValueError(f"Currency '{ccy_code}' not found in conventions.")
        return self.currencies[ccy_code]

    def get_inflation_config(self, index_name: str) -> Dict[str, Any]:
        """
        Retrieves configuration for a specific inflation index.

        Args:
            index_name (str): The name of the inflation index (e.g., "HICPxT", "UKRPI").

        Returns:
            Dict[str, Any]: A dictionary containing the index's conventions.

        Raises:
            ValueError: If the inflation index name is not found in the configuration.
        """
        if index_name not in self.inflation_indices:
            raise ValueError(f"Inflation index '{index_name}' not found in conventions.")
        return self.inflation_indices[index_name]

    def get_futures_config(self, future_type: str) -> Dict[str, Any]:
        """
        Retrieves configuration for a specific type of futures.

        Args:
            future_type (str): The type of futures (e.g., "bond_futures", "stir_futures").

        Returns:
            Dict[str, Any]: A dictionary containing the futures conventions.

        Raises:
            ValueError: If the future type is not found in the configuration.
        """
        if future_type not in self.futures_conventions:
            raise ValueError(f"Future type '{future_type}' not found in conventions.")
        return self.futures_conventions[future_type]


# Instantiate the singleton config object for easy access
config = AppConfig()