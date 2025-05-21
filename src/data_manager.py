# src/data_manager.py

import pdblp
import pandas as pd
from pathlib import Path
import pickle
import os
import time
from typing import Dict, Any, List, Optional, Union
from src.config import config  # Import the global config object


class BloombergConnection:
    """
    Manages a single Bloomberg API connection using pdblp.

    This class implements the Singleton pattern to ensure only one Bloomberg
    connection is active at any time. It provides methods to interact
    with the Bloomberg terminal.

    Attributes:
        _instance (BloombergConnection): The singleton instance.
        _con (pdblp.BCon): The Bloomberg connection object.
        _port (int): The port for the Bloomberg API connection.
        _timeout (int): The timeout for the Bloomberg API connection in milliseconds.
    """

    _instance = None

    def __new__(cls):
        """
        Ensures a single instance of BloombergConnection is used (Singleton pattern).
        Initializes the connection if it doesn't exist.
        """
        if cls._instance is None:
            cls._instance = super(BloombergConnection, cls).__new__(cls)
            cls._instance._port = config.general_settings.get('default_bloomberg_port', 8194)
            cls._instance._timeout = config.general_settings.get('default_bloomberg_timeout', 50000)
            cls._instance._con = None
            cls._instance._connect()
        return cls._instance

    def _connect(self) -> None:
        """
        Establishes a connection to the Bloomberg terminal.
        """
        if self._con is None:
            try:
                print(f"Attempting to connect to Bloomberg API on port {self._port}...")
                self._con = pdblp.BCon(debug=False, port=self._port, timeout=self._timeout)
                self._con.start()
                print("Bloomberg API connected successfully.")
            except Exception as e:
                print(f"Error connecting to Bloomberg API: {e}")
                self._con = None  # Ensure _con is None if connection fails

    def _ensure_connection(self) -> None:
        """
        Ensures the Bloomberg connection is active before making a request.
        Reconnects if necessary.
        """
        if self._con is None or not self._con.is_connected:
            self._connect()
        if self._con is None:  # If connection still fails after retry
            raise ConnectionError("Failed to connect to Bloomberg API.")

    def ref(self, tickers: Union[str, List[str]], fields: Union[str, List[str]],
            ovrds: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Fetches reference data (current values) from Bloomberg.

        Args:
            tickers (Union[str, List[str]]): Single ticker string or list of ticker strings.
            fields (Union[str, List[str]]): Single field string or list of field strings.
            ovrds (Optional[List[tuple]]): Optional list of overrides for the request.
                                            Format: [('field', 'value'), ...]

        Returns:
            pd.DataFrame: A DataFrame containing the requested reference data.

        Raises:
            ConnectionError: If Bloomberg API connection cannot be established.
            Exception: For other Bloomberg API errors.
        """
        self._ensure_connection()
        try:
            return self._con.ref(tickers, fields, ovrds)
        except Exception as e:
            raise Exception(f"Bloomberg ref data request failed for {tickers}, {fields}: {e}")

    def bdh(self, tickers: Union[str, List[str]], fields: Union[str, List[str]],
            start_date: str, end_date: str, ovrds: Optional[List[tuple]] = None, **kwargs) -> pd.DataFrame:
        """
        Fetches historical data from Bloomberg.

        Args:
            tickers (Union[str, List[str]]): Single ticker string or list of ticker strings.
            fields (Union[str, List[str]]): Single field string or list of field strings.
            start_date (str): Start date in 'YYYYMMDD' format.
            end_date (str): End date in 'YYYYMMDD' format.
            ovrds (Optional[List[tuple]]): Optional list of overrides for the request.
            **kwargs: Additional keyword arguments for pdblp.bdh (e.g., longdata=True).

        Returns:
            pd.DataFrame: A DataFrame containing the requested historical data.

        Raises:
            ConnectionError: If Bloomberg API connection cannot be established.
            Exception: For other Bloomberg API errors.
        """
        self._ensure_connection()
        try:
            return self._con.bdh(tickers, fields, start_date, end_date, ovrds=ovrds, **kwargs)
        except Exception as e:
            raise Exception(
                f"Bloomberg historical data request failed for {tickers}, {fields} ({start_date} to {end_date}): {e}")

    def bulkref(self, ticker: str, field: str, ovrds: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Fetches bulk reference data from Bloomberg.

        Args:
            ticker (str): The ticker string for the bulk request.
            field (str): The bulk data field (e.g., 'OPT_CHAIN', 'CURVE_TENOR_RATES').
            ovrds (Optional[List[tuple]]): Optional list of overrides for the request.

        Returns:
            pd.DataFrame: A DataFrame containing the requested bulk data.

        Raises:
            ConnectionError: If Bloomberg API connection cannot be established.
            Exception: For other Bloomberg API errors.
        """
        self._ensure_connection()
        try:
            return self._con.bulkref(ticker, field, ovrds)
        except Exception as e:
            raise Exception(f"Bloomberg bulk reference data request failed for {ticker}, {field}: {e}")

    def ref_hist(self, tickers: Union[str, List[str]], fields: Union[str, List[str]], dates: Optional[List[str]] = None,
                 overrides: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Fetches historical reference data with specific dates.

        Args:
            tickers (Union[str, List[str]]): Single ticker string or list of ticker strings.
            fields (Union[str, List[str]]): Single field string or list of field strings.
            dates (Optional[List[str]]): List of dates in 'YYYYMMDD' format for historical reference.
            overrides (Optional[List[tuple]]): Optional list of overrides for the request.

        Returns:
            pd.DataFrame: A DataFrame containing the requested historical reference data.

        Raises:
            ConnectionError: If Bloomberg API connection cannot be established.
            Exception: For other Bloomberg API errors.
        """
        self._ensure_connection()
        try:
            return self._con.ref_hist(tickers, fields, dates=dates, ovrds=overrides)
        except Exception as e:
            raise Exception(
                f"Bloomberg historical reference data request failed for {tickers}, {fields} on dates {dates}: {e}")

    def close(self) -> None:
        """
        Closes the Bloomberg API connection.
        """
        if self._con:
            try:
                self._con.stop()
                print("Bloomberg API connection closed.")
            except Exception as e:
                print(f"Error closing Bloomberg API connection: {e}")
            finally:
                self._con = None  # Ensure _con is None after attempting to stop


class HistoricalDataLoader:
    """
    Manages loading and saving of historical data (e.g., curve nodes, fixings)
    from the specified data lake directory.

    Attributes:
        data_lake_path (Path): The path to the data_lake directory.
    """

    def __init__(self):
        """
        Initializes the HistoricalDataLoader with the path from the global config.
        Ensures the data_lake directory exists.
        """
        self.data_lake_path = Path(config.general_settings.get('data_lake_path', 'data/data_lake'))
        self.data_lake_path.mkdir(parents=True, exist_ok=True)
        print(f"Historical data will be loaded/saved from: {self.data_lake_path}")

    def load_pickle(self, filename: str) -> Any:
        """
        Loads data from a pickle file in the data_lake directory.

        Args:
            filename (str): The name of the pickle file (e.g., 'SOFR_DC_H.pkl').

        Returns:
            Any: The data loaded from the pickle file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: For other errors during loading.
        """
        file_path = self.data_lake_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Historical data file not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Data loaded successfully from {filename}.")
            return data
        except Exception as e:
            raise Exception(f"Error loading data from {filename}: {e}")

    def save_pickle(self, data: Any, filename: str) -> None:
        """
        Saves data to a pickle file in the data_lake directory.

        Args:
            data (Any): The data to be saved.
            filename (str): The name of the pickle file (e.g., 'SOFR_DC_H.pkl').

        Raises:
            Exception: For errors during saving.
        """
        file_path = self.data_lake_path / filename
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Data saved successfully to {filename}.")
        except Exception as e:
            raise Exception(f"Error saving data to {filename}: {e}")

    def load_excel(self, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Loads data from an Excel file in the data_lake directory.

        Args:
            filename (str): The name of the Excel file (e.g., 'eco_master.xlsx').
            sheet_name (Optional[str]): The specific sheet name to read. If None, reads the first sheet.

        Returns:
            pd.DataFrame: The DataFrame loaded from the Excel file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: For other errors during loading.
        """
        file_path = self.data_lake_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            print(f"Data loaded successfully from {filename}.")
            return df
        except Exception as e:
            raise Exception(f"Error loading Excel file {filename}: {e}")


# Instantiate the singleton Bloomberg connection and data loader for easy access
b_con = BloombergConnection()
data_loader = HistoricalDataLoader()