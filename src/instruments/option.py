# src/instruments/option.py

import QuantLib as ql
import datetime
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.instruments.base_instrument import BaseInstrument
from src.curves.ois_curve_builder import OISCurveBuilder  # For STIR options discounting
from src.quantlib_utils import get_ql_date, get_ql_business_day_convention, get_ql_day_counter, \
    get_ql_yield_term_structure_handle, get_ql_black_vol_term_structure_handle  #
from src.date_utils import to_bbg_date_str  #
from src.data_manager import b_con  #
from src.financial_math_utils import bond_fut_yield  # Inferred utility


class Option(BaseInstrument):
    """
    Represents a financial option instrument, serving as a base for bond futures options
    and STIR (Short Term Interest Rate) options.

    Handles common QuantLib Black-Scholes-Merton model setup and pricing for European options.
    """

    def __init__(self, currency_code: str,
                 option_type: str,  # 'C' for Call, 'P' for Put
                 strike_price: float,
                 expiry_date_input: Union[str, datetime.datetime, ql.Date],
                 underlying_ticker: str,  # Bloomberg ticker of the underlying future
                 valuation_date_input: Optional[Union[str, int, datetime.datetime, ql.Date]] = None):
        """
        Initializes the Option instrument.

        Args:
            currency_code (str): The three-letter currency code (e.g., "USD", "GBP").
            option_type (str): 'C' for Call, 'P' for Put.
            strike_price (float): The strike price of the option.
            expiry_date_input (Union[str, datetime.datetime, ql.Date]):
                The expiry date of the option.
            underlying_ticker (str): Bloomberg ticker of the underlying future.
            valuation_date_input (Optional[Union[str, int, datetime.datetime, ql.Date]]):
                The date for which the instrument is valued. If None, uses today's date.
        """
        super().__init__(currency_code, valuation_date_input)

        self.option_type_str = option_type
        self.strike_price = strike_price
        self.expiry_date = get_ql_date(expiry_date_input)
        self.underlying_ticker = underlying_ticker

        self._risk_free_rate: float = None
        self._dividend_rate: float = None  # Typically 0.0 for futures options
        self._volatility: float = None

        self._ql_risk_free_ts_handle: ql.YieldTermStructureHandle = None
        self._ql_dividend_ts_handle: ql.YieldTermStructureHandle = None
        self._ql_vol_ts_handle: ql.BlackVolTermStructureHandle = None
        self._ql_bsm_process: ql.BlackScholesMertonProcess = None

        self._fetch_market_data()  # Fetch underlying price and default rates
        self._setup_ql_instrument()  # Setup QuantLib option and process

    def _fetch_market_data(self) -> None:
        """
        Fetches necessary market data for the option, primarily the underlying future price
        and the risk-free rate.
        """
        ref_date_bbg = to_bbg_date_str(self.valuation_date, ql_date=1)

        try:
            # Fetch underlying future's last price
            underlying_price_df = b_con.ref(self.underlying_ticker, 'PX_LAST')
            self._market_data['underlying_price'] = underlying_price_df['value'].iloc[0]

            # Fetch risk-free rate
            # Uses `default_risk_free_rate_ticker` from config (e.g., 'FEDL01 Index')
            risk_free_rate_ticker = self._convention_config.get('default_risk_free_rate_ticker', 'FEDL01 Index')
            risk_free_rate_df = b_con.ref(risk_free_rate_ticker, 'PX_LAST')
            self._risk_free_rate = risk_free_rate_df['value'].iloc[0] / 100.0  # Convert to decimal
            self._dividend_rate = self._convention_config.get('default_div_rate',
                                                              0.0)  # Futures options usually have 0 dividend

        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch market data for option {self.underlying_ticker} (Strike: {self.strike_price}, Type: {self.option_type_str}): {e}")

    def _setup_ql_instrument(self) -> None:
        """
        Sets up the QuantLib option object and its pricing engine (Black-Scholes-Merton).
        Requires volatility to be set via `set_volatility` for pricing.
        """
        if self._risk_free_rate is None or self._market_data.get('underlying_price') is None:
            raise ValueError(
                "Market data (risk-free rate, underlying price) not loaded. Call _fetch_market_data first.")

        # Define QuantLib option type
        ql_option_type = ql.Option.Call if self.option_type_str == 'C' else ql.Option.Put

        # Payoff and Exercise
        payoff = ql.PlainVanillaPayoff(ql_option_type, self.strike_price)
        exercise = ql.EuropeanExercise(self.expiry_date)
        self._ql_instrument = ql.VanillaOption(payoff, exercise)

        # Default volatility for BSM process initialization (will be updated by set_volatility)
        default_vol = self._convention_config.get('default_option_initial_vol', 0.05)  # Example default
        self._volatility = default_vol  # Set internal volatility attribute

        # Create QuantLib Term Structure Handles
        self._ql_risk_free_ts_handle = get_ql_yield_term_structure_handle(self._risk_free_rate, self.calendar,
                                                                          ql.Actual365Fixed())
        self._ql_dividend_ts_handle = get_ql_yield_term_structure_handle(self._dividend_rate, self.calendar,
                                                                         ql.Actual365Fixed())
        self._ql_vol_ts_handle = get_ql_black_vol_term_structure_handle(self._volatility, self.calendar,
                                                                        ql.Actual365Fixed())

        # Create Black-Scholes-Merton Process
        self._ql_bsm_process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(self._market_data['underlying_price'])),  # Underlying spot handle
            self._ql_dividend_ts_handle,  #
            self._ql_risk_free_ts_handle,  #
            self._ql_vol_ts_handle  #
        )

        # Set initial pricing engine (Analytic European for vanilla options)
        self._ql_engine = ql.AnalyticEuropeanEngine(self._ql_bsm_process)
        self._ql_instrument.setPricingEngine(self._ql_engine)

    def set_underlying_price(self, price: float) -> None:
        """
        Updates the underlying price in the BSM process.

        Args:
            price (float): The new underlying price.
        """
        if self._ql_bsm_process:
            self._ql_bsm_process.stateVariable().setValue(price)
        else:
            raise RuntimeError("BSM process not initialized. Call _setup_ql_instrument first.")

    def set_risk_free_rate(self, rate: float) -> None:
        """
        Updates the risk-free rate in the BSM process.

        Args:
            rate (float): The new risk-free rate (decimal).
        """
        self._risk_free_rate = rate
        if self._ql_bsm_process:
            self._ql_risk_free_ts_handle.linkTo(ql.FlatForward(self.valuation_date, rate, ql.Actual365Fixed()))
        else:
            raise RuntimeError("BSM process not initialized. Call _setup_ql_instrument first.")

    def set_volatility(self, volatility: float) -> None:
        """
        Sets the volatility for the option's pricing engine.

        Args:
            volatility (float): The volatility (as a decimal, e.g., 0.15 for 15%).
        """
        self._volatility = volatility
        if self._ql_bsm_process:
            self._ql_vol_ts_handle.linkTo(
                ql.BlackConstantVol(self.valuation_date, self.calendar, volatility, ql.Actual365Fixed()))
        else:
            raise RuntimeError("BSM process not initialized. Call _setup_ql_instrument first.")

    def calculate_npv(self) -> float:
        """
        Calculates the Net Present Value (NPV) of the option.

        Returns:
            float: The NPV of the option.
        """
        if self._ql_instrument is None or self._volatility is None:
            raise RuntimeError(
                "Option instrument or volatility not set up. Call _setup_ql_instrument and set_volatility() first.")

        # Ensure engine is updated with current BSM process if volatility was changed
        self._ql_instrument.setPricingEngine(ql.AnalyticEuropeanEngine(self._ql_bsm_process))
        return self._ql_instrument.NPV()

    def calculate_implied_volatility(self, market_price: float, guess: Optional[float] = None) -> float:
        """
        Calculates the implied volatility of the option given its market price.

        Args:
            market_price (float): The market price of the option.
            guess (Optional[float]): An optional initial guess for the implied volatility.

        Returns:
            float: The implied volatility in percentage (e.g., 15.0 for 15%).
        """
        if self._ql_instrument is None:
            raise RuntimeError("Option instrument not set up. Call _setup_ql_instrument first.")

        # Ensure the BSM process has the *current* underlying price and rates
        # (volatility is what we are solving for)
        current_bsm_process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(self._market_data['underlying_price'])),  #
            self._ql_dividend_ts_handle,  #
            self._ql_risk_free_ts_handle,  #
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.valuation_date, self.calendar, 0.01, ql.Actual365Fixed()))
            # Dummy vol for process
        )

        try:
            implied_vol = self._ql_instrument.impliedVolatility(
                market_price,  #
                current_bsm_process,  #
                guess=guess if guess else self._volatility if self._volatility else 0.5,  # Initial guess for solver
                maxVol=self._convention_config.get('default_option_max_vol', 100.0)  # Max vol from config
            ) * 100.0  # Convert to percentage
            return implied_vol
        except Exception as e:
            print(
                f"Warning: Could not calculate implied volatility for option (Strike: {self.strike_price}, Type: {self.option_type_str}, Market Price: {market_price}). Error: {e}")
            return np.nan

    def calculate_delta(self) -> float:
        """
        Calculates the option's delta (sensitivity to underlying price).

        Returns:
            float: The option's delta.
        """
        if self._ql_instrument is None or self._volatility is None:
            raise RuntimeError("Option instrument or volatility not set up.")
        self._ql_instrument.setPricingEngine(ql.AnalyticEuropeanEngine(self._ql_bsm_process))
        return self._ql_instrument.delta() * 100.0  # Often expressed per 100

    def calculate_gamma(self) -> float:
        """
        Calculates the option's gamma (sensitivity of delta to underlying price).

        Returns:
            float: The option's gamma.
        """
        if self._ql_instrument is None or self._volatility is None:
            raise RuntimeError("Option instrument or volatility not set up.")
        self._ql_instrument.setPricingEngine(ql.AnalyticEuropeanEngine(self._ql_bsm_process))
        return self._ql_instrument.gamma()

    def calculate_theta(self) -> float:
        """
        Calculates the option's theta (sensitivity to time decay).

        Returns:
            float: The option's theta.
        """
        if self._ql_instrument is None or self._volatility is None:
            raise RuntimeError("Option instrument or volatility not set up.")
        self._ql_instrument.setPricingEngine(ql.AnalyticEuropeanEngine(self._ql_bsm_process))
        return self._ql_instrument.theta()

    def calculate_vega(self) -> float:
        """
        Calculates the option's vega (sensitivity to volatility).

        Returns:
            float: The option's vega.
        """
        if self._ql_instrument is None or self._volatility is None:
            raise RuntimeError("Option instrument or volatility not set up.")
        self._ql_instrument.setPricingEngine(ql.AnalyticEuropeanEngine(self._ql_bsm_process))
        return self._ql_instrument.vega()

    @property
    def underlying_price(self) -> float:
        """Returns the current underlying price."""
        return self._market_data.get('underlying_price')

    @property
    def risk_free_rate(self) -> float:
        """Returns the risk-free rate used."""
        return self._risk_free_rate

    @property
    def dividend_rate(self) -> float:
        """Returns the dividend rate used."""
        return self._dividend_rate

    @property
    def volatility(self) -> Optional[float]:
        """Returns the current volatility used for pricing."""
        return self._volatility

    @property
    def ql_option_type(self) -> ql.Option.Type:
        """Returns the QuantLib option type enum."""
        return ql.Option.Call if self.option_type_str == 'C' else ql.Option.Put