# src/quantlib_utils.py

import QuantLib as ql
import datetime
from typing import Union, Optional
from src.date_utils import datetime_to_ql, ql_to_datetime
from src.config import config


def get_ql_date(date_input: Union[str, int, datetime.datetime, ql.Date]) -> ql.Date:
    """
    Converts various date input formats into a QuantLib Date object.

    Args:
        date_input (Union[str, int, datetime.datetime, ql.Date]):
            - str: 'YYYY-MM-DD' or 'DD-MM-YYYY' format.
            - int: Number of days offset from today (current QuantLib evaluation date).
            - datetime.datetime: Python datetime object.
            - ql.Date: QuantLib Date object (returned as is).

    Returns:
        ql.Date: The corresponding QuantLib Date object.

    Raises:
        ValueError: If the date_input format is unrecognized or invalid.
    """
    if isinstance(date_input, ql.Date):
        return date_input
    elif isinstance(date_input, datetime.datetime):
        return datetime_to_ql(date_input)
    elif isinstance(date_input, str):
        try:
            # Attempt YYYY-MM-DD format
            dt_obj = datetime.datetime.strptime(date_input, '%Y-%m-%d')
            return datetime_to_ql(dt_obj)
        except ValueError:
            try:
                # Attempt DD-MM-YYYY format
                dt_obj = datetime.datetime.strptime(date_input, '%d-%m-%Y')
                return datetime_to_ql(dt_obj)
            except ValueError:
                raise ValueError(
                    f"Unrecognized date string format: {date_input}. Expected 'YYYY-MM-DD' or 'DD-MM-YYYY'.")
    elif isinstance(date_input, int):
        # Interpret as days offset from current QL evaluation date
        eval_date = ql.Settings.instance().evaluationDate
        return eval_date + ql.Period(date_input, ql.Days)
    else:
        raise ValueError(f"Unsupported date input type: {type(date_input)}.")


def setup_ql_settings(evaluation_date: Optional[Union[str, datetime.datetime, ql.Date]] = None) -> None:
    """
    Sets the global QuantLib evaluation date. If None, uses today's date.

    Args:
        evaluation_date (Optional[Union[str, datetime.datetime, ql.Date]]):
            The date to set as the evaluation date. Can be string, datetime, or ql.Date.
            If None, current system date is used.
    """
    if evaluation_date is None:
        today = datetime.datetime.now()
        ql_eval_date = datetime_to_ql(today)
    else:
        ql_eval_date = get_ql_date(evaluation_date)

    ql.Settings.instance().evaluationDate = ql_eval_date
    print(
        f"QuantLib evaluation date set to: {ql_to_datetime(ql.Settings.instance().evaluationDate).strftime('%Y-%m-%d')}")


def get_ql_yield_term_structure_handle(rate: float, calendar: ql.Calendar,
                                       day_count: ql.DayCounter) -> ql.YieldTermStructureHandle:
    """
    Creates a QuantLib YieldTermStructureHandle from a flat rate.

    Args:
        rate (float): The constant rate (e.g., risk-free rate, dividend yield) as a decimal.
        calendar (ql.Calendar): The QuantLib calendar.
        day_count (ql.DayCounter): The QuantLib day count convention.

    Returns:
        ql.YieldTermStructureHandle: A handle to a flat forward yield term structure.
    """
    calc_date = ql.Settings.instance().evaluationDate
    return ql.YieldTermStructureHandle(ql.FlatForward(calc_date, rate, day_count))


def get_ql_black_vol_term_structure_handle(volatility: float, calendar: ql.Calendar,
                                           day_count: ql.DayCounter) -> ql.BlackVolTermStructureHandle:
    """
    Creates a QuantLib BlackVolTermStructureHandle from a constant volatility.

    Args:
        volatility (float): The constant volatility as a decimal.
        calendar (ql.Calendar): The QuantLib calendar.
        day_count (ql.DayCounter): The QuantLib day count convention.

    Returns:
        ql.BlackVolTermStructureHandle: A handle to a black constant volatility term structure.
    """
    calc_date = ql.Settings.instance().evaluationDate
    return ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calc_date, calendar, volatility, day_count))


def get_ql_business_day_convention(convention_str: str) -> ql.BusinessDayConvention:
    """
    Converts a string representation of a business day convention to QuantLib's enum.

    Args:
        convention_str (str): String representing the convention (e.g., 'Following', 'ModifiedFollowing').

    Returns:
        ql.BusinessDayConvention: The corresponding QuantLib enum.

    Raises:
        ValueError: If the convention_str is not recognized.
    """
    mapping = {
        'Following': ql.Following,
        'ModifiedFollowing': ql.ModifiedFollowing,
        'Preceding': ql.Preceding,
        'ModifiedPreceding': ql.ModifiedPreceding,
        'Unadjusted': ql.Unadjusted
    }
    conv = mapping.get(convention_str)
    if conv is None:
        raise ValueError(f"Unrecognized business day convention: {convention_str}")
    return conv


def get_ql_day_counter(day_count_str: str) -> ql.DayCounter:
    """
    Converts a string representation of a day count convention to QuantLib's enum.

    Args:
        day_count_str (str): String representing the day count (e.g., 'Actual360', 'Actual365Fixed').

    Returns:
        ql.DayCounter: The corresponding QuantLib enum.

    Raises:
        ValueError: If the day_count_str is not recognized.
    """
    mapping = {
        'Actual360': ql.Actual360(),
        'Actual365Fixed': ql.Actual365Fixed(),
        'Thirty360': ql.Thirty360(),
        'ActualActual': ql.ActualActual(),  # Often used with ISDA or AFB basis
        'ActualActualISDA': ql.ActualActual(ql.ActualActual.ISDA)
    }
    dc = mapping.get(day_count_str)
    if dc is None:
        raise ValueError(f"Unrecognized day count convention: {day_count_str}")
    return dc


def get_ql_frequency(frequency_str: str) -> ql.Frequency:
    """
    Converts a string representation of a frequency to QuantLib's enum.

    Args:
        frequency_str (str): String representing the frequency (e.g., 'Annual', 'Quarterly', 'Semiannual', 'Monthly').

    Returns:
        ql.Frequency: The corresponding QuantLib enum.

    Raises:
        ValueError: If the frequency_str is not recognized.
    """
    mapping = {
        'Once': ql.Once,
        'NoFrequency': ql.NoFrequency,
        'Daily': ql.Daily,
        'Weekly': ql.Weekly,
        'BiWeekly': ql.BiWeekly,
        'Monthly': ql.Monthly,
        'Quarterly': ql.Quarterly,
        'Semiannual': ql.Semiannual,
        'Annual': ql.Annual
    }
    freq = mapping.get(frequency_str)
    if freq is None:
        raise ValueError(f"Unrecognized frequency: {frequency_str}")
    return freq


def get_ql_calendar(calendar_str: str) -> ql.Calendar:
    """
    Converts a string representation of a calendar to QuantLib's object.

    Args:
        calendar_str (str): String representing the calendar (e.g., 'UnitedStates', 'TARGET', 'UnitedKingdom').

    Returns:
        ql.Calendar: The corresponding QuantLib Calendar object.

    Raises:
        ValueError: If the calendar_str is not recognized.
    """
    try:
        # Dynamically get QuantLib calendar object
        calendar = getattr(ql, calendar_str)()
        return calendar
    except AttributeError:
        raise ValueError(f"Unrecognized QuantLib calendar: {calendar_str}")


def get_ql_currency(currency_str: str) -> ql.Currency:
    """
    Converts a string representation of a currency to QuantLib's object.

    Args:
        currency_str (str): String representing the currency (e.g., 'USDCurrency', 'GBPCurrency', 'EURCurrency').

    Returns:
        ql.Currency: The corresponding QuantLib Currency object.

    Raises:
        ValueError: If the currency_str is not recognized.
    """
    try:
        # Dynamically get QuantLib currency object
        currency = getattr(ql, currency_str)()
        return currency
    except AttributeError:
        raise ValueError(f"Unrecognized QuantLib currency: {currency_str}")