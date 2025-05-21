# src/date_utils.py

import datetime
import QuantLib as ql
from typing import Union, List, Optional
import re


def datetime_to_ql(dt_obj: datetime.datetime) -> ql.Date:
    """
    Converts a Python datetime object to a QuantLib Date object.

    Args:
        dt_obj (datetime.datetime): The datetime object to convert.

    Returns:
        ql.Date: The corresponding QuantLib Date object.
    """
    return ql.Date(dt_obj.day, dt_obj.month, dt_obj.year)


def ql_to_datetime(ql_date: ql.Date) -> datetime.datetime:
    """
    Converts a QuantLib Date object to a Python datetime object.

    Args:
        ql_date (ql.Date): The QuantLib Date object to convert.

    Returns:
        datetime.datetime: The corresponding datetime object.
    """
    return datetime.datetime(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())


def to_bbg_date_str(date_obj: Union[datetime.datetime, ql.Date], ql_date: int = 0) -> str:
    """
    Converts a date object (datetime or QuantLib Date) to a Bloomberg-compatible
    date string (YYYYMMDD format).

    Args:
        date_obj (Union[datetime.datetime, ql.Date]): The date object to convert.
        ql_date (int): If 1, indicates date_obj is a QuantLib Date. Otherwise, assumes datetime.

    Returns:
        str: The date string in 'YYYYMMDD' format.
    """
    if ql_date == 1:
        # Assuming date_obj is a QuantLib Date object
        day = str(date_obj.dayOfMonth()).zfill(2)
        month = str(date_obj.month()).zfill(2)
        year = str(date_obj.year())
    else:
        # Assuming date_obj is a datetime object
        day = str(date_obj.day).zfill(2)
        month = str(date_obj.month).zfill(2)
        year = str(date_obj.year)
    return f"{year}{month}{day}"


def get_next_imm(start_date: ql.Date, num_quarters: int) -> List[ql.Date]:
    """
    Calculates the next series of IMM (International Monetary Market) future dates.

    IMM dates are the second Wednesday of March, June, September, and December.

    Args:
        start_date (ql.Date): The starting QuantLib Date.
        num_quarters (int): The number of IMM dates to generate.

    Returns:
        List[ql.Date]: A list of QuantLib Date objects representing the IMM dates.
    """
    imm_dates = []
    current_date = start_date

    while len(imm_dates) < num_quarters:
        # Find the next IMM month (March, June, Sept, Dec)
        year = current_date.year()
        month = current_date.month()

        if month < 3:
            imm_month = 3  # March
        elif month < 6:
            imm_month = 6  # June
        elif month < 9:
            imm_month = 9  # September
        elif month < 12:
            imm_month = 12  # December
        else:  # month is Dec, move to next year March
            imm_month = 3
            year += 1

        # Adjust if current_date is past the IMM date of the current quarter
        if datetime.datetime(year, imm_month, 1) < ql_to_datetime(current_date):
            # Move to the next IMM quarter
            if imm_month == 12:
                imm_month = 3
                year += 1
            else:
                imm_month += 3

        # Find the second Wednesday of that month
        # Start from the 8th (earliest possible 2nd Wednesday)
        temp_date = ql.Date(8, imm_month, year)
        while temp_date.weekday() != ql.Wednesday:
            temp_date += 1  # Move to the next day

        # If the generated date is before or equal to the start_date, advance by a quarter
        # This handles cases where start_date is on or slightly after an IMM date for the same quarter
        if temp_date <= start_date:
            current_date = temp_date + ql.Period(3, ql.Months)  # Advance to the next quarter's start for next iteration
            continue  # Skip adding this date and re-evaluate for the next quarter

        imm_dates.append(temp_date)
        current_date = temp_date + ql.Period(1, ql.Days)  # Move just past the current IMM date to search for the next

    return imm_dates