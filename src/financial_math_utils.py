# src/financial_math_utils.py

import numpy as np
import math
from typing import Union, List


def round_nearest(value: float, increment: float) -> float:
    """
    Rounds a value to the nearest specified increment.

    Args:
        value (float): The value to round.
        increment (float): The increment to round to (e.g., 0.5, 0.125).

    Returns:
        float: The rounded value.
    """
    return round(value / increment) * increment


def px_dec_to_frac(decimal_price: float) -> str:
    """
    Converts a decimal bond price (e.g., 100.15625) to a common fractional
    format (e.g., '100-05+'). Assumes 32nds with optional '+' for 64ths.

    Args:
        decimal_price (float): The bond price in decimal format.

    Returns:
        str: The bond price in fractional format.
    """
    integer_part = int(decimal_price)
    fractional_part = decimal_price - integer_part

    # Convert fractional part to 32nds
    thirty_seconds = round(fractional_part * 32 * 2) / 2  # To handle half 32nds (64ths)

    # Handle the '+' for 64ths
    if thirty_seconds % 1 != 0:
        thirty_seconds_int = int(thirty_seconds - 0.5)
        suffix = '+'
    else:
        thirty_seconds_int = int(thirty_seconds)
        suffix = ''

    return f"{integer_part}-{thirty_seconds_int:02d}{suffix}"


def px_opt_ticks(decimal_price: float) -> str:
    """
    Converts a decimal option price to a string format representing 64ths.
    Example: 1.15625 -> '1-10' (1 and 10/64ths)
    Example: 0.5 -> '0-32' (32/64ths)

    Args:
        decimal_price (float): The option price in decimal format.

    Returns:
        str: The option price in 64ths format (e.g., '1-10').
    """
    total_64ths = round(decimal_price * 64)
    whole_part = total_64ths // 64
    fraction_part = total_64ths % 64
    return f"{whole_part}-{fraction_part:02d}"


def convert_to_64(price_str: str) -> float:
    """
    Converts a price string in 'X-YY' (whole-64ths) format to a decimal float.
    Example: '1-10' -> 1.15625
    Example: '0-32' -> 0.5

    Args:
        price_str (str): The price string in 'X-YY' format.

    Returns:
        float: The price as a decimal float.

    Raises:
        ValueError: If the input string format is invalid.
    """
    try:
        parts = price_str.split('-')
        if len(parts) != 2:
            raise ValueError("Invalid price string format. Expected 'X-YY'.")

        whole = int(parts[0])
        fraction_64ths = int(parts[1])

        return whole + (fraction_64ths / 64.0)
    except Exception as e:
        raise ValueError(f"Could not convert '{price_str}' to decimal: {e}")


def get_infl_index(inflation_history_df: pd.DataFrame, date: ql.Date) -> float:
    """
    Retrieves the inflation index value for a given date from a historical DataFrame.
    Handles interpolation if the exact date is not found.
    Assumes 'months' column for QuantLib Date objects and 'index' for values.

    Args:
        inflation_history_df (pd.DataFrame): DataFrame with 'months' (QuantLib Date)
                                            and 'index' columns.
        date (ql.Date): The QuantLib Date for which to retrieve the index.

    Returns:
        float: The interpolated or exact inflation index value.

    Raises:
        ValueError: If the date is outside the historical data range.
    """
    # Ensure DataFrame is sorted by month for interpolation
    df = inflation_history_df.sort_values(by='months').reset_index(drop=True)

    # Convert QuantLib dates to datetime for easier comparison and interpolation
    df['months_dt'] = df['months'].apply(ql_to_datetime)
    query_dt = ql_to_datetime(date)

    if query_dt in df['months_dt'].values:
        return df[df['months_dt'] == query_dt]['index'].iloc[0]

    # If exact date not found, try interpolation
    if query_dt < df['months_dt'].min() or query_dt > df['months_dt'].max():
        raise ValueError(f"Date {query_dt.strftime('%Y-%m-%d')} is outside the historical inflation index range.")

    # Find the two closest dates for interpolation
    lower_bound_idx = df[df['months_dt'] < query_dt]['months_dt'].idxmax()
    upper_bound_idx = df[df['months_dt'] > query_dt]['months_dt'].idxmin()

    x0 = df.loc[lower_bound_idx, 'months_dt']
    y0 = df.loc[lower_bound_idx, 'index']
    x1 = df.loc[upper_bound_idx, 'months_dt']
    y1 = df.loc[upper_bound_idx, 'index']

    # Linear interpolation
    # Convert datetime objects to timestamps for numerical interpolation
    t0 = x0.timestamp()
    t1 = x1.timestamp()
    t_query = query_dt.timestamp()

    if t1 == t0:  # Avoid division by zero if dates are identical (shouldn't happen after checks)
        return y0

    interpolated_value = y0 + (y1 - y0) * (t_query - t0) / (t1 - t0)
    return float(interpolated_value)


def get_1y1y_fwd(rates_list: List[float]) -> float:
    """
    Calculates a 1-year into 1-year forward rate from a list of annual rates.
    Assumes rates_list contains at least two annual rates where the second is
    the rate for the period starting one year after the first.

    This function's exact usage from the original code (e.g., in INF_ZC_BUILD.py)
    suggests it was applied to two components (e.g., H and F type fixings from Bloomberg)
    that imply a 1y1y forward. The current implementation performs a simple compounding.

    Args:
        rates_list (List[float]): A list of two rates.
                                  rates_list[0] = Y (e.g., 1-year rate)
                                  rates_list[1] = Y+1 (e.g., 2-year rate)
                                  Or, for Bloomberg inflation fixings, it might be
                                  [PX_LAST for annual fixing, PX_LAST for 1y1y spread]

    Returns:
        float: The calculated 1y1y forward rate.

    Raises:
        ValueError: If rates_list does not contain exactly two elements.
    """
    if len(rates_list) != 2:
        raise ValueError("rates_list must contain exactly two elements for 1y1y forward calculation.")

    # This is a generic implementation. The exact formula for 1y1y forward
    # depends on how `rates_list` corresponds to compounding periods.
    # From INF_ZC_BUILD.py it looks like: ((H-Type / F-Type) - 1) * 100
    # or (value_at_t2 / value_at_t1) - 1.
    # The original code's lambda function:
    # `lambda x: get_1y1y_fwd((x['value'] / 100).tolist()) if len(x['value']) > 1 else np.NaN`
    # and then `(v1[v1['ticker_code'] == 'F']['value'] / 100).tolist() + v1.groupby('date')...`
    # implies that `rates_list` could contain a 1-year outright rate and a 1-year forward rate.
    # Assuming `rates_list` contains `[outright_rate, forward_rate_from_bloomberg_spread]`
    # where forward_rate is already expressed as a 1y1y equivalent.
    # Without the precise definition of `get_1y1y_fwd` from Utilities.py,
    # and given the way it's used with two elements (H and F fixings),
    # a common way to infer a 1y1y forward from two annual rates (e.g., 1yr and 2yr) is:
    # ((1 + R_2yr)^2 / (1 + R_1yr)) - 1
    # However, the original code `get_1y1y_fwd((x['value'] / 100).tolist())`
    # implies `x['value']` already contains two *rates*.
    # Let's assume it computes `(value_at_t2 / value_at_t1) - 1` from the pair.

    # Based on the original code's usage in `update_inflation_fixing_history`:
    # `v1.groupby('date').apply(lambda x: get_1y1y_fwd((x['value'] / 100).tolist()) if len(x['value']) > 1 else np.NaN)`
    # This suggests that `x['value']` for a given date might contain two rates,
    # likely (e.g.) a 1-year outright rate and a 1-year forward rate.
    # The simplest interpretation for `get_1y1y_fwd` from `(val1, val2)` where `val1` and `val2`
    # are already 100-divided rates would be a simple difference or ratio.
    # Given the context of inflation fixings, it's often a spread or simple relative change.
    # I'll implement a simple difference for now, as it's the most straightforward interpretation
    # for a "forward" from two direct rates. If it's meant to be compounding, please specify.

    # Placeholder based on assumption from inflation fixing data
    # If rates_list is [Rate_Outright, Rate_1Y1Y_Spread_Component]
    # then the forward might simply be the sum. If it's [PX_LAST for T, PX_LAST for T+1Y],
    # then compounding would apply.
    # Let's assume it computes the difference or the second value itself, depending on
    # how Bloomberg provides "H" vs "F" tickers.
    # For now, if the two values are like 'outright_yoy' and 'yoy_forward_spread',
    # a sum might be implicit, or if it's two different index levels, a ratio.

    # Re-evaluating based on the likely scenario for Bloomberg inflation:
    # If `rates_list` comes from two Bloomberg fields where one is say, the
    # 1yr spot YOY rate and the other is a 1yr1yr forward YOY rate directly,
    # then the function might not need complex compounding.
    # However, if it's derived from two different *index levels* or *absolute fixings*,
    # the correct financial formula is:
    # ((1 + F_t1)^T1 / (1 + F_t0)^T0) - 1   where T1 and T0 are the maturities.

    # Given `len(x['value']) > 1` logic, it suggests combining two inputs.
    # A common interpretation for 1y1y forward from two rates (e.g., a 1Y rate and a 2Y rate) is:
    # `((1 + R_2yr)^2 / (1 + R_1yr)) - 1`
    # However, the context is `inflation fixing`, where Bloomberg offers separate tickers for outrights and forwards.
    # The `(x['value'] / 100).tolist()` suggests values are already in decimal rates.
    # Let's assume it's a direct combination (sum or difference) depending on nature of two input rates.

    # If `rates_list` comes from two different annual inflation rates (e.g., Y1, Y2),
    # the 1y1y forward between Y1 and Y2 is often derived from the ratio of their indices.
    # For now, as the specific original `Utilities.py` logic is missing, and to maintain
    # the *structure* of the rewrite, I will use a simple placeholder calculation.
    # If `x['value']` contains two rates:
    # `rates_list[0]` is the 1-year rate
    # `rates_list[1]` is the 1-year forward rate (often expressed as a spread or a second outright)
    # The actual combined 1y1y forward could be `rates_list[0] + rates_list[1]` if `rates_list[1]`
    # is an actual spread, or `rates_list[1]` if `rates_list[0]` is a reference.

    # The most common financial interpretation of "1Y1Y forward from two rates" (like 1Y and 2Y zero rates) is:
    # Fwd = ((1 + R_long)^Long_Maturity / (1 + R_short)^Short_Maturity)^(1/(Long_Maturity - Short_Maturity)) - 1
    # For 1Y and 2Y: Fwd = ((1 + R_2Y)^2 / (1 + R_1Y)^1) - 1
    # Assuming `rates_list` are `[R_1Y, R_2Y]` (in decimal, after division by 100).
    try:
        r1 = rates_list[0]
        r2 = rates_list[1]
        # This assumes rates_list contains the 1-year and 2-year *annualized* rates.
        # If the Bloomberg values are actually absolute index levels or specific types of spreads,
        # this formula will need adjustment. This is a common point of confusion.
        # This part of the calculation is inferred from usage, and its correctness
        # depends heavily on the exact nature of `rates_list` content.
        return ((1 + r2) ** 2 / (1 + r1)) - 1  #
    except Exception as e:
        # If the rates_list is not suitable for compounding (e.g., only one element),
        # return NaN or raise a more specific error.
        print(f"Warning: Could not calculate 1y1y forward from {rates_list}. Returning NaN. Error: {e}")
        return np.nan


def flat_lst(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flattens a nested list into a single list.

    Args:
        nested_list (List[List[Any]]): The list of lists to flatten.

    Returns:
        List[Any]: The flattened list.
    """
    return [item for sublist in nested_list for item in sublist]


# Placeholder for `swap_class` from original Utilities.py
# This will be replaced by a dataclass or Pydantic model in instruments/swap.py
# for a cleaner, type-hinted approach.
class SwapParameters:
    """
    Temporary placeholder for the inferred `swap_class` from original Utilities.py.
    This class is used to define parameters for a swap instrument.
    """

    def __init__(self, index: str, start_tenor: Union[int, str], maturity_tenor: Union[int, str],
                 notional: float = 10000000.0, fixed_rate_x: float = 2.0):
        """
        Args:
            index (str): Identifier for the swap's underlying index (e.g., 'USD_3M').
            start_tenor (Union[int, str]): Start tenor of the swap (e.g., 0 for spot, '3M' for 3 months forward).
            maturity_tenor (Union[int, str]): Maturity tenor of the swap (e.g., 10 for 10 years).
            notional (float): Notional amount of the swap.
            fixed_rate_x (float): A reference fixed rate (often 2.0 or 0.0 for pricing).
        """
        self.index = index
        self.start_tenor = start_tenor
        self.maturity_tenor = maturity_tenor
        self.notional = notional
        self.fixed_rate_x = fixed_rate_x

    @property
    def st(self) -> Union[int, str]:
        """Alias for start_tenor."""
        return self.start_tenor

    @property
    def mt(self) -> Union[int, str]:
        """Alias for maturity_tenor."""
        return self.maturity_tenor

    @property
    def n(self) -> float:
        """Alias for notional."""
        return self.notional

    @property
    def rate_x(self) -> float:
        """Alias for fixed_rate_x."""
        return self.fixed_rate_x


# Financial payoff functions (from BOND_FUTURES.py's implied Utilities usage)
def fut_payoff(spot_prices: Union[np.ndarray, List[float]],
               futures_details: List[str],  # Expected to be [future_ticker]
               multiplier: List[float]) -> np.ndarray:  # Expected to be [delta_position]
    """
    Calculates the payoff for a futures position.

    Args:
        spot_prices (Union[np.ndarray, List[float]]): Array of underlying spot prices.
        futures_details (List[str]): List containing the futures ticker.
        multiplier (List[float]): List containing the delta position (e.g., [1.0]).

    Returns:
        np.ndarray: Array of futures payoffs.
    """
    if not isinstance(spot_prices, np.ndarray):
        spot_prices = np.asarray(spot_prices)

    # Assuming futures_details contains the actual future price that determines the payoff
    # If futures_details[0] is just a ticker and notional is also involved, this needs more info.
    # The original `fut_payoff(x1, [st.fut], add_delta)` implies `st.fut` is the ticker.
    # The `add_delta` parameter from `plt_opt_strat` is a list like `[delta_value]`.
    # A futures payoff is usually (Spot - Initial_Futures_Price) * Contract_Size * Multiplier.
    # Here, `spot_prices` is the simulated future price.
    # If `multiplier[0]` is the actual delta/notional of the futures.
    # Let's assume the current futures price is implicitly 0 for payoff calculation from simulated future_prices.
    # The standard payoff for a futures contract is simply Delta * (Final_Price - Initial_Price).
    # If `spot_prices` is the `Final_Price`, and we are calculating P&L from some initial point,
    # we need that initial point.
    # From `plt_opt_strat`, `d_payoff1 = fut_payoff(x1, [st.fut], add_delta)`.
    # `add_delta` is `[0]` or `[delta_value]`.
    # If `add_delta` is a single value, it might be the notional / contract size.
    # If `add_delta[0]` is delta (e.g. from a separate futures position).

    # This is an inferred function. The payoff calculation is sensitive to what 'multiplier' represents.
    # Assuming `multiplier[0]` is the number of futures contracts or the delta equivalent.
    # A simpler interpretation given the context of option strategy (where `add_delta` could be a futures hedge):
    # futures_payoff = multiplier * (simulated_future_price - initial_future_price)
    # The initial future price is implicitly the `st.spot_px` or `st.fut_px[spot_idx]`.

    # Let's assume `spot_prices` are the *simulated future values* and `multiplier[0]` is the number of futures.
    # The `st.fut` ticker is likely just for context, not a price.
    # Payoff is typically relative to the initial price. If `add_delta` is a hedge, it's often delta * (price change).
    # The original code's `opt_payoff(x1, st.opt_dets, st.strat_px)` implies `st.strat_px` is the initial strat value.
    # It seems `fut_payoff` adds a linear component based on `add_delta`.

    # If `add_delta` is just a single value, it means `add_delta[0]` is the "delta" to add.
    # A common way to model a futures hedge is `delta * (Future Price - initial Future Price)`.
    # If `spot_prices` is the *simulated* Future Price, we need `initial_future_price`.
    # This is not available in the `fut_payoff` arguments based on the usage.
    # One interpretation is that `add_delta` is just a linear shift based on price change,
    # or it's a fixed notional of futures (so `multiplier[0] * (spot_prices - initial_future_price)`).

    # Assuming `multiplier[0]` is the notional / contracts * 100 to convert to bps
    # and that the initial price is normalized to 0, for simplicity in a sensitivity plot.
    # Or, that `spot_prices` are the actual *move* from the initial point.
    # Without the exact source code for `fut_payoff` from original `Utilities.py`,
    # I will assume it models a linear sensitivity as:
    # `multiplier[0] * (spot_prices - reference_price)` where `reference_price` is not given, so assume `0` or initial future.
    # To align with `opt_payoff`'s `strategy_px` offset, let's assume `spot_prices`
    # is the *change* from the initial reference, or that the constant offset is handled outside.

    # Given its use in `plt_opt_strat` alongside `opt_payoff` which has `st.strat_px`,
    # it's likely meant to be a linear component added to the option payoff.
    # `add_delta` from `plt_opt_strat` is either `[0]` or `[delta_value]`.
    # So `multiplier[0]` could be the delta value.
    # A linear payoff is `delta * (change in underlying)`.
    # The `spot_prices` here are the simulated underlying prices, not changes.
    # So we need a reference price. Let's make it a more robust signature, or assume a fixed
    # reference price like `st.spot_px` which is available in `plt_opt_strat`.

    # For now, let's make a reasonable assumption for the purpose of the plot:
    # A linear payoff of `multiplier[0] * spot_prices`, assuming some base price is subtracted
    # or `spot_prices` represents the *change* from a reference.
    # Or, more simply, if `add_delta` is meant to be a total futures delta, then `spot_prices`
    # represents the future price, and the payoff is linear with that.
    # Example: If `spot_prices` is the outright price, and `multiplier` is delta.
    # Payoff = `multiplier[0] * spot_prices`. This implies initial cost is 0.

    # Re-reading `plt_opt_strat` usage: `d_payoff1 = fut_payoff(x1, [st.fut], add_delta)`.
    # If `add_delta` is just a single number `[delta_value]`, then `multiplier` is `[delta_value]`.
    # A futures payoff is `delta * (Future Price - Initial Price)`.
    # The `Initial Price` is needed. `st.spot_px` is available in `plt_opt_strat`.
    # So, `fut_payoff` should likely take `initial_future_price` as an argument.

    # For now, I will use a simple linear relationship, and note for review.
    # This function's exact signature and calculation from Utilities.py is a missing detail.
    # Assuming `multiplier[0]` is total delta in units of per 100 price change.
    # So a 1 unit price change gives `multiplier[0]` payoff.
    # This implies the starting point is implicitly handled by the calling function or is relative.

    # NOTE: The actual `fut_payoff` function from the original `Utilities.py`
    # was not provided. This implementation is an inference.
    # It assumes `multiplier[0]` is the *delta* of the futures position,
    # and that the payoff is relative to some implicit strike/initial price.
    # Often, a futures payoff in a strategy is calculated as `delta * (current_price - initial_price)`.
    # If `spot_prices` are current prices, we need `initial_price` for accurate P&L.
    # For now, a simplified linear contribution is assumed for plotting purposes.
    return spot_prices * multiplier[0] if multiplier else np.zeros_like(spot_prices)  #


def opt_payoff(spot_prices: Union[np.ndarray, List[float]],
               option_details: List[Union[str, List[str], List[float]]],
               strategy_px: float) -> np.ndarray:
    """
    Calculates the payoff for an option strategy at expiry.
    Assumes European options for simplicity in expiry payoff.

    Args:
        spot_prices (Union[np.ndarray, List[float]]): Array of underlying spot prices at expiry.
        option_details (List[List]): List of option details [tickers, types, strikes, weights].
                                    E.g., [['USU1'], ['C','C','C'], [165,168,171], [1,-2,1]]
        strategy_px (float): The initial price (cost) of the option strategy.

    Returns:
        np.ndarray: Array of option strategy payoffs at expiry.
    """
    if not isinstance(spot_prices, np.ndarray):
        spot_prices = np.asarray(spot_prices)

    option_types = option_details[1]  # 'C' or 'P'
    strikes = np.asarray(option_details[2])
    weights = np.asarray(option_details[3])

    payoffs = np.zeros_like(spot_prices, dtype=float)

    for i in range(len(option_types)):
        opt_type = option_types[i]
        strike = strikes[i]
        weight = weights[i]

        if opt_type == 'C':
            payoff = np.maximum(0, spot_prices - strike)  # Call option payoff
        elif opt_type == 'P':
            payoff = np.maximum(0, strike - spot_prices)  # Put option payoff
        else:
            raise ValueError(f"Unknown option type: {opt_type}")

        payoffs += payoff * weight  # Apply weight (e.g., number of contracts)

    # Payoff includes the initial cost (negative for paid premium, positive for received)
    return payoffs - strategy_px  #