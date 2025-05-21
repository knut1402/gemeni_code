# src/plotting/matplotlib_plots.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import QuantLib as ql
import datetime
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from src.plotting.base_plotter import BasePlotter
from src.curves.ois_curve_builder import OISCurveBuilder
from src.curves.swap_curve_builder import SwapCurveBuilder
from src.curves.inflation_curve_builder import InflationCurveBuilder
from src.pricers.swap_pricer import SwapPricer
from src.pricers.inflation_pricer import InflationPricer
from src.strategy_analytics.option_strategy import OptionStrategy
from src.market_data_processors.economic_data_processor import EconomicDataProcessor
from src.data_manager import b_con
from src.date_utils import ql_to_datetime, to_bbg_date_str
from src.financial_math_utils import px_dec_to_frac, px_opt_ticks, round_nearest
from src.config import config


class MatplotlibPlotter(BasePlotter):
    """
    Provides a collection of Matplotlib-based plotting functions for financial data.
    """

    def __init__(self):
        super().__init__()
        self.swap_pricer = SwapPricer()
        self.inflation_pricer = InflationPricer()
        self.eco_data_processor = EconomicDataProcessor()

    def plot(self, plot_type: str, data: Any, **kwargs) -> plt.Figure:
        """
        Generic plot method to dispatch to specific plotting functions.

        Args:
            plot_type (str): The type of plot to generate (e.g., 'swap_curve', 'inflation_curve', etc.).
            data (Any): The primary data required for the plot (e.g., list of curve builders, option strategy object).
            **kwargs: Additional parameters specific to the plot type.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        if plot_type == 'swap_curve':
            return self.plot_swap_curve(curve_builders=data, **kwargs)
        elif plot_type == 'inflation_curve':
            return self.plot_inflation_curve(inflation_curve_builders=data, **kwargs)
        elif plot_type == 'option_strategy':
            return self.plot_option_strategy(strategy=data, **kwargs)
        elif plot_type == 'vol_surface':
            return self.plot_vol_surface(vol_surface_builders=data, **kwargs)
        elif plot_type == 'economic_forecast':
            return self.plot_economic_forecast(**kwargs)
        elif plot_type == 'generic_timeseries':
            return self.plot_generic_timeseries(tickers=data, **kwargs)
        elif plot_type == 'economic_heatmap':
            return self.plot_economic_heatmap(eco_data_dfs=data, **kwargs)
        elif plot_type == 'curve_heatmap':
            return self.plot_curve_heatmap(curve_hmap_data=data, **kwargs)
        elif plot_type == 'rates_heatmap':
            return self.plot_rates_heatmap(curve_hmap_data=data, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    def plot_swap_curve(self,
                        curve_builders: List[Union[SwapCurveBuilder, OISCurveBuilder]],
                        # Original `c1` as list of names, now builders
                        max_tenor_years: int = 30,
                        bar_changes: bool = False,
                        show_spreads: bool = False,
                        plot_title: str = ''  # Original `name`
                        ) -> plt.Figure:
        """
        Plots interest rate swap curves (e.g., Libor/Euribor, OIS).
        Can show multiple curves, historical changes, and spreads.

        Args:
            curve_builders (List[Union[SwapCurveBuilder, OISCurveBuilder]]): List of built curve builder objects.
            max_tenor_years (int): Maximum tenor in years for the plot.
            bar_changes (bool): If True, plots changes between the first curve and subsequent historical curves as bar charts.
            show_spreads (bool): If True, plots spreads between the first curve and subsequent curves.
            plot_title (str): Title of the plot.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        n_curves = len(curve_builders)
        n_historical_points = len(curve_builders)  # Assuming each builder is for a different historical point

        # Determine number of subplots
        num_subplots = 1  # Always at least one for the curves
        if show_spreads:  #
            num_subplots += (n_curves - 1)  # One subplot for each spread against the first curve
        if bar_changes:  #
            if n_historical_points > 1:  # Only if historical points exist
                num_subplots += (n_curves * (n_historical_points - 1)) if not show_spreads else (n_curves - 1) * (
                            n_historical_points - 1)

        # Determine height ratios for subplots
        height_ratios = [2.5] + [1] * (num_subplots - 1)

        fig, axs = self._setup_subplots(num_subplots, height_ratios=height_ratios, figsize=(14, 10), hspace=0)

        if num_subplots == 1:
            axes_list = [axs]
        else:
            axes_list = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        # Plot curves
        rates_data: Dict[str, List[float]] = {}  # To store rates for each curve

        for i, builder in enumerate(curve_builders):
            builder.build_curve()  # Ensure curve is built

            # For OIS, plot 1-day forward rates. For Swaps, plot annual spot rates.
            if isinstance(builder, OISCurveBuilder):
                # Dates for OIS curve plotting (up to max_tenor_years)
                start_date = builder.reference_date
                end_date = start_date + ql.Period(max_tenor_years, ql.Years)
                ql_dates = [ql.Date(serial) for serial in range(start_date.serialNumber(), end_date.serialNumber() + 1)]
                yr_axis = [(d - start_date) / 365.25 for d in ql_dates]  # Years from start

                # Get 1-day forward rates from OIS curve
                rates = [builder.curve.forwardRate(d, builder.calendar.advance(d, 1, ql.Days),
                                                   builder.curve.dayCount(), ql.Simple).rate() * 100 for d in ql_dates]
                label = f"{builder.currency_code} OIS: {ql_to_datetime(builder.reference_date).strftime('%Y-%m-%d')}"

            elif isinstance(builder, SwapCurveBuilder):
                yr_axis = list(range(1, max_tenor_years + 1))  #
                rates = []
                for tenor in yr_axis:
                    # Calculate spot swap rates using the SwapPricer
                    swap_params = SwapParameters(
                        currency_code=builder.currency_code,
                        start_tenor=0,  # Spot starting
                        maturity_tenor=tenor  # Years
                    )
                    swap_obj = Swap(swap_params, pricing_curve_builder=builder)
                    rates.append(swap_obj.calculate_fair_rate())
                label = f"{builder.currency_code} Swap: {ql_to_datetime(builder.reference_date).strftime('%Y-%m-%d')}"
            else:
                raise ValueError(f"Unsupported curve builder type: {type(builder)}")

            rates_data[label] = rates  # Store rates for spread/change calculations

            # Plot current curves on the first subplot
            ax = axes_list[0]
            ax.plot(yr_axis, rates, lw=0.5, marker='.', label=label)
            ax.xaxis.tick_top()
            self._add_grid_and_labels(ax, y_label='Rate (%)')
            ax.legend(prop={"size": 9}, bbox_to_anchor=(1, 1), loc='upper left')

        # Plot spreads
        current_subplot_idx = 1
        if show_spreads and n_curves > 1:  #
            ax_spread = axes_list[current_subplot_idx]
            self._add_grid_and_labels(ax_spread, y_label='Spread (bps)')
            plt.setp(ax_spread, xlim=axes_list[0].get_xlim())

            first_curve_label = list(rates_data.keys())[0]  #
            first_curve_rates = np.array(rates_data[first_curve_label])  #

            for i in range(1, n_curves):
                comparison_curve_label = list(rates_data.keys())[i]  #
                comparison_curve_rates = np.array(rates_data[comparison_curve_label])  #

                # Ensure arrays are same length for element-wise operation
                min_len = min(len(first_curve_rates), len(comparison_curve_rates))
                spread = 100 * (first_curve_rates[:min_len] - comparison_curve_rates[:min_len])

                ax_spread.plot(yr_axis[:min_len], spread, lw=0.5, marker='.',
                               label=f"{first_curve_label.split(':')[0]} - {comparison_curve_label.split(':')[0]}")
            ax_spread.legend(prop={"size": 9}, bbox_to_anchor=(1, 1), loc='upper left')
            current_subplot_idx += 1

        # Plot bar changes
        if bar_changes and n_historical_points > 1:  #
            # If `show_spreads` is true, we plot changes of spreads. Otherwise, changes of outrights.
            # Original logic (from `PLOT.py`): If sprd=1, `bar_dict = rates_sprd`. Else, `bar_dict = rates_chg`.

            # To handle multiple historical points, we need to compare current vs each historical.
            # Assuming `curve_builders[0]` is the current/reference curve.

            # Example logic for original `plt_curve`: `h1` contains offsets like `[0, -1, -30]`.
            # `rates_diff` was `100*(np.array(rates_diff[c1[int(np.floor(i/len(h1)))]])[0] - np.array(rates_c))`
            # This was current curve vs historical.

            # Iterating through each curve builder (except the first one, which is 'current')
            # and plotting its change relative to the first (current) curve.

            # The structure from `PLOT.py` for bar changes `bar_dict[list(bar_dict.keys())[i]][m]`
            # implies that `bar_dict` is structured by currency (key) and then by historical offset (list of rates).

            # Let's adjust `rates_data` to also include changes relative to `curve_builders[0]`
            changes_data: Dict[str, Dict[str, List[float]]] = {}  # {ccy_label: {offset_label: rates_list}}
            first_builder = curve_builders[0]  #
            first_builder_rates = np.array(rates_data[list(rates_data.keys())[0]])  #

            for i in range(1, n_historical_points):  # Iterate through historical builders
                hist_builder = curve_builders[i]  #
                hist_builder_rates = np.array(rates_data[list(rates_data.keys())[i]])  #

                # Ensure same length for comparison
                min_len = min(len(first_builder_rates), len(hist_builder_rates))
                rate_change = 100 * (first_builder_rates[:min_len] - hist_builder_rates[:min_len])  #

                # Label for the change, using historical date
                change_label = f"vs {ql_to_datetime(hist_builder.reference_date).strftime('%Y-%m-%d')}"

                if first_builder.currency_code not in changes_data:  #
                    changes_data[first_builder.currency_code] = {}  #
                changes_data[first_builder.currency_code][change_label] = rate_change.tolist()  #

            plot_idx_offset = 0  # To manage subplot index
            if show_spreads:
                plot_idx_offset = 1  # Start after spreads plot

            bar_width = 0.15  #
            for ccy_label, historical_changes in changes_data.items():  #
                # Each currency gets its own row of bar charts if multiple currencies and `n_historical_points > 1`
                # If `n_curves == 1` and `n_historical_points > 1`, this will be one row of bars.
                # If `n_curves > 1`, then `bar_changes` plotting can be per-currency.

                if n_curves == 1:  # Single currency, multiple historical points
                    ax_change = axes_list[current_subplot_idx]
                    self._add_grid_and_labels(ax_change, y_label='Change (bps)')
                    plt.setp(ax_change, xlim=axes_list[0].get_xlim())

                    bar_offset_along_x = 0  # For staggering bars
                    for change_label, rates_list in historical_changes.items():  #
                        ax_change.bar(yr_axis[:len(rates_list)] + bar_offset_along_x, rates_list, bar_width,
                                      label=change_label)
                        bar_offset_along_x += bar_width + 0.1  #
                    ax_change.legend(prop={"size": 9}, bbox_to_anchor=(1, 1), loc='upper left')
                    current_subplot_idx += 1
                else:  # Multiple currencies, for each currency show changes vs first curve
                    # This implies a subplot per currency showing its own changes
                    # This part needs careful mapping to original `PLOTS.py` subplot logic
                    # The original `PLOTS.py` logic `axs[j].bar` where `j` is complex
                    # suggesting dedicated subplot for each currency's changes.
                    # For `plt_curve(['USD_3M','EUR_6M'], h1=[0,-1], bar_chg = 1, sprd = 0)`:
                    # Plots Curve for USD, EUR. Then two rows of bars (USD changes, EUR changes).
                    # This would be `current_subplot_idx = 1` for USD bars, `current_subplot_idx = 2` for EUR bars.

                    # For now, plot all changes on one subplot if `num_subplots` allows, else create new ones.
                    # Replicates the spirit of `PLOTS.py` where it might add many subplots.
                    ax_change_idx = current_subplot_idx  #
                    if ax_change_idx >= len(axes_list):  # Dynamically add more subplots if needed by `bar_changes`
                        # This should ideally be determined upfront in `_setup_subplots`
                        # For now, print warning if we run out of predefined subplots.
                        print(
                            f"Warning: Not enough subplots allocated for all bar changes. Plotting {ccy_label} changes on existing subplot {ax_change_idx - 1}.")
                        ax_change = axes_list[-1]
                    else:
                        ax_change = axes_list[ax_change_idx]

                    self._add_grid_and_labels(ax_change, y_label=f'{ccy_label} Change (bps)')
                    plt.setp(ax_change, xlim=axes_list[0].get_xlim())

                    bar_offset_along_x = 0  # For staggering bars
                    for change_label, rates_list in historical_changes.items():
                        ax_change.bar(yr_axis[:len(rates_list)] + bar_offset_along_x, rates_list, bar_width,
                                      label=change_label)
                        bar_offset_along_x += bar_width + 0.1
                    ax_change.legend(prop={"size": 9}, bbox_to_anchor=(1, 1), loc='upper left')
                    current_subplot_idx += 1

        if plot_title:  #
            fig.suptitle(plot_title, fontsize=12, y=0.98)  # Use suptitle for overall title
        plt.tight_layout()
        return fig

    def plot_inflation_curve(self,
                             inflation_curve_builders: List[InflationCurveBuilder],  # Original `c1`
                             max_tenor_years: int = 30,
                             bar_changes: bool = False,
                             show_spreads: bool = False,
                             plot_title: str = ''  # Original `name`
                             ) -> plt.Figure:
        """
        Plots inflation zero-coupon curves.
        Can show multiple curves, historical changes, and spreads.

        Args:
            inflation_curve_builders (List[InflationCurveBuilder]): List of built inflation curve builder objects.
            max_tenor_years (int): Maximum tenor in years for the plot.
            bar_changes (bool): If True, plots changes between the first curve and subsequent historical curves as bar charts.
            show_spreads (bool): If True, plots spreads between the first curve and subsequent curves.
            plot_title (str): Title of the plot.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        n_curves = len(inflation_curve_builders)  #
        n_historical_points = len(inflation_curve_builders)  # Assuming each builder is for a different historical point

        # Determine number of subplots
        num_subplots = 1
        if show_spreads and n_curves > 1:  #
            num_subplots += 1
        if bar_changes and n_historical_points > 1:  #
            # Original `PLOT.py` logic based on `n_chg` (number of historical points)
            # and `n_ccy` (number of currencies).
            # If `n_ccy == 1`, one bar chart row. If `n_ccy > 1`, multiple rows (one per currency).
            if n_curves == 1:
                num_subplots += 1
            else:
                num_subplots += (n_curves - 1)  # Changes in spreads or individual curves

        height_ratios = [2.5] + [1] * (num_subplots - 1)
        fig, axs = self._setup_subplots(num_subplots, height_ratios=height_ratios, figsize=(10, 8), hspace=0)

        if num_subplots == 1:
            axes_list = [axs]
        else:
            axes_list = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        # Plot curves
        rates_data: Dict[str, List[float]] = {}  # {label: rates_list}
        yr_axis = list(range(1, max_tenor_years + 1))  #

        for i, builder in enumerate(inflation_curve_builders):  #
            builder.build_curve()  # Ensure curve is built

            rates = []
            for tenor in yr_axis:
                metrics = self.inflation_pricer.calculate_zc_metrics(
                    builder,
                    start_date_input=0,  # Spot starting
                    tenor_years=tenor,
                    lag_months=builder._lag_months,  # Use builder's configured lag
                    use_market_fixing_curve=True  # Default to market fixing curve for inflation
                )
                rates.append(metrics['zc_rate'])

            label = f"{builder._inflation_index_name}: {ql_to_datetime(builder.reference_date).strftime('%Y-%m-%d')}"
            rates_data[label] = rates  #

            # Plot current curves on the first subplot
            ax = axes_list[0]
            ax.plot(yr_axis, rates, lw=0.5, marker='.', label=label)
            self._add_grid_and_labels(ax, y_label='Rate (%)')
            ax.legend(prop={"size": 9}, bbox_to_anchor=(0.72, 1), loc='upper left')

        # Plot spreads
        current_subplot_idx = 1
        if show_spreads and n_curves > 1:  #
            ax_spread = axes_list[current_subplot_idx]
            self._add_grid_and_labels(ax_spread, y_label='Spread (bps)')
            plt.setp(ax_spread, xlim=axes_list[0].get_xlim())

            first_curve_label = list(rates_data.keys())[0]  #
            first_curve_rates = np.array(rates_data[first_curve_label])  #

            for i in range(1, n_curves):
                comparison_curve_label = list(rates_data.keys())[i]  #
                comparison_curve_rates = np.array(rates_data[comparison_curve_label])  #

                # Ensure arrays are same length
                min_len = min(len(first_curve_rates), len(comparison_curve_rates))
                spread = 100 * (first_curve_rates[:min_len] - comparison_curve_rates[:min_len])

                ax_spread.plot(yr_axis[:min_len], spread, lw=0.5, marker='.',
                               label=f"{first_curve_label.split(':')[0]} - {comparison_curve_label.split(':')[0]}")
            ax_spread.legend(prop={"size": 9}, bbox_to_anchor=(0.8, 0.3), loc='upper left')
            current_subplot_idx += 1

        # Plot bar changes
        if bar_changes and n_historical_points > 1:  #
            changes_data: Dict[str, Dict[str, List[float]]] = {}  # {index_name: {offset_label: rates_list}}
            first_builder = inflation_curve_builders[0]  #
            first_builder_rates = np.array(rates_data[list(rates_data.keys())[0]])  #

            for i in range(1, n_historical_points):  # Iterate through historical builders
                hist_builder = inflation_curve_builders[i]  #
                hist_builder_rates = np.array(rates_data[list(rates_data.keys())[i]])  #

                min_len = min(len(first_builder_rates), len(hist_builder_rates))
                rate_change = 100 * (first_builder_rates[:min_len] - hist_builder_rates[:min_len])  #

                change_label = f"vs {ql_to_datetime(hist_builder.reference_date).strftime('%Y-%m-%d')}"

                if first_builder._inflation_index_name not in changes_data:  #
                    changes_data[first_builder._inflation_index_name] = {}  #
                changes_data[first_builder._inflation_index_name][change_label] = rate_change.tolist()  #

            bar_width = 0.15
            for index_name, historical_changes in changes_data.items():  #
                ax_change = axes_list[current_subplot_idx]
                self._add_grid_and_labels(ax_change, y_label=f'{index_name} Change (bps)')
                plt.setp(ax_change, xlim=axes_list[0].get_xlim())

                bar_offset_along_x = 0  # For staggering bars
                for change_label, rates_list in historical_changes.items():  #
                    ax_change.bar(yr_axis[:len(rates_list)] + bar_offset_along_x, rates_list, bar_width,
                                  label=change_label)
                    bar_offset_along_x += bar_width + 0.1  #
                ax_change.legend(prop={"size": 9}, bbox_to_anchor=(1, 1), loc='upper left')
                current_subplot_idx += 1

        if plot_title:  #
            fig.suptitle(plot_title, fontsize=12, y=0.98)  #
        plt.tight_layout()
        return fig

    def plot_option_strategy(self,
                             strategy: OptionStrategy,  # Original `st`
                             add_delta_hedge: Optional[float] = None,  # Original `add_delta`
                             payoff_increment_calc: int = 100,
                             strategy_pv_increment: float = 0.25  # Original `strat_pv_increm`
                             ) -> plt.Figure:
        """
        Plots the payoff and delta profile for an option strategy.

        Args:
            strategy (OptionStrategy): An OptionStrategy object containing simulation results.
            add_delta_hedge (Optional[float]): If provided, adds a linear delta hedge to the strategy payoff/delta.
                                               This is typically the notional/delta of the futures hedge.
            payoff_increment_calc (int): Number of points for calculating the expiry payoff curve.
            strategy_pv_increment (float): Increment for Y-axis ticks on the strategy P&L plot.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        strategy_details = strategy.strategy_details  #
        strategy_df = strategy_details['strategy_simulation_df']  #

        spot_idx = strategy_df.loc[strategy_df['ATM_K'] == 0].index[0]  #

        # Determine price label format based on strategy type (USD for bond, stir for STIR)
        px_label: str
        if strategy_details['strategy_type'] == 'USD':  #
            px_label = px_opt_ticks(strategy_details['strategy_current_px'])
        elif strategy_details['strategy_type'] == 'stir':  #
            px_label = str(np.round(100 * strategy_details['strategy_current_px'], 2))
        else:  #
            px_label = str(np.round(strategy_details['strategy_current_px'], 3))  #

        fig, axs = self._setup_subplots(2, height_ratios=[2.5, 1], figsize=(8, 6), hspace=0.25)
        ax1, ax2 = axs  #

        # --- Plot 1: Payoff at Expiry and Strategy P&L ---
        ax1.set_ylabel('Strategy P&L')
        ax1.set_title(
            f"{strategy_details['strategy_name']} | Px: {px_label} | Delta: {np.round(strategy_df['strat_delta'][spot_idx], 1)}")

        x_spot = strategy_df['fut_px'][spot_idx]

        # Payoff at Expiry (original `fut_payoff` and `opt_payoff` functions)
        x_expiry_sim = np.linspace(strategy_df['fut_px'].min(), strategy_df['fut_px'].max(), num=payoff_increment_calc)

        expiry_payoff = np.zeros_like(x_expiry_sim, dtype=float)

        # Calculate option components of expiry payoff
        option_types_list = strategy_details['option_details'][1]  #
        strikes_list = np.asarray(strategy_details['option_details'][2])  #
        weights_list = np.asarray(strategy_details['option_details'][3])  #

        for i in range(len(option_types_list)):  #
            opt_type = option_types_list[i]  #
            strike = strikes_list[i]  #
            weight = weights_list[i]  #

            if opt_type == 'C':  #
                payoff = np.maximum(0, x_expiry_sim - strike)  #
            elif opt_type == 'P':  #
                payoff = np.maximum(0, strike - x_expiry_sim)  #
            else:
                pass  # Already handled by OptionStrategy validation

            expiry_payoff += payoff * weight  #

        # Add initial cost offset to expiry payoff
        expiry_payoff -= strategy_details['strategy_current_px']

        # Add linear futures hedge if specified (`add_delta_hedge`)
        if add_delta_hedge is not None:  #
            # `fut_payoff` was inferred to be `multiplier[0] * (spot_prices)`
            # Here, `add_delta_hedge` is assumed to be the total delta notional of the hedge
            # Payoff of a futures hedge is `delta * (final_price - initial_price)`.
            # We need the initial price (current spot) to calculate P&L.
            expiry_payoff += add_delta_hedge * (x_expiry_sim - x_spot)  #

        ax1.plot(x_expiry_sim, expiry_payoff, lw=0.5, color='k', label='Payoff at Expiry')

        # Strategy P&L (simulated)
        ax1.plot(strategy_df['fut_px'], strategy_df['strat_px'] - strategy_details['strategy_current_px'], marker='x',
                 markersize=5, lw=0.5, label='Strategy P&L')  #

        ax1.axvline(x_spot, lw=0.5, color='y', linestyle='--', label='Current Spot')  #
        ax1.axhline(0, lw=0.5, color='k', linestyle=':', label='Break-even')

        self._add_grid_and_labels(ax1, y_label='Strategy P&L')

        # X-axis ticks and labels (strikes, ATM_K, Delta)
        # Original code removed spot_idx from labels to avoid overlap.
        # This can be simplified.
        x_ticks = strategy_df['fut_px'].tolist()
        x_labels = [f"{px:.1f}\n\n{atm:.1f}\n\n{delta:.1f}" for px, atm, delta in
                    zip(strategy_df['fut_px'], strategy_df['ATM_K'], strategy_df['strat_delta'])]

        ax1.set_xticks(x_ticks)  #
        ax1.set_xticklabels(x_labels, fontsize=9)

        # Y-axis ticks for P&L
        min_y_val = min(expiry_payoff.min(), (strategy_df['strat_px'] - strategy_details['strategy_current_px']).min())
        max_y_val = max(expiry_payoff.max(), (strategy_df['strat_px'] - strategy_details['strategy_current_px']).max())
        payoff_ticks = np.arange(round_nearest(min_y_val, strategy_pv_increment),
                                 round_nearest(max_y_val + strategy_pv_increment, strategy_pv_increment),
                                 strategy_pv_increment)

        ax1.set_yticks(payoff_ticks)  #
        if strategy_details['strategy_type'] == 'USD':  #
            ax1.set_yticklabels([px_opt_ticks(p) for p in payoff_ticks])
        else:
            ax1.set_yticklabels([f"{p:.2f}" for p in payoff_ticks])  #
        ax1.legend(prop={"size": 8}, loc='upper left')

        # --- Plot 2: Strategy Delta ---
        ax2.set_ylabel('Strategy Delta')
        ax2.plot(strategy_df['fut_px'], strategy_df['strat_delta'] / 100, marker='x', markersize=5, lw=0.5, color='g',
                 label='Delta')
        ax2.axvline(x_spot, lw=0.5, color='y', linestyle='--', label='Current Spot')

        # Add delta hedge line if specified
        if add_delta_hedge is not None:  #
            ax2.plot(strategy_df['fut_px'], (strategy_df['strat_delta'] / 100) + add_delta_hedge,
                     marker='x', markersize=5, lw=0.5, color='k', ls='dashed', label='Delta + Hedge')  #
            # Update strategy_df if original code used this for further analysis
            strategy_df['strat_delta_updt'] = (strategy_df['strat_delta'] / 100) + add_delta_hedge

        self._add_grid_and_labels(ax2)
        ax2.legend(prop={"size": 8}, loc='upper left')

        plt.tight_layout()
        return fig

    def plot_vol_surface(self,
                         vol_surface_builders: List[Union[BondFutureVolSurface, StirVolSurface]],  # Original `v`
                         option_type: str = 'P',
                         x_axis_type: str = 'delta'  # Original `x_ax` ('delta' or 'strikes')
                         ) -> plt.Figure:
        """
        Plots a volatility surface (implied volatility vs. delta or strike).

        Args:
            vol_surface_builders (List[Union[BondFutureVolSurface, StirVolSurface]]):
                List of built volatility surface builder objects (e.g., for different dates).
            option_type (str): 'C' for Calls, 'P' for Puts.
            x_axis_type (str): The type of x-axis to plot ('delta' or 'strikes').

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        fig, ax = self._setup_subplots(1, figsize=(8, 6))
        ax.set_ylabel('Implied Volatility (bp)')
        ax.set_xlabel(x_axis_type.capitalize())

        if vol_surface_builders:  #
            # Use the first builder's ticker root for the title
            title_ticker = vol_surface_builders[0].underlying_ticker.split(' ')[0][:4]
            ax.set_title(f"{title_ticker} {option_type.upper()}")

        ql_option_type_int = 1 if option_type == 'C' else -1  # 1 for Call, -1 for Put

        for builder in vol_surface_builders:  #
            builder.build_surface(builder.underlying_ticker.split(' ')[0][:4],
                                  builder.chain_length)  # Ensure surface is built
            df = builder.surface_data  #

            # Filter for the requested option type
            df_filtered = df[df['option_type'] == option_type]

            if x_axis_type == 'strikes':  #
                x_values = df_filtered['strikes']
            elif x_axis_type == 'delta':  #
                x_values = df_filtered['delta']
                # For puts, delta is negative, original plot flips it `-vol_s2['ATM_K']`.
                # If delta is intended to be plotted as absolute or relative to 0 for symmetry.
                # Assuming here that `delta` is plotted as-is, which will be negative for puts.
            else:
                raise ValueError(f"Unsupported x-axis type: {x_axis_type}. Choose 'delta' or 'strikes'.")

            ax.plot(x_values, df_filtered['iv'], marker='.',
                    label=ql_to_datetime(builder.reference_date).strftime('%Y-%m-%d'))

            # Annotate ATM strike (if x_axis_type is strike or relevant)
            if x_axis_type == 'strikes':  #
                atm_strike = builder.at_the_money_strike  #
                atm_option = df_filtered[df_filtered['strikes'] == atm_strike]  #
                if not atm_option.empty:
                    ax.annotate(str(atm_strike),
                                xy=(atm_option[x_values.name].iloc[0], atm_option['iv'].iloc[0]),
                                fontsize=8, horizontalalignment='center', verticalalignment='top')
            elif x_axis_type == 'delta':  # For delta plots, ATM is around 50 for calls, -50 for puts
                # Add a vertical line for ATM delta (50 or -50)
                atm_delta_line = 50 if option_type == 'C' else -50  #
                ax.axvline(x=atm_delta_line, color='grey', linestyle=':', linewidth=0.8)  #

        ax.legend(prop={"size": 9}, loc='best')
        self._add_grid_and_labels(ax)
        plt.tight_layout()  #
        return fig

    def plot_economic_forecast(self,
                               economic_data_type: str,  # 'GDP', 'CPI', 'PCE', 'Core-PCE', 'UNEMP', 'FISC'
                               country_code: str,  # 'EU', 'US'
                               forecast_year: str,  # '2023', '2024'
                               contributor_highlight: str = 'GS',  # 'GS', 'BAR', 'JPM'
                               official_source: str = 'IMF'  # 'IMF', 'ECB', 'FED'
                               ) -> plt.Figure:
        """
        Plots historical economic forecasts from various contributors and an official source.

        Args:
            economic_data_type (str): Type of economic data (e.g., 'GDP', 'CPI').
            country_code (str): Country code for the forecast (e.g., 'EU', 'US').
            forecast_year (str): The forecast year (e.g., '2023', '2024').
            contributor_highlight (str): The contributor to highlight in the plot.
            official_source (str): The official source to plot.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        # Mapping economic data type to Bloomberg convention (e.g., 'PI' for CPI)
        eco_type_map = {
            'GDP': 'GD', 'CPI': 'PI', 'PCE': 'PC', 'Core-PCE': 'CC',
            'UNEMP': 'UP', 'FISC': 'BB'
        }
        bbg_eco_code = eco_type_map.get(economic_data_type)
        if not bbg_eco_code:
            raise ValueError(f"Unsupported economic data type: {economic_data_type}")

        # Standard list of contributors based on original `PLOT.py`
        all_contributors = [
            'BAR', 'BOA', 'BNP', 'CE', 'CIT', 'CAG', 'CSU', 'DNS', 'FTC', 'GS', 'HSB', 'IG',
            'JPM', 'MS', 'NTX', 'NS', 'NDA', 'PMA', 'UBS', 'WF', 'SCB'
        ]

        today_dt = datetime.datetime.now()  #
        bbg_today_str = to_bbg_date_str(today_dt, ql_date=0)

        # Fetch data for all contributors and official source
        # NOTE: Bloomberg ticker construction is inferred. Please verify.
        # Original: `EC` + eco_code + `b` + `c[-2:]` + `i` + `Index`
        # `b` is country_code, `c[-2:]` is last two digits of forecast_year.
        #
        contributor_tickers = [f"EC{bbg_eco_code}{country_code} {forecast_year[-2:]} {contrib} Index" for contrib in
                               all_contributors]
        official_ticker = f"EC{bbg_eco_code}{country_code} {forecast_year[-2:]} {official_source} Index"

        try:
            df_contributors_raw = b_con.bdh(contributor_tickers, 'PX_LAST',
                                            to_bbg_date_str(today_dt - datetime.timedelta(days=365), ql_date=0),
                                            bbg_today_str, longdata=True)
            df_official_raw = b_con.bdh(official_ticker, 'PX_LAST',
                                        to_bbg_date_str(today_dt - datetime.timedelta(days=365), ql_date=0),
                                        bbg_today_str, longdata=True)

            if df_contributors_raw.empty or df_official_raw.empty:
                print("Warning: No economic forecast data found for requested parameters.")  #
                return plt.figure()  # Return empty figure

            # Calculate average forecast over time (rolling average inspired by original `m1`)
            # Original `d1` was `[ql_to_datetime(cal.advance(today, ql.Period( int(-10-i), ql.Days))) for i in np.arange(240)]`
            # This represents a shifting window ending 10 days before current `d1[i]`
            # For simplicity, calculate a 30-day rolling mean for the average line.
            df_contributors_raw['date'] = pd.to_datetime(df_contributors_raw['date'])
            df_avg = df_contributors_raw.groupby('date')['value'].mean().rolling(window='30D').mean().reset_index()
            df_avg.columns = ['date', 'avg_value']

            fig, ax = self._setup_subplots(1, figsize=(6.5, 4.75))

            # Plot individual contributor lines
            unique_contributor_tickers = df_contributors_raw['ticker'].unique()
            for ticker in unique_contributor_tickers:
                contrib_df = df_contributors_raw[df_contributors_raw['ticker'] == ticker]
                contrib_name = ticker.split(' ')[-2]

                if contrib_name == contributor_highlight:
                    line_color = 'darkorange'
                    label_text = f"{contrib_name}: {contrib_df['value'].iloc[-1]:.2f}"
                else:
                    line_color = 'silver'
                    label_text = ''  # Don't label others

                ax.plot(contrib_df['date'], contrib_df['value'], c=line_color, label=label_text)

            # Plot official source
            ax.plot(df_official_raw['date'], df_official_raw['value'], c='forestgreen',
                    label=f"{official_source}: {df_official_raw['value'].iloc[-1]:.2f}")

            # Plot average line
            ax.plot(df_avg['date'], df_avg['avg_value'], c='dodgerblue',
                    label=f"Avg: {df_avg['avg_value'].iloc[-1]:.2f}")

            self._add_grid_and_labels(ax)
            ax.set_title(f"{country_code} {economic_data_type} forecast: {forecast_year}")
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            plt.tight_layout()
            return fig

    def plot_generic_timeseries(self,
                                tickers: List[str],  # Original `ticker_list` or `curves`
                                start_date_input: Union[str, int, datetime.datetime, ql.Date],  # Original `d1`
                                end_date_input: Union[str, int, datetime.datetime, ql.Date] = '',  # Original `d2`
                                plot_type: str = 'Outright',  # 'Outright', 'Spread', 'Fly'
                                change_flag: bool = False,  # If True, plots cumulative changes
                                invert_flag: bool = False,  # If True, inverts values
                                # For internal use by plotool:
                                bbg_instrument_type: str = 'Fwd',  # 'Par', 'Fwd', 'Cash'
                                curve_codes: Optional[List[str]] = None  # e.g., ['USD_3M', 'GBP_6M']
                                ) -> plt.Figure:
        """
        Generic function to fetch and plot historical time series data for Bloomberg instruments.

        This function consolidates the logic from the original `plotool` in PLOT.py.

        Args:
            tickers (List[str]): List of Bloomberg tickers or financial instrument codes.
                                 If `curve_codes` is provided, this list will be used as maturities.
            start_date_input (Union[str, int, datetime.datetime, ql.Date]): Start date for data.
            end_date_input (Union[str, int, datetime.datetime, ql.Date]): End date for data.
            plot_type (str): Type of plot ('Outright', 'Spread', 'Fly').
            change_flag (bool): If True, plots cumulative changes.
            invert_flag (bool): If True, inverts the plotted values.
            bbg_instrument_type (str): Bloomberg instrument type prefix ('Par', 'Fwd', 'Cash').
            curve_codes (Optional[List[str]]): List of currency/index codes (e.g., 'USD_3M', 'GBP_6M').
                                              If provided, `tickers` are interpreted as maturities.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        # Determine actual start and end dates in QuantLib format
        ql_start_date = get_ql_date(start_date_input)
        ql_end_date = get_ql_date(end_date_input) if end_date_input else ql.Settings.instance().evaluationDate

        bbg_start_date_str = to_bbg_date_str(ql_start_date, ql_date=1)
        bbg_end_date_str = to_bbg_date_str(ql_end_date, ql_date=1)

        final_tickers_to_fetch: List[str] = []
        labels_for_plot: List[str] = []

        # Build Bloomberg tickers based on input `curve_codes` and `tickers` (maturities)
        # Or directly use `tickers` if `curve_codes` is None.
        if curve_codes:  # This implies `tickers` are maturities like '5Y', '10Y', '5Y5Y'
            # Construct Bloomberg tickers (e.g., 'S0USSF FS 5Y5Y BLC Curncy')
            # NOTE: Bloomberg ticker construction is inferred. Please verify.
            # It uses `S0` + `curve_code_suffix` + `bbg_instrument_type_prefix` + `maturity` + `suffix`.
            # `bbgplot_tickers` from config holds components.

            # Extract relevant parts from config for each curve_code
            curve_ticker_map = {}
            for c_code in curve_codes:  #
                ccy_conf = config.get_currency_config(c_code)  #
                bbg_plot_tkrs = ccy_conf.get('bbgplot_tickers', [])  #
                # Original `plotool` implies `curve_code[j]` refers to the suffix like `US` or `GB`.
                # And `str_att1` (FS/Z/FC), `str_att2` (BLC Curncy).

                # Assuming `bbg_plot_tkrs[0]` is the base for swap tickers like 'USSW', 'BPSWS'.
                # And `bbg_plot_tkrs[1]` is for forwards like 'USSF', 'BPSF'.
                # And `bbg_plot_tkrs[2]` for OIS futures (used in WIRP).
                #
                # The original `plotool` had `S0` prefix always: `S0`+`curve_code[j]`+`str_att1`+`mat[i]`+`str_att2`.
                # `curve_code[j]` was `c1[i].bbg_curve.split()[0][-3:]` which could be `SW`, `SF`.
                # This is confusing. Let's rely on `bbgplot_tickers` from config.

                prefix = ''
                suffix = ' Curncy'
                mid_part = ''

                if bbg_instrument_type == 'Par':
                    mid_part = 'Z '  #
                    suffix = ' BLC2 Curncy'  #
                    # Check if `bbg_plot_tkrs[0]` is base, e.g. `USSW`
                    base_prefix = bbg_plot_tkrs[0] if bbg_plot_tkrs else ''  #
                    final_prefix = f"S0{base_prefix[-2:]}" if base_prefix else "S0"  # S0US, S0BP
                elif bbg_instrument_type == 'Fwd':
                    mid_part = 'FS '  #
                    suffix = ' BLC Curncy'  #
                    base_prefix = bbg_plot_tkrs[1] if len(bbg_plot_tkrs) > 1 else ''  # e.g. `USSF`
                    final_prefix = f"S0{base_prefix[-2:]}" if base_prefix else "S0"  # S0US, S0BP
                elif bbg_instrument_type == 'Cash':
                    mid_part = 'FC '  #
                    suffix = ' BLC Curncy'  #
                    base_prefix = bbg_plot_tkrs[1] if len(bbg_plot_tkrs) > 1 else ''  #
                    final_prefix = f"S0{base_prefix[-2:]}" if base_prefix else "S0"  #

                curve_ticker_map[c_code] = {
                    'prefix': final_prefix,
                    'mid_part': mid_part,
                    'suffix': suffix
                }

            for c_code in curve_codes:  #
                map_details = curve_ticker_map[c_code]  #
                for maturity in tickers:  # `tickers` here are actual maturities like '5Y5Y', '10Y'
                    full_ticker = f"{map_details['prefix']}{c_code.split('_')[0]}{map_details['mid_part']}{maturity}{map_details['suffix']}"  #
                    # The construction `S0USSF FS 5Y5Y BLC Curncy` might be problematic if 'US' is not the correct suffix.
                    # It relies on `USD_3M` -> `US`.
                    # More robust: use lookup directly from `bbgplot_tickers` from `config.py`
                    # `bbgplot_tickers` might be more directly `['USSW', 'USSF', 'USSP']`
                    # So ticker could be `USSF 5Y5Y BLC Curncy`

                    # Revised ticker construction based on PLOT.py `plt_ois_curve`:
                    # `bbgplot_tickers[0]` for base, `[1]` for forwards etc.
                    # This needs to be very precise.

                    # Let's assume `bbg_plot_tickers` entry 0 is the base for Par, 1 for Fwd/Cash.
                    base_bbg_ticker_root = ''
                    if bbg_instrument_type == 'Par':
                        base_bbg_ticker_root = config.get_currency_config(c_code).get('bbgplot_tickers', [])[0]
                    elif bbg_instrument_type in ['Fwd', 'Cash']:
                        base_bbg_ticker_root = config.get_currency_config(c_code).get('bbgplot_tickers', [])[1]

                    if not base_bbg_ticker_root:
                        print(
                            f"Warning: No Bloomberg plot ticker root found for {c_code} with type {bbg_instrument_type}. Skipping.")
                        continue

                    full_ticker = f"{base_bbg_ticker_root} {maturity}{suffix}"  # E.g., 'USSF 5Y5Y BLC Curncy'
                    labels_for_plot.append(f"{c_code}:{maturity}")  #
                    final_tickers_to_fetch.append(full_ticker)  #

        else:  # `tickers` is already a list of full Bloomberg tickers
            final_tickers_to_fetch = tickers  #
            labels_for_plot = tickers  #

        # Fetch data from Bloomberg
        try:
            raw_data = b_con.bdh(final_tickers_to_fetch, 'PX_LAST', bbg_start_date_str, bbg_end_date_str)
            raw_data = raw_data.rename(columns={'value': 'PX_LAST'})  #

            # Pivot the DataFrame to have tickers as columns
            plot_df = raw_data.pivot_table(index='date', columns='ticker', values='PX_LAST')  #
            plot_df.columns = labels_for_plot  # Rename columns to user-friendly labels

            if plot_df.empty:
                print("No data fetched for plotting.")  #
                return plt.figure()  # Return empty figure

            # Apply inversion if requested
            if invert_flag:  #
                plot_df = plot_df * -1  #

            # Calculate cumulative changes if requested
            if change_flag:  #
                plot_df = plot_df.diff().cumsum()  #

            fig, ax = self._setup_subplots(1, figsize=(8, 6))

            # Plot the data
            ax.plot(plot_df.index, plot_df.values, lw=1)  #

            # Add legend with current values
            legend_labels = [f"{col}...{plot_df[col].iloc[-1]:.3f}" for col in plot_df.columns]
            ax.legend(legend_labels, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right", borderaxespad=0)

            # X-axis tick formatting
            ax.set_xticks(plot_df.index[::len(plot_df) // 10])  # Adjust tick frequency
            ax.set_xticklabels([d.strftime("%b-%y") for d in plot_df.index[::len(plot_df) // 10]], rotation=90)

            self._add_grid_and_labels(ax)  #
            plt.tight_layout()
            return fig

        except Exception as e:
            raise RuntimeError(f"Error plotting generic time series: {e}")  #

    def plot_economic_heatmap(self,
                              eco_data_dfs: Dict[str, pd.DataFrame],
                              # Output from EconomicDataProcessor.get_economic_data
                              instruments_for_overlay: Optional[List[str]] = None,  # Original `inst`
                              num_years_to_plot: int = 10,
                              minmax_scale_data: bool = True,
                              figsize: Tuple[int, int] = (20, 30)
                              ) -> plt.Figure:
        """
        Generates heatmaps for various economic data transformations (e.g., Z-scores).

        Args:
            eco_data_dfs (Dict[str, pd.DataFrame]): Dictionary of transformed economic data DataFrames
                                                    (e.g., 'zs_pct', 'zs_raw') from `get_economic_data`.
            instruments_for_overlay (Optional[List[str]]): List of Bloomberg tickers for instruments
                                                           to overlay on the heatmap (e.g., 'GT10 Govt').
            num_years_to_plot (int): Number of most recent years to include in the heatmap.
            minmax_scale_data (bool): If True, applies Min-Max scaling to the data for heatmap.
            figsize (Tuple[int, int]): Figure size for the plot.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        # Original `df1` was `[us.zs_pct]` so `eco_data_dfs` is assumed to be a list of DFs.
        # Here, `eco_data_dfs` is a dict from `get_economic_data`, so we need to select them.
        # Common usage implies plotting 'zs_pct'.
        dfs_to_plot = [df for name, df in eco_data_dfs.items() if name.startswith('zs_')]  #

        # Optionally add instrument data as a new column
        processed_dfs: List[pd.DataFrame] = []
        for df in dfs_to_plot:  #
            current_df = df.copy()  #
            if instruments_for_overlay:  #
                for inst_ticker in instruments_for_overlay:  #
                    try:  #
                        inst_data = b_con.bdh(
                            inst_ticker, 'PX_LAST',
                            to_bbg_date_str(current_df.index[0] - pd.DateOffset(days=32), ql_date=0),  #
                            to_bbg_date_str(current_df.index[-1] + pd.DateOffset(days=32), ql_date=0),  #
                            longdata=True  #
                        )  #
                        inst_data = inst_data.pivot_table(index='date', columns='ticker', values='value')  #
                        inst_data.index = pd.to_datetime(inst_data.index)  #
                        inst_data_monthly = inst_data.resample('M').last()  #
                        inst_data_monthly = inst_data_monthly.loc[current_df.index]  # Align index
                        current_df[inst_ticker.split(' ')[0]] = inst_data_monthly.iloc[:, 0].diff(
                            1)  # Original code uses diff(1) for instrument overlay
                        print(f"Overlaying {inst_ticker} data on heatmap.")
                    except Exception as e:  #
                        print(f"Warning: Could not overlay instrument {inst_ticker}: {e}")  #
            processed_dfs.append(current_df)  #

        num_dfs = len(processed_dfs)
        fig, axs = plt.subplots(num_dfs, 1, figsize=figsize)

        if num_dfs == 1:  #
            axes_list = [axs]  #
        else:
            axes_list = axs.flatten() if isinstance(axs, np.ndarray) else [axs]  #

        for i, df_to_plot in enumerate(processed_dfs):  #
            ax = axes_list[i]  #

            # Select the most recent `num_years_to_plot`
            plot_data = df_to_plot.iloc[(-num_years_to_plot * 12):]

            # Apply min-max scaling if requested
            scaled_data = plot_data.copy()
            if minmax_scale_data:  #
                # ZS_excl from original MINING.py was used to exclude cols from Z-score avg, not from minmax.
                # All columns are scaled for heatmap by default unless specified otherwise.
                for col in scaled_data.columns:
                    # minmax_scale expects 2D array, so reshape
                    scaled_data[col] = minmax_scale(scaled_data[col].values.reshape(-1, 1)).flatten()

            sns.heatmap(
                scaled_data.T,  # Transpose for heatmap
                cmap='vlag_r',
                linewidths=1.0,
                xticklabels=plot_data.index.strftime(("%b-%y")),  #
                yticklabels=plot_data.columns,  #
                fmt=".5g",  #
                cbar=False,
                ax=ax  #
            )
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=12)
            ax.set_title(df_to_plot.name if hasattr(df_to_plot, 'name') else f"Heatmap {i + 1}", fontsize=8,
                         color='indigo', loc='left')

            # Add right-side axis for last values
            ax2 = ax.twinx()
            ax2.set_ylim(0, len(df_to_plot.columns))
            ax2.set_yticks(np.arange(0.5, len(df_to_plot.columns) + 0.5, 1))
            ax2.set_yticklabels(np.round(df_to_plot.iloc[-1][::-1], 1))  # Last row, reversed
            ax2.yaxis.set_tick_params(labelsize=12)

        plt.tight_layout()  #
        return fig

    def plot_curve_heatmap(self,
                           curve_hmap_data: Dict[str, pd.DataFrame],
                           # Output from SwapTableGenerator.generate_swap_tables (specifically `curve_hmap` portion)
                           change_offset_key: str = '-1',
                           # Key for the desired change offset (e.g., '-1' for 1-day change)
                           roll_keys: List[str] = ['3M', '6M'],  # Keys for the desired roll data
                           ) -> plt.Figure:
        """
        Plots heatmaps for yield curve steepness/spreads and rolls.

        This function consolidates `curve_hm` from the original PLOT.py.

        Args:
            curve_hmap_data (Dict[str, pd.DataFrame]): Dictionary containing curve analysis DataFrames:
                                                      'steep', 'steep_chg', 'roll'.
            change_offset_key (str): Key identifying the specific change offset within 'steep_chg' (e.g., '-1').
            roll_keys (List[str]): List of keys for the specific roll data (e.g., ['3M', '6M']).

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        steep_df = curve_hmap_data['steep']
        steep_chg_dict = curve_hmap_data['steep_chg']
        roll_dict = curve_hmap_data['roll']

        # Ensure data is transposed for heatmap as in original
        steep_df_T = steep_df.transpose()
        steep_chg_df_T = steep_chg_dict.get(change_offset_key, pd.DataFrame()).transpose()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.5)

        # Plot 1: Curve Steepness
        ax1 = axs[0, 0]  #
        ax1.set_title('Curve', fontdict={'fontsize': 9})
        scaled_steep_df = minmax_scale(steep_df_T)  #
        sns.heatmap(
            scaled_steep_df, cmap='Blues', linewidths=1, annot=steep_df_T,
            xticklabels=steep_df_T.columns, yticklabels=steep_df_T.index,
            fmt=".5g", cbar=False, ax=ax1  #
        )

        # Plot 2: Change in Curve Steepness
        ax2 = axs[0, 1]  #
        ax2.set_title(
            f'Chg: {change_offset_key} | {ql_to_datetime(ql.Settings.instance().evaluationDate).strftime("%Y-%m-%d")}',
            fontdict={'fontsize': 9})
        scaled_steep_chg_df = minmax_scale(steep_chg_df_T)  #
        sns.heatmap(
            scaled_steep_chg_df, cmap='Purples_r', linewidths=1, annot=steep_chg_df_T,
            xticklabels=steep_chg_df_T.columns, yticklabels=steep_chg_df_T.index,
            fmt=".5g", cbar=False, ax=ax2  #
        )

        # Plot 3: Roll 1
        ax3 = axs[1, 0]  #
        ax3.set_title(f'Roll: {roll_keys[0]}', fontdict={'fontsize': 9})
        roll1_df_T = roll_dict.get(roll_keys[0], pd.DataFrame()).transpose()  #
        sns.heatmap(
            roll1_df_T, cmap='coolwarm_r', linewidths=1, annot=True,
            fmt=".5g", cbar=False, ax=ax3  #
        )

        # Plot 4: Roll 2
        ax4 = axs[1, 1]  #
        ax4.set_title(f'Roll: {roll_keys[1]}', fontdict={'fontsize': 9})
        roll2_df_T = roll_dict.get(roll_keys[1], pd.DataFrame()).transpose()  #
        sns.heatmap(
            roll2_df_T, cmap='coolwarm_r', linewidths=1, annot=True,
            fmt=".5g", cbar=False, ax=ax4  #
        )

        plt.tight_layout()
        return fig

    def plot_rates_heatmap(self,
                           curve_hmap_data: Dict[str, pd.DataFrame],
                           # Output from SwapTableGenerator.generate_swap_tables (specifically `curve_hmap` portion)
                           change_offset_key: str = '-1',
                           # Key for the desired change offset (e.g., '-1' for 1-day change)
                           ) -> plt.Figure:
        """
        Plots heatmaps for absolute interest rates and their changes, and forward rates.

        This function consolidates `rates_hm` from the original PLOT.py.

        Args:
            curve_hmap_data (Dict[str, pd.DataFrame]): Dictionary containing curve analysis DataFrames:
                                                      'rates', 'rates_chg', 'curves', 'chg'.
            change_offset_key (str): Key identifying the specific change offset within 'rates_chg' and 'chg'.

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        rates_df = curve_hmap_data['rates']
        rates_chg_dict = curve_hmap_data['rates_chg']
        curves_df = curve_hmap_data['curves']  # Original `curves` was for forwards
        chg_dict = curve_hmap_data['chg']

        rates_chg_df = rates_chg_dict.get(change_offset_key, pd.DataFrame())
        chg_df = chg_dict.get(change_offset_key, pd.DataFrame())

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Filter out '11Y', '13Y', '40Y' as per original `rates_hm`
        # `df_rates.drop(['11Y', '13Y', '40Y'])`
        filter_out_tenors = ['11Y', '13Y', '40Y']
        rates_df_filtered = rates_df.drop(index=[t for t in filter_out_tenors if t in rates_df.index], errors='ignore')
        rates_chg_df_filtered = rates_chg_df.drop(index=[t for t in filter_out_tenors if t in rates_chg_df.index],
                                                  errors='ignore')

        # Plot 1: Rates
        ax1 = axs[0, 0]  #
        ax1.set_title('Rates', fontdict={'fontsize': 9})
        # Invert y-axis for heatmap as in original (e.g., `[::-1]`)
        sns.heatmap(
            rates_df_filtered[::-1], cmap='coolwarm', linewidths=1, annot=True,
            fmt=".5g", cbar=False, ax=ax1, mask=rates_df_filtered[::-1].isnull()
        )

        # Plot 2: Changes in Rates
        ax2 = axs[0, 1]  #
        ax2.set_title(
            f'Chg: {change_offset_key} | {ql_to_datetime(ql.Settings.instance().evaluationDate).strftime("%Y-%m-%d")}',
            fontdict={'fontsize': 9})
        sns.heatmap(
            rates_chg_df_filtered[::-1], cmap='Purples_r', linewidths=1, annot=True,
            fmt=".5g", cbar=False, ax=ax2, mask=rates_chg_df_filtered[::-1].isnull()
        )

        # Plot 3: Forwards (or other curves)
        ax3 = axs[1, 0]  #
        ax3.set_title('Forwards', fontdict={'fontsize': 9})
        sns.heatmap(
            curves_df[::-1], cmap='coolwarm', linewidths=1, annot=True,
            fmt=".5g", cbar=False, ax=ax3  #
        )

        # Plot 4: Changes in Forwards
        ax4 = axs[1, 1]  #
        ax4.set_title(
            f'Chg: {change_offset_key} | {ql_to_datetime(ql.Settings.instance().evaluationDate).strftime("%Y-%m-%d")}',
            fontdict={'fontsize': 9})
        sns.heatmap(
            chg_df[::-1], cmap='Purples_r', linewidths=1, annot=True,
            fmt=".5g", cbar=False, ax=ax4  #
        )

        plt.tight_layout()
        return fig