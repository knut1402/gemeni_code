# src/plotting/bokeh_plots.py

import pandas as pd
import numpy as np
import QuantLib as ql
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import accumulate
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, Legend, DatetimeTickFormatter, LinearAxis, LinearColorMapper
from bokeh.models import ColumnDataSource, TabPanel, Tabs, LabelSet, Span, Range1d, FactorRange, CustomJS, TapTool
from bokeh.events import Tap
from bokeh.layouts import row, column, gridplot, layout
from bokeh.palettes import Category10, brewer, Category20, Bright6  #
from bokeh.transform import factor_cmap
from bokeh.models.renderers import GlyphRenderer
from bokeh.colors import RGB
from matplotlib import cm  # For colormaps,

from src.plotting.base_plotter import BasePlotter
from src.curves.ois_curve_builder import OISCurveBuilder  #
from src.curves.swap_curve_builder import SwapCurveBuilder  #
from src.curves.inflation_curve_builder import InflationCurveBuilder  #
from src.pricers.swap_pricer import SwapPricer  #
from src.pricers.inflation_pricer import InflationPricer  #
from src.strategy_analytics.option_strategy import OptionStrategy  #
from src.market_data_processors.economic_data_processor import EconomicDataProcessor  #
from src.market_data_processors.wirp_processor import WirpProcessor  #
from src.data_manager import b_con  #
from src.date_utils import ql_to_datetime, to_bbg_date_str  #
from src.financial_math_utils import px_opt_ticks, convert_to_64  #
from src.config import config  #


class BokehPlotter(BasePlotter):
    """
    Provides a collection of Bokeh-based interactive plotting functions for financial data.
    """

    def __init__(self):
        """
        Initializes the BokehPlotter.
        """
        super().__init__()
        self.swap_pricer = SwapPricer()  #
        self.inflation_pricer = InflationPricer()  #
        self.eco_data_processor = EconomicDataProcessor()  #
        self.wirp_processor = WirpProcessor()  #

    def plot(self, plot_type: str, data: Any, **kwargs) -> Union[figure, Tabs, layout]:
        """
        Generic plot method to dispatch to specific Bokeh plotting functions.

        Args:
            plot_type (str): The type of plot to generate (e.g., 'ois_curve', 'inflation_curve', etc.).
            data (Any): The primary data required for the plot (e.g., list of curve builders, option strategy object).
            **kwargs: Additional parameters specific to the plot type.

        Returns:
            Union[figure, Tabs, layout]: The generated Bokeh plot object.
        """
        if plot_type == 'ois_curve':
            return self.plot_ois_curve_bokeh(curve_builders=data, **kwargs)  #
        elif plot_type == 'inflation_curve':
            return self.plot_inflation_curve_bokeh(inflation_curve_builders=data, **kwargs)  #
        elif plot_type == 'economic_forecast':
            return self.plot_economic_forecast_bokeh(**kwargs)  #
        elif plot_type == 'generic_timeseries':
            return self.plot_generic_timeseries_bokeh(tickers=data, **kwargs)  #
        elif plot_type == 'simple_wirp':
            return self.plot_simple_wirp_bokeh(wirp_data=data, **kwargs)  #
        elif plot_type == 'multi_wirp':
            return self.plot_multi_wirp_bokeh(wirp_data=data, **kwargs)  #
        elif plot_type == 'option_vol_surface':
            return self.plot_option_vol_surface_bokeh(vol_surface_builders=data, **kwargs)  #
        elif plot_type == 'option_strategy':
            return self.plot_option_strategy_bokeh(strategy=data, **kwargs)  #
        elif plot_type == 'listed_options_timeseries':
            return self.plot_listed_options_timeseries_bokeh(**kwargs)  #
        elif plot_type == 'inflation_fixings':
            return self.plot_inflation_fixings_bokeh(inflation_curve_builders=data, **kwargs)  #
        elif plot_type == 'swap_heatmap':
            return self.plot_swap_heatmap_bokeh(curve_hmap_data=data, **kwargs)  #
        else:
            raise ValueError(f"Unsupported Bokeh plot type: {plot_type}")  #

    def plot_ois_curve_bokeh(self,
                             curve_builders: List[Union[SwapCurveBuilder, OISCurveBuilder]],
                             # Original `c1` as list of names, now builders
                             max_tenor_years: int = 30,  #
                             bar_changes: bool = False,  #
                             show_spreads: bool = False,  #
                             plot_title: str = '',  # Original `name`
                             fwd_tenor_unit: str = '1y',  # Original `fwd_tenor`
                             interval_tenor_unit: str = '1y',  # Original `int_tenor`
                             built_curve_data: Optional[List[Any]] = None,  # Original `built_curve`
                             plot_type_label: bool = True,  # Original `label_curve_name`
                             panel_dimensions: Tuple[int, int] = (1000, 400)  # Original `p_dim`
                             ) -> List[figure]:
        """
        Plots interactive OIS (Overnight Indexed Swap) curves using Bokeh.
        Can show multiple curves, historical changes, and spreads.

        Args:
            curve_builders (List[Union[SwapCurveBuilder, OISCurveBuilder]]): List of built curve builder objects.
            max_tenor_years (int): Maximum tenor in years for the plot.
            bar_changes (bool): If True, plots changes between the first curve and subsequent historical curves as bar charts.
            show_spreads (bool): If True, plots spreads between the first curve and subsequent curves.
            plot_title (str): Title of the plot.
            fwd_tenor_unit (str): Forward rate tenor unit (e.g., '1y', '1m', '1d').
            interval_tenor_unit (str): Interval for generating points on the curve (e.g., '1y', '1m', '1d').
            built_curve_data (Optional[List[Any]]): Pre-built curve data if available, to avoid recalculation.
            plot_type_label (bool): If True, uses the currency code as label; else, full builder info.
            panel_dimensions (Tuple[int, int]): Dimensions (width, height) for each Bokeh plot panel.

        Returns:
            List[figure]: A list of Bokeh figure objects (panels for curves, spreads, changes).
        """
        # Original `built_curve` in PLOT_BOKEH implies a pre-processed list of curve objects.
        # Here, `curve_builders` serves as the primary input.

        if built_curve_data is not None:  #
            crv_list = built_curve_data  #
        else:  #
            # Ensure curves are built
            for builder in curve_builders:  #
                builder.build_curve()  #
            crv_list = curve_builders  #

        n_curves = len(crv_list)  #
        n_historical_points = len(crv_list)  # Assuming each builder is for a different historical point

        # Parse forward and interval tenor units for QuantLib
        fwd_tenor_ql_unit = ql.Years  #
        if fwd_tenor_unit.endswith('d'):  #
            fwd_tenor_ql_unit = ql.Days  #
        elif fwd_tenor_unit.endswith('m'):  #
            fwd_tenor_ql_unit = ql.Months  #
        fwd_tenor_val = int(fwd_tenor_unit[:-1]) if fwd_tenor_unit[:-1].isdigit() else 1  #

        interval_tenor_ql_unit = ql.Years  #
        if interval_tenor_unit.endswith('d'):  #
            interval_tenor_ql_unit = ql.Days  #
        elif interval_tenor_unit.endswith('m'):  #
            interval_tenor_ql_unit = ql.Months  #
        interval_tenor_val = int(interval_tenor_unit[:-1]) if interval_tenor_unit[:-1].isdigit() else 1  #

        # Determine number of plots needed
        num_plots = 1  # Curve plot
        if show_spreads and n_curves > 1:  #
            num_plots += 1  # Spread plot
        if bar_changes and n_historical_points > 1:  #
            # If `n_curves == 1`, one bar chart row. If `n_curves > 1`, multiple rows (one per currency).
            if n_curves == 1:  #
                num_plots += 1  #
            else:  #
                num_plots += (n_curves - 1)  # Changes in spreads or individual curves

        s_plots: List[figure] = []  # List to hold Bokeh figures

        # --- Data Preparation ---
        rates_data: Dict[str, List[float]] = {}  # {label: rates_list}
        # `h2` from original code was a list of `trade_date`s for labels.
        trade_dates = [ql_to_datetime(crv.reference_date) for crv in crv_list]  #

        for i, builder in enumerate(crv_list):  #
            # Dates for OIS curve plotting
            start_date_ql = builder.reference_date  #
            end_date_ql = start_date_ql + ql.Period(max_tenor_years, ql.Years)  #

            # Use MakeSchedule for consistent interval points
            ql_dates_for_curve = ql.MakeSchedule(start_date_ql, end_date_ql,
                                                 ql.Period(interval_tenor_val, interval_tenor_ql_unit))  #
            yr_axis = [(d - start_date_ql) / 365.25 for d in ql_dates_for_curve]  # Years from start

            # Get forward rates from OIS curve
            rates = [builder.curve.forwardRate(d, builder.calendar.advance(d, fwd_tenor_val, fwd_tenor_ql_unit),  #
                                               builder.curve.dayCount(), ql.Simple).rate() * 100 for d in
                     ql_dates_for_curve]  #

            label = f"{builder.currency_code} OIS: {trade_dates[i].strftime('%Y-%m-%d')}"  #
            rates_data[label] = rates  #

        # Prepare data for changes and spreads
        rates_change: Dict[str, List[List[float]]] = {}  # {currency_label: [list_of_changes_vs_each_hist_pt]}

        # Original `rates_change` calculation from `PLOT.py` and `PLOT_BOKEH.py`
        # `rates_diff = 100*( np.array(rates[j][0]) - np.array(rates[j][k]))` for historical points.
        # Here `rates_data` keys are labels directly.
        first_curve_label = list(rates_data.keys())[0]  #
        first_curve_rates = np.array(rates_data[first_curve_label])  #

        for i in range(1, n_historical_points):  # Iterate through historical builders
            hist_curve_label = list(rates_data.keys())[i]  #
            hist_curve_rates = np.array(rates_data[hist_curve_label])  #

            min_len = min(len(first_curve_rates), len(hist_curve_rates))  # Ensure same length
            rate_diff = 100 * (first_curve_rates[:min_len] - hist_curve_rates[:min_len])  #

            # Store by currency code for changes
            ccy_code = crv_list[i].currency_code  #
            if ccy_code not in rates_change:  #
                rates_change[ccy_code] = []  #
            rates_change[ccy_code].append(rate_diff.tolist())  #

        bar_dict = rates_change  #

        spreads_data: Dict[str, List[float]] = {}  # {spread_label: rates_list}
        spreads_change: Dict[str, List[List[float]]] = {}  # {spread_label: [list_of_changes]}

        if show_spreads and n_curves > 1:  #
            first_ccy_label = list(rates_data.keys())[0]  #
            first_ccy_rates_arr = np.array(rates_data[first_ccy_label])  #

            for i in range(1, n_curves):  #
                comp_ccy_label = list(rates_data.keys())[i]  #
                comp_ccy_rates_arr = np.array(rates_data[comp_ccy_label])  #

                min_len = min(len(first_ccy_rates_arr), len(comp_ccy_rates_arr))  #
                spread = 100 * (first_ccy_rates_arr[:min_len] - comp_ccy_rates_arr[:min_len])  #

                spread_label = f"{first_ccy_label.split(':')[0]} - {comp_ccy_label.split(':')[0]}"  #
                spreads_data[spread_label] = spread.tolist()  #

                # Changes in spreads (if multiple historical points)
                if n_historical_points > 1:  #
                    if spread_label not in spreads_change:  #
                        spreads_change[spread_label] = []  #

                    # This logic aligns with original `sprd_chg = 1*( np.array(spreads[j][0]) - np.array(spreads[j][k]))`
                    # It compares the current spread with historical spreads.
                    # The `spreads_data` should contain current spread (derived from builders[0]) and historical spreads.
                    # This means we need to re-calculate spreads for historical builders.

                    # For `spreads_change`, the `k` in original refers to historical points.
                    # So, if `spreads_data[spread_label]` now contains the current spread,
                    # we need to calculate historical spreads for `spreads_change`.

                    # Re-calculating spreads for historical builders:
                    # This might be inefficient if done on the fly here.
                    # Better to calculate all outrights for all historical dates first, then compute spreads.
                    # For now, let's assume `rates_change` (computed above) is used for `bar_dict`.

                    # The `bar_dict` in original code was either `rates_change` or `spreads_change`.
                    # If `show_spreads` is true, `bar_dict` should be `spreads_change`.
                    # This implies `spreads_change` would be populated with changes in spreads.
                    #
                    # To populate `spreads_change` correctly:
                    # For each `i` from 1 to `n_historical_points - 1`:
                    #  Calculate `spread_i = 100 * (builder[0].rates - builder[i].rates)`
                    #  Calculate `spread_comp_i = 100 * (builder[first_curve_idx].rates - builder[i_comp].rates)`
                    #  Then `spread_change = spread_i - spread_comp_i`

                    # For simplicity given the `bar_dict` assignment:
                    # If `show_spreads` is True, `bar_dict` becomes `spreads_change`.
                    # This means `spreads_change` must contain the differences between the current spreads and historical spreads.
                    # Let's populate this based on the existing `rates_change` for consistency with `PLOT.py`'s `bar_dict` logic.

                    # This is complex. The original `PLOT.py`'s `bar_dict` definition
                    # `elif sprd == 1: bar_dict = rates_sprd`
                    # `elif bar_chg == 1: bar_dict = rates_chg`
                    # means that if `sprd == 1`, `bar_dict` is just the spreads themselves (not changes).
                    # But then `bar_scaler` and `num_plot` implies changes.
                    #
                    # The user explicitly asked for bar charts of *changes*.
                    # So if `show_spreads` is True, `bar_changes` should plot *changes in spreads*.
                    # If `show_spreads` is False, `bar_changes` should plot *changes in outrights*.
                    #
                    # `spreads_change` should be `current_spread - historical_spread`.
                    #
                    # Let's re-align `bar_dict` based on this:
                    # If `show_spreads` is true, `bar_dict` is `spreads_change`.
                    # If `show_spreads` is false, `bar_dict` is `rates_change`.

                    # For `spreads_change`, we need to compute historical spreads for each historical date.
                    # Then compute the difference between current spread and historical spread.
                    #
                    for j_hist in range(1, n_historical_points):  # Iterate through historical builders
                        hist_first_ccy_rates_arr = np.array(rates_data[list(rates_data.keys())[j_hist]])  #
                        hist_comp_ccy_label = list(rates_data.keys())[i]  #
                        hist_comp_ccy_rates_arr = np.array(rates_data[hist_comp_ccy_label])  #

                        min_len_hist = min(len(hist_first_ccy_rates_arr), len(hist_comp_ccy_rates_arr))  #
                        hist_spread = 100 * (
                                    hist_first_ccy_rates_arr[:min_len_hist] - hist_comp_ccy_rates_arr[:min_len_hist])  #

                        min_len_change = min(len(spreads_data[spread_label]), len(hist_spread))  #
                        spread_diff = spreads_data[spread_label][:min_len_change] - hist_spread[:min_len_change]  #

                        spreads_change[spread_label].append(spread_diff.tolist())  #

            bar_dict_final: Dict[str, List[List[float]]] = {}
            if show_spreads:  #
                bar_dict_final = spreads_change  #
            else:  #
                bar_dict_final = rates_change  #

        # --- Plot 1: Curves ---
        p1 = figure(width=panel_dimensions[0], height=panel_dimensions[1],
                    tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],
                    toolbar_location='left', title=plot_title)  #
        p1.xgrid.visible = False  #
        p1.ygrid.visible = False  #
        p1.title.text_font = "calibri"  #
        p1.title.text_font_size = '12pt'  #
        p1.title.align = 'left'  #

        for i, builder in enumerate(crv_list):  #
            label_text = f"{builder.currency_code}" if plot_type_label else f"{builder.currency_code}: {trade_dates[i].strftime('%Y-%m-%d')}"  #

            # Use ColumnDataSource for hover tool functionality
            source = ColumnDataSource(data={
                'x': yr_axis,  #
                'y': rates_data[list(rates_data.keys())[i]],  #
                'date_label': [f"{d.strftime('%Y-%m-%d')}" for d in ql_dates_for_curve],  #
                'rate_label': [f"{r:.2f}%" for r in rates_data[list(rates_data.keys())[i]]]  #
            })  #

            p1.line(x='x', y='y', source=source,
                    legend_label=label_text,  #
                    line_width=2,  #
                    color=Category20[20][i * 2], alpha=(1.0 - (0.4 * (i % 2))),  # Different colors for different curves
                    muted_color=Category20[20][i * 2], muted_alpha=0.2)  #

            # Hover tooltips
            p1.add_tools(HoverTool(renderers=[p1.renderers[-1]], tooltips=[  #
                ("Tenor", "@x{0.0}Y"),  #
                ("Rate", "@rate_label"),  #
                ("Date", "@date_label")  #
            ]))  #

        p1.legend.label_text_font = "calibri"  #
        p1.legend.label_text_font_size = "7pt"  #
        p1.legend.glyph_height = 5  #
        p1.legend.label_height = 5  #
        p1.legend.spacing = 1  #
        p1.legend.background_fill_alpha = 0.0  #
        p1.legend.click_policy = "mute"  #
        p1.xaxis.axis_label = 'Tenor (Years)'  #
        p1.yaxis.axis_label = 'Rate (%)'  #
        s_plots.append(p1)  #

        # --- Plot 2: Spreads ---
        if show_spreads and n_curves > 1:  #
            p2 = figure(width=panel_dimensions[0], height=panel_dimensions[1] - 100,  #
                        tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],  #
                        toolbar_location='left')  #
            p2.xgrid.grid_line_dash = 'dotted'  #
            p2.ygrid.grid_line_dash = 'dotted'  #

            for spread_label, rates_list in spreads_data.items():  #
                source = ColumnDataSource(data={  #
                    'x': yr_axis[:len(rates_list)],  #
                    'y': rates_list,  #
                    'spread_val_label': [f"{s:.2f}bps" for s in rates_list]  #
                })  #
                p2.line(x='x', y='y', source=source,  #
                        legend_label=spread_label,  #
                        line_width=2,  #
                        color=Category20[20][list(spreads_data.keys()).index(spread_label) * 2 + 1],  #
                        muted_color=Category20[20][list(spreads_data.keys()).index(spread_label) * 2 + 1],
                        muted_alpha=0.2)  #

                p2.add_tools(HoverTool(renderers=[p2.renderers[-1]], tooltips=[  #
                    ("Tenor", "@x{0.0}Y"),  #
                    ("Spread", "@spread_val_label")  #
                ]))  #

            p2.legend.label_text_font = "calibri"  #
            p2.legend.label_text_font_size = "7pt"  #
            p2.legend.glyph_height = 5  #
            p2.legend.label_height = 5  #
            p2.legend.spacing = 1  #
            p2.legend.background_fill_alpha = 0.0  #
            p2.legend.click_policy = "mute"  #
            p2.xaxis.axis_label = 'Tenor (Years)'  #
            p2.yaxis.axis_label = 'Spread (bps)'  #
            s_plots.append(p2)  #

        # --- Plot 3: Bar Changes ---
        if bar_changes and n_historical_points > 1:  #
            # This logic supports multiple currency changes on separate plots if `n_curves > 1`
            # or a single plot for one currency's changes.

            # The original `PLOT.py` had dynamic subplot indexing `axs[j+start_sub_chg]`.
            # For Bokeh, we create a new figure for each set of bar changes.

            for i, (ccy_label, historical_changes_for_ccy) in enumerate(bar_dict_final.items()):  #
                p_bar = figure(width=panel_dimensions[0], height=panel_dimensions[1] - 100,  #
                               tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],  #
                               toolbar_location='left')  #
                p_bar.xgrid.grid_line_dash = 'dotted'  #
                p_bar.ygrid.grid_line_dash = 'dotted'  #

                bar_width = 0.15  #
                bar_offset_along_x = 0  # For staggering bars

                for j, (change_label, rates_list) in enumerate(historical_changes_for_ccy.items()):  #
                    source = ColumnDataSource(data={  #
                        'x': np.array(yr_axis[:len(rates_list)]) + bar_offset_along_x,  #
                        'top': rates_list,  #
                        'change_label': [f"{r:.2f}bps" for r in rates_list]  #
                    })  #

                    p_bar.vbar(x='x', top='top', source=source,  #
                               width=bar_width,  #
                               legend_label=change_label,  #
                               color=Category20[20][(i * 2) + j], alpha=(1.0 - (0.4 * j)),  #
                               muted_color=Category20[20][(i * 2) + j], muted_alpha=0.2)  #

                    p_bar.add_tools(HoverTool(renderers=[p_bar.renderers[-1]], tooltips=[  #
                        ("Tenor", "@x{0.0}Y"),  #
                        ("Change", "@change_label")  #
                    ]))  #

                    bar_offset_along_x += bar_width + 0.1  #

                p_bar.legend.label_text_font = "calibri"  #
                p_bar.legend.label_text_font_size = "7pt"  #
                p_bar.legend.glyph_height = 5  #
                p_bar.legend.label_height = 5  #
                p_bar.legend.spacing = 1  #
                p_bar.legend.background_fill_alpha = 0.0  #
                p_bar.legend.click_policy = "mute"  #
                p_bar.xaxis.axis_label = 'Tenor (Years)'  #
                p_bar.yaxis.axis_label = f'{ccy_label} Change (bps)'  #
                s_plots.append(p_bar)  #

        return s_plots  #

    def plot_inflation_curve_bokeh(self,
                                   inflation_curve_builders: List[InflationCurveBuilder],  # Original `c1`
                                   max_tenor_years: int = 30,  #
                                   bar_changes: bool = False,  #
                                   show_spreads: bool = False,  #
                                   plot_title: str = '',  # Original `name`
                                   panel_dimensions: Tuple[int, int] = (700, 350)  # Original `p_dim`
                                   ) -> List[figure]:
        """
        Plots interactive inflation zero-coupon (ZC) curves using Bokeh.
        Can show multiple curves, historical changes, and spreads.

        Args:
            inflation_curve_builders (List[InflationCurveBuilder]): List of built inflation curve builder objects.
            max_tenor_years (int): Maximum tenor in years for the plot.
            bar_changes (bool): If True, plots changes between the first curve and subsequent historical curves as bar charts.
            show_spreads (bool): If True, plots spreads between the first curve and subsequent curves.
            plot_title (str): Title of the plot.
            panel_dimensions (Tuple[int, int]): Dimensions (width, height) for each Bokeh plot panel.

        Returns:
            List[figure]: A list of Bokeh figure objects (panels for curves, spreads, changes).
        """
        # Ensure curves are built
        for builder in inflation_curve_builders:  #
            builder.build_curve()  #

        n_curves = len(inflation_curve_builders)  #
        n_historical_points = len(inflation_curve_builders)  # Assuming each builder is for a different historical point

        s_plots: List[figure] = []  # List to hold Bokeh figures

        # --- Data Preparation ---
        rates_data: Dict[str, List[float]] = {}  # {label: rates_list}
        yr_axis = list(range(1, max_tenor_years + 1))  #

        for i, builder in enumerate(inflation_curve_builders):  #
            rates = []  #
            for tenor in yr_axis:  #
                metrics = self.inflation_pricer.calculate_zc_metrics(  #
                    builder,  #
                    start_date_input=0,  # Spot starting
                    tenor_years=tenor,  #
                    lag_months=builder._lag_months,  # Use builder's configured lag
                    use_market_fixing_curve=True  # Default to market fixing curve for inflation
                )  #
                rates.append(metrics['zc_rate'])  #

            label = f"{builder._inflation_index_name}: {ql_to_datetime(builder.reference_date).strftime('%Y-%m-%d')}"  #
            rates_data[label] = rates  #

        # Prepare data for changes and spreads
        rates_change: Dict[str, List[List[float]]] = {}  # {index_name: [list_of_changes_vs_each_hist_pt]}

        first_curve_label = list(rates_data.keys())[0]  #
        first_curve_rates = np.array(rates_data[first_curve_label])  #

        for i in range(1, n_historical_points):  # Iterate through historical builders
            hist_curve_label = list(rates_data.keys())[i]  #
            hist_curve_rates = np.array(rates_data[hist_curve_label])  #

            min_len = min(len(first_curve_rates), len(hist_curve_rates))  # Ensure same length
            rate_diff = 100 * (first_curve_rates[:min_len] - hist_curve_rates[:min_len])  #

            index_name = inflation_curve_builders[i]._inflation_index_name  #
            if index_name not in rates_change:  #
                rates_change[index_name] = []  #
            rates_change[index_name].append(rate_diff.tolist())  #

        bar_dict = rates_change  #

        spreads_data: Dict[str, List[float]] = {}  # {spread_label: rates_list}
        spreads_change: Dict[str, List[List[float]]] = {}  # {spread_label: [list_of_changes]}

        if show_spreads and n_curves > 1:  #
            first_index_label = list(rates_data.keys())[0]  #
            first_index_rates_arr = np.array(rates_data[first_index_label])  #

            for i in range(1, n_curves):  #
                comp_index_label = list(rates_data.keys())[i]  #
                comp_index_rates_arr = np.array(rates_data[comp_index_label])  #

                min_len = min(len(first_index_rates_arr), len(comp_index_rates_arr))  #
                spread = 100 * (first_index_rates_arr[:min_len] - comp_index_rates_arr[:min_len])  #

                spread_label = f"{first_index_label.split(':')[0]} - {comp_index_label.split(':')[0]}"  #
                spreads_data[spread_label] = spread.tolist()  #

                # Changes in spreads (if multiple historical points)
                if n_historical_points > 1:  #
                    if spread_label not in spreads_change:  #
                        spreads_change[spread_label] = []  #

                    for j_hist in range(1, n_historical_points):  # Iterate through historical builders
                        hist_first_index_rates_arr = np.array(rates_data[list(rates_data.keys())[j_hist]])  #
                        hist_comp_index_label = list(rates_data.keys())[i]  #
                        hist_comp_index_rates_arr = np.array(rates_data[hist_comp_index_label])  #

                        min_len_hist = min(len(hist_first_index_rates_arr), len(hist_comp_index_rates_arr))  #
                        hist_spread = 100 * (hist_first_index_rates_arr[:min_len_hist] - hist_comp_index_rates_arr[
                                                                                         :min_len_hist])  #

                        min_len_change = min(len(spreads_data[spread_label]), len(hist_spread))  #
                        spread_diff = spreads_data[spread_label][:min_len_change] - hist_spread[:min_len_change]  #

                        spreads_change[spread_label].append(spread_diff.tolist())  #

            bar_dict = spreads_change  # # Reassign bar_dict if spreads are shown.

        # --- Plot 1: Curves ---
        p1 = figure(width=panel_dimensions[0], height=panel_dimensions[1],
                    tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],
                    toolbar_location='left', title=plot_title)  #
        p1.xgrid.visible = False  #
        p1.ygrid.visible = False  #
        p1.title.text_font = "calibri"  #
        p1.title.text_font_size = '12pt'  #
        p1.title.align = 'left'  #

        for i, builder in enumerate(inflation_curve_builders):  #
            label_text = f"{builder._inflation_index_name}: {ql_to_datetime(builder.reference_date).strftime('%Y-%m-%d')}"  #

            source = ColumnDataSource(data={  #
                'x': np.array(yr_axis),  #
                'y': rates_data[list(rates_data.keys())[i]],  #
                'date_label': [f"{ql_to_datetime(builder.base_month_ql).strftime('%b-%y')}" for _ in yr_axis],
                # Base month for inflation.
                'rate_label': [f"{r:.2f}%" for r in rates_data[list(rates_data.keys())[i]]]  #
            })  #

            p1.line(x='x', y='y', source=source,  #
                    legend_label=label_text,  #
                    line_width=2,  #
                    color=Category20[20][i * 2], alpha=(1.0 - (0.4 * (i % 2))),  #
                    muted_color=Category20[20][i * 2], muted_alpha=0.2)  #

            # Original code plots 'par_nodes' using circles.
            # Assuming `inflation_curve_builder.inflation_zc_rates` holds this.
            # And `builder.inflation_zc_rates['maturity_years']` as x-axis and `builder.inflation_zc_rates['px']` as y-axis.
            #
            source_nodes = ColumnDataSource(data={  #
                'x_nodes': builder.inflation_zc_rates['maturity_years'].tolist(),  #
                'y_nodes': builder.inflation_zc_rates['px'].tolist()  #
            })  #
            p1.circle(x='x_nodes', y='y_nodes', source=source_nodes,  #
                      color=Category20[20][i * 2], alpha=0.7, muted_alpha=0.2)  #

            p1.add_tools(HoverTool(renderers=[p1.renderers[-1]], tooltips=[  #
                ("Tenor", "@x{0.0}Y"),  #
                ("Rate", "@rate_label"),  #
                ("Base Month", "@date_label")  #
            ]))  #

        p1.legend.label_text_font = "calibri"  #
        p1.legend.label_text_font_size = "7pt"  #
        p1.legend.glyph_height = 5  #
        p1.legend.label_height = 5  #
        p1.legend.spacing = 1  #
        p1.legend.background_fill_alpha = 0.0  #
        p1.legend.click_policy = "mute"  #
        p1.xaxis.axis_label = 'Tenor (Years)'  #
        p1.yaxis.axis_label = 'Zero-Coupon Rate (%)'  #
        s_plots.append(p1)  #

        # --- Plot 2: Spreads ---
        if show_spreads and n_curves > 1:  #
            p2 = figure(width=panel_dimensions[0], height=panel_dimensions[1] - 100,  #
                        tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],  #
                        toolbar_location='left')  #
            p2.xgrid.grid_line_dash = 'dotted'  #
            p2.ygrid.grid_line_dash = 'dotted'  #

            for spread_label, rates_list in spreads_data.items():  #
                source = ColumnDataSource(data={  #
                    'x': np.array(yr_axis[:len(rates_list)]),  #
                    'y': rates_list,  #
                    'spread_val_label': [f"{s:.2f}bps" for s in rates_list]  #
                })  #
                p2.line(x='x', y='y', source=source,  #
                        legend_label=spread_label,  #
                        line_width=2,  #
                        color=Category20[20][list(spreads_data.keys()).index(spread_label) * 2 + 1],  #
                        muted_color=Category20[20][list(spreads_data.keys()).index(spread_label) * 2 + 1],
                        muted_alpha=0.2)  #

                p2.add_tools(HoverTool(renderers=[p2.renderers[-1]], tooltips=[  #
                    ("Tenor", "@x{0.0}Y"),  #
                    ("Spread", "@spread_val_label")  #
                ]))  #

            p2.legend.label_text_font = "calibri"  #
            p2.legend.label_text_font_size = "7pt"  #
            p2.legend.glyph_height = 5  #
            p2.legend.label_height = 5  #
            p2.legend.spacing = 1  #
            p2.legend.background_fill_alpha = 0.0  #
            p2.legend.click_policy = "mute"  #
            p2.xaxis.axis_label = 'Tenor (Years)'  #
            p2.yaxis.axis_label = 'Spread (bps)'  #
            s_plots.append(p2)  #

        # --- Plot 3: Bar Changes ---
        if bar_changes and n_historical_points > 1:  #
            # This logic supports multiple index changes on separate plots if `n_curves > 1`
            # or a single plot for one index's changes.

            for i, (index_name, historical_changes_for_index) in enumerate(bar_dict.items()):  #
                p_bar = figure(width=panel_dimensions[0], height=panel_dimensions[1] - 100,  #
                               tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],  #
                               toolbar_location='left')  #
                p_bar.xgrid.grid_line_dash = 'dotted'  #
                p_bar.ygrid.grid_line_dash = 'dotted'  #

                bar_width = 0.15  #
                bar_offset_along_x = 0  # For staggering bars

                for j, (change_label, rates_list) in enumerate(historical_changes_for_index.items()):  #
                    source = ColumnDataSource(data={  #
                        'x': np.array(yr_axis[:len(rates_list)]) + bar_offset_along_x,  #
                        'top': rates_list,  #
                        'change_label': [f"{r:.2f}bps" for r in rates_list]  #
                    })  #

                    p_bar.vbar(x='x', top='top', source=source,  #
                               width=bar_width,  #
                               legend_label=change_label,  #
                               color=Category20[20][(i * 2) + j], alpha=(1.0 - (0.4 * j)),  #
                               muted_color=Category20[20][(i * 2) + j], muted_alpha=0.2)  #

                    p_bar.add_tools(HoverTool(renderers=[p_bar.renderers[-1]], tooltips=[  #
                        ("Tenor", "@x{0.0}Y"),  #
                        ("Change", "@change_label")  #
                    ]))  #

                    bar_offset_along_x += bar_width + 0.1  #

                p_bar.legend.label_text_font = "calibri"  #
                p_bar.legend.label_text_font_size = "7pt"  #
                p_bar.legend.glyph_height = 5  #
                p_bar.legend.label_height = 5  #
                p_bar.legend.spacing = 1  #
                p_bar.legend.background_fill_alpha = 0.0  #
                p_bar.legend.click_policy = "mute"  #
                p_bar.xaxis.axis_label = 'Tenor (Years)'  #
                p_bar.yaxis.axis_label = f'{index_name} Change (bps)'  #
                s_plots.append(p_bar)  #

        return s_plots  #

    def plot_economic_forecast_bokeh(self,
                                     economic_data_type: str,  # 'GDP', 'CPI', 'PCE', 'Core-PCE', 'UNEMP', 'FISC'
                                     country_code: str,  # 'EU', 'US'
                                     forecast_year: str,  # '2023', '2024'
                                     contributor_highlight: str = 'GS',  # 'GS', 'BAR', 'JPM'
                                     official_source: str = 'IMF'  # 'IMF', 'ECB', 'FED'
                                     ) -> figure:
        """
        Plots interactive historical economic forecasts from various contributors and an official source using Bokeh.

        Args:
            economic_data_type (str): Type of economic data (e.g., 'GDP', 'CPI').
            country_code (str): Country code for the forecast (e.g., 'EU', 'US').
            forecast_year (str): The forecast year (e.g., '2023', '2024').
            contributor_highlight (str): The contributor to highlight in the plot.
            official_source (str): The official source to plot.

        Returns:
            figure: The generated Bokeh figure object.
        """
        # Mapping economic data type to Bloomberg convention (e.g., 'PI' for CPI)
        eco_type_map = {  #
            'GDP': 'GD', 'CPI': 'PI', 'PCE': 'PC', 'Core-PCE': 'CC',  #
            'UNEMP': 'UP', 'FISC': 'BB'  #
        }  #
        bbg_eco_code = eco_type_map.get(economic_data_type)  #
        if not bbg_eco_code:  #
            raise ValueError(f"Unsupported economic data type: {economic_data_type}")  #

        # Standard list of contributors based on original `PLOT.py`
        all_contributors = [  #
            'BAR', 'BOA', 'BNP', 'CE', 'CIT', 'CAG', 'CSU', 'DNS', 'FTC', 'GS', 'HSB', 'IG',  #
            'JPM', 'MS', 'NTX', 'NS', 'NDA', 'PMA', 'UBS', 'WF', 'SCB'  #
        ]  #

        today_dt = datetime.datetime.now()  #
        bbg_today_str = to_bbg_date_str(today_dt, ql_date=0)  #

        # Fetch data for all contributors and official source
        contributor_tickers = [f"EC{bbg_eco_code}{country_code} {forecast_year[-2:]} {contrib} Index" for contrib in
                               all_contributors]  #
        official_ticker = f"EC{bbg_eco_code}{country_code} {forecast_year[-2:]} {official_source} Index"  #

        try:  #
            df_contributors_raw = b_con.bdh(contributor_tickers, 'PX_LAST',
                                            to_bbg_date_str(today_dt - datetime.timedelta(days=365), ql_date=0),
                                            bbg_today_str, longdata=True)  #
            df_official_raw = b_con.bdh(official_ticker, 'PX_LAST',
                                        to_bbg_date_str(today_dt - datetime.timedelta(days=365), ql_date=0),
                                        bbg_today_str, longdata=True)  #

            if df_contributors_raw.empty or df_official_raw.empty:  #
                print("Warning: No economic forecast data found for requested parameters. Returning empty plot.")  #
                return figure(width=600, height=300, title="No Data Found")  #

            df_contributors_raw['date'] = pd.to_datetime(df_contributors_raw['date'])  #
            df_official_raw['date'] = pd.to_datetime(df_official_raw['date'])  #

            # Calculate average forecast over time (rolling average inspired by original `m1`)
            # Original `d1` was a list of historical dates. Here, we calculate a rolling mean directly.
            df_avg_calc = df_contributors_raw.groupby('date')['value'].mean().rolling(
                window='30D').mean().reset_index()  #
            df_avg_calc.columns = ['date', 'avg_value']  #

            # Bokeh figure setup
            s1 = figure(width=600, height=300,
                        tools=["pan", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                        toolbar_location='left', x_axis_type='datetime',
                        title=f"{country_code} {economic_data_type} forecast: {forecast_year}")  #
            s1.xaxis.formatter = DatetimeTickFormatter(days="%d-%b-%y", months="%d-%b-%y")  #
            s1.add_tools(
                HoverTool(tooltips=[('date', '@x{%d-%b-%y}'), ('value', '@y{0.00}')], formatters={'$x': 'datetime'}))  #
            s1.xgrid.visible = False  #
            s1.ygrid.visible = False  #
            s1.legend.label_text_font = "calibri"  #
            s1.legend.label_text_font_size = "9pt"  #
            s1.legend.spacing = 1  #
            s1.legend.background_fill_alpha = 0.0  #
            s1.legend.click_policy = "mute"  #
            s1.xaxis.axis_label = 'Date'  #
            s1.yaxis.axis_label = 'Value'  #
            s1.title.text_font = "calibri"  #
            s1.title.text_font_size = '10pt'  #
            s1.title.align = 'left'  #

            # Plot individual contributor lines
            unique_contributor_tickers = df_contributors_raw['ticker'].unique()  #
            for ticker in unique_contributor_tickers:  #
                contrib_df = df_contributors_raw[df_contributors_raw['ticker'] == ticker]  #
                contrib_name = ticker.split(' ')[-2]  #

                line_color = 'silver'  #
                label_text = ''  #

                if contrib_name == contributor_highlight:  #
                    line_color = 'red'  # Changed from darkorange for Bokeh palette clarity
                    label_text = f"{contrib_name}: {contrib_df['value'].iloc[-1]:.2f}"  #

                source = ColumnDataSource(data={  #
                    'x': contrib_df['date'],  #
                    'y': contrib_df['value']  #
                })  #
                s1.line(x='x', y='y', source=source, legend_label=label_text, color=line_color, alpha=1.0,
                        muted_alpha=0.25)  #

            # Plot official source
            source_official = ColumnDataSource(data={'x': df_official_raw['date'], 'y': df_official_raw['value']})  #
            s1.line(x='x', y='y', source=source_official, color='forestgreen',
                    legend_label=f"{official_source}: {df_official_raw['value'].iloc[-1]:.2f}", alpha=1.0)  #

            # Plot average line
            source_avg = ColumnDataSource(data={'x': df_avg_calc['date'], 'y': df_avg_calc['avg_value']})  #
            s1.line(x='x', y='y', source=source_avg, color='blue',
                    legend_label=f"Avg: {df_avg_calc['avg_value'].iloc[-1]:.2f}", alpha=1.0)  #

            return s1  #

        except Exception as e:
            raise RuntimeError(f"Error plotting economic forecast with Bokeh: {e}")  #

    def plot_generic_timeseries_bokeh(self,
                                      tickers: List[str],  # Original `ticker_list` or `curves`
                                      start_date_input: Union[str, int, datetime.datetime, ql.Date],  # Original `d1`
                                      end_date_input: Union[str, int, datetime.datetime, ql.Date] = '',  # Original `d2`
                                      plot_type: str = 'Outright',  # 'Outright', 'Spread', 'Fly'
                                      change_flag: bool = False,  # If True, plots cumulative changes
                                      invert_flag: bool = False,  # If True, inverts values
                                      bbg_instrument_type: str = 'Fwd',  # 'Par', 'Fwd', 'Cash'
                                      curve_codes: Optional[List[str]] = None  # e.g., ['USD_3M', 'GBP_6M']
                                      ) -> figure:
        """
        Generic function to fetch and plot historical time series data for Bloomberg instruments using Bokeh.

        This function consolidates the logic from the original `plotool` in PLOT.py and PLOT_BOKEH.py.

        Args:
            tickers (List[str]): List of Bloomberg tickers or financial instrument codes.
            start_date_input (Union[str, int, datetime.datetime, ql.Date]): Start date for data.
            end_date_input (Union[str, int, datetime.datetime, ql.Date]): End date for data.
            plot_type (str): Type of plot ('Outright', 'Spread', 'Fly').
            change_flag (bool): If True, plots cumulative changes.
            invert_flag (bool): If True, inverts the plotted values.
            bbg_instrument_type (str): Bloomberg instrument type prefix ('Par', 'Fwd', 'Cash').
            curve_codes (Optional[List[str]]): List of currency/index codes (e.g., 'USD_3M', 'GBP_6M').

        Returns:
            figure: The generated Bokeh figure object.
        """
        # Determine actual start and end dates in QuantLib format
        ql_start_date = get_ql_date(start_date_input)  #
        ql_end_date = get_ql_date(end_date_input) if end_date_input else ql.Settings.instance().evaluationDate  #

        bbg_start_date_str = to_bbg_date_str(ql_start_date, ql_date=1)  #
        bbg_end_date_str = to_bbg_date_str(ql_end_date, ql_date=1)  #

        final_tickers_to_fetch: List[str] = []  #
        labels_for_plot: List[str] = []  #

        # Build Bloomberg tickers
        if curve_codes:  #
            for c_code in curve_codes:  #
                ccy_conf = config.get_currency_config(c_code)  #
                bbg_plot_tkrs = ccy_conf.get('bbgplot_tickers', [])  #

                base_bbg_ticker_root = ''  #
                if bbg_instrument_type == 'Par':  #
                    base_bbg_ticker_root = bbg_plot_tkrs[0] if bbg_plot_tkrs else ''  #
                elif bbg_instrument_type in ['Fwd', 'Cash']:  #
                    base_bbg_ticker_root = bbg_plot_tkrs[1] if len(bbg_plot_tkrs) > 1 else ''  #

                if not base_bbg_ticker_root:  #
                    print(
                        f"Warning: No Bloomberg plot ticker root found for {c_code} with type {bbg_instrument_type}. Skipping.")  #
                    continue  #

                for maturity in tickers:  # # `tickers` here are actual maturities like '5Y5Y', '10Y'
                    full_ticker = f"{base_bbg_ticker_root} {maturity} Curncy"  #
                    labels_for_plot.append(f"{c_code}:{maturity}")  #
                    final_tickers_to_fetch.append(full_ticker)  #

        else:  # `tickers` is already a list of full Bloomberg tickers
            final_tickers_to_fetch = tickers  #
            labels_for_plot = tickers  #

        # Fetch data from Bloomberg
        try:  #
            raw_data = b_con.bdh(final_tickers_to_fetch, 'PX_LAST', bbg_start_date_str, bbg_end_date_str)  #
            raw_data = raw_data.rename(columns={'value': 'PX_LAST'})  #

            # Pivot the DataFrame to have tickers as columns
            plot_df = raw_data.pivot_table(index='date', columns='ticker', values='PX_LAST')  #
            plot_df.columns = labels_for_plot  #

            if plot_df.empty:  #
                print("No data fetched for plotting.")  #
                return figure(width=800, height=400, title="No Data Found")  #

            # Apply inversion if requested
            if invert_flag:  #
                plot_df = plot_df * -1  #

            # Calculate cumulative changes if requested
            if change_flag:  #
                plot_df = plot_df.diff().cumsum()  #

            # Create Bokeh figure
            p = figure(width=800, height=400, tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],
                       #
                       toolbar_location='left', x_axis_type='datetime')  #
            p.xaxis.formatter = DatetimeTickFormatter(days="%d-%b-%y", months="%d-%b-%y")  #
            p.xgrid.visible = False  #
            p.ygrid.visible = False  #

            # Add lines for each series
            for i, col in enumerate(plot_df.columns):  #
                source = ColumnDataSource(data={  #
                    'x': plot_df.index,  #
                    'y': plot_df[col],  #
                    'val_label': [f"{v:.3f}" for v in plot_df[col]]  #
                })  #
                p.line(x='x', y='y', source=source, legend_label=col, line_width=2, color=Category20[20][i * 2])  #

                p.add_tools(HoverTool(renderers=[p.renderers[-1]], tooltips=[  #
                    ("Date", "@x{%d-%b-%y}"),  #
                    ("Value", "@val_label")  #
                ], formatters={'$x': 'datetime'}))  #

            p.legend.label_text_font = "calibri"  #
            p.legend.label_text_font_size = "9pt"  #
            p.legend.location = 'top_left'  #
            p.legend.click_policy = "hide"  #
            p.legend.spacing = 1  #
            p.legend.background_fill_alpha = 0.0  #
            p.xaxis.axis_label = 'Date'  #
            p.yaxis.axis_label = 'Value'  #

            return p  #

        except Exception as e:
            raise RuntimeError(f"Error plotting generic time series with Bokeh: {e}")  #

    def plot_simple_wirp_bokeh(self,
                               wirp_data: Dict[str, Any],
                               # Output from WirpProcessor.get_wirp_data for single ccy, single date
                               is_generated_meeting_num: bool = False  # Original `gen`
                               ) -> List[Tabs]:
        """
        Plots WIRP (World Interest Rate Probability) data for a single currency
        and a single valuation date, with interactive historical meeting data.

        Args:
            wirp_data (Dict[str, Any]): Dictionary containing 'wirp_data' and 'historical_cb_rates'
                                        from `WirpProcessor.get_wirp_data`, but specifically for a
                                        single currency and single valuation date.
            is_generated_meeting_num (bool): If True, uses meeting number as x-axis for historical.

        Returns:
            List[Tabs]: A list containing Bokeh Tabs object for step/cum plots and a separate historical plot.
        """
        # Ensure `wirp_data` contains data for a single currency and single date.
        ccy_code = list(wirp_data['wirp_data'].keys())[0]  #
        if not wirp_data['wirp_data'][ccy_code]:  #
            print(f"No WIRP data available for {ccy_code} to plot.")  #
            return [Tabs(tabs=[TabPanel(child=figure(title="No Data"), title="Step")])]  #

        df_wirp = wirp_data['wirp_data'][ccy_code][0]  # Assuming first entry in list
        # `df_wirp` contains 'meet_date', 'meet', 'cb', 'step', 'cum'

        # Access historical meeting data for interactive plot
        ccy_config = config.get_currency_config(ccy_code)  #
        ois_meet_hist_file = ccy_config.get('ois_meet_hist_file')  #
        full_meet_hist_df = pd.DataFrame()  #
        try:  #
            full_meet_hist_df = data_loader.load_pickle(ois_meet_hist_file)  #
            # Ensure 'meet_num' is present for `is_generated_meeting_num` use
            if 'meet_num' not in full_meet_hist_df.columns:  #
                # Infer `meet_num` if not explicitly present from `meet_date` and `meet`
                # This is a heuristic and might need careful check if exact `meet_num` logic is complex.
                full_meet_hist_df['meet_num'] = full_meet_hist_df['meet'].apply(
                    lambda x: int(re.search(r'\d+', x).group()))  # Example for 'May-25' -> 25
        except Exception as e:  #
            print(
                f"Warning: Could not load historical WIRP meetings for {ccy_code}: {e}. Interactive historical plot may not work.")  #

        # Prepare ColumnDataSource
        _month = df_wirp['meet'].tolist()  #
        _step = df_wirp['step'].tolist()  #
        _cum = df_wirp['cum'].tolist()  #
        _fwd_base = df_wirp['cb'].tolist()  #

        s1_source = ColumnDataSource(data=dict(x=_month, y=_step, z=_cum, k=_fwd_base))  #

        # --- Plot 1: Step Plot ---
        s1 = figure(x_range=_month, width=550, height=400,  #
                    tools=["pan", "tap", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],
                    # Added hover for step
                    toolbar_location='right', title="WIRP Step Changes")  #
        s1.xgrid.visible = False  #
        s1.ygrid.visible = False  #
        s1.vbar(x='x', top='y', width=0.7, source=s1_source, color='lightsteelblue')  #

        zero_line = Span(location=0, dimension='width', line_color='darkseagreen', line_width=1)  #
        s1.renderers.extend([zero_line])  #

        labels_1 = LabelSet(x='x', y='y', text='y', level='glyph', text_align='center', y_offset=-16, source=s1_source,
                            text_font_size='10px', text_color='midnightblue')  #
        s1.add_layout((labels_1))  #
        s1.xaxis.major_label_orientation = np.pi / 4  #
        s1.y_range = Range1d(min(_step) - 5, max(_step) + 5)  #
        s1.min_border_bottom = 50  #
        s1.min_border_right = 50  #

        # Hover tool for steps
        s1.add_tools(HoverTool(tooltips=[  #
            ("Meeting", "@x"),  #
            ("Step", "@y{0.0}bps"),  #
            ("Fwd Base", "@k{0.00}")  #
        ]))  #

        # --- Plot 2: Cumulative Plot ---
        s2 = figure(x_range=_month, width=550, height=400,  #
                    tools=["pan", "tap", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],  #
                    toolbar_location='right', title="WIRP Cumulative Changes")  #
        s2.xgrid.visible = False  #
        s2.ygrid.visible = False  #
        s2.vbar(x='x', top='z', width=0.7, source=s1_source, color='lightsteelblue')  #
        s2.renderers.extend([zero_line])  #

        labels_2 = LabelSet(x='x', y='z', text='z', level='glyph', text_align='center', y_offset=3, source=s1_source,
                            text_font_size='8px', text_color='midnightblue')  #
        labels_fwd_base = LabelSet(x='x', y='z', text='k', level='glyph', text_align='center', y_offset=-10,
                                   source=s1_source, text_font_size='8px', text_color='firebrick')  #
        s2.add_layout((labels_2))  #
        s2.add_layout((labels_fwd_base))  #
        s2.xaxis.major_label_orientation = np.pi / 4  #
        s2.y_range = Range1d(min(_cum) - 10, max(_cum) + 10)  #

        s2.add_tools(HoverTool(tooltips=[  #
            ("Meeting", "@x"),  #
            ("Cumulative", "@z{0.0}bps"),  #
            ("Fwd Base", "@k{0.00}")  #
        ]))  #

        tab1 = TabPanel(child=s1, title="Step")  #
        tab2 = TabPanel(child=s2, title="Cumulative")  #
        tabs = Tabs(tabs=[tab1, tab2])  #

        # --- Plot 3: Historical Meeting Data (Interactive) ---
        s3 = figure(width=570, height=400,  #
                    tools=["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                    toolbar_location='right', title="Historical Meeting Data (Click on bar to view)")  #
        s3.xaxis.formatter = DatetimeTickFormatter(days="%d-%b-%y", months="%d-%b-%y")  #
        s3.xgrid.visible = False  #
        s3.ygrid.visible = False  #
        s3.visible = False  # Hidden by default
        s3.renderers.extend([zero_line])  #
        s3.min_border_top = 30  #
        s3.min_border_left = 50  #

        # Hover tool for historical plot
        s3.add_tools(HoverTool(tooltips=[  #
            ("Date", "@x{%d-%b-%y}"),  #
            ("Value", "@y{0.0}bps")  #
        ], formatters={'$x': 'datetime'}))  #

        # --- Tap Callbacks for interactivity ---
        def _remove_glyphs(figure_obj: figure, glyph_name: str) -> None:  # Helper from original PLOT_BOKEH
            renderers = figure_obj.select(dict(type=GlyphRenderer))  #
            for r in renderers:  #
                if r.name == glyph_name:  #
                    r.data_source.data['y'] = [np.nan] * len(r.data_source.data['y'])  # Hide by setting to NaN
            figure_obj.legend.items = []  # Clear legend

        def step_callback(event: Tap) -> None:  #
            _remove_glyphs(s3, 'meet_plot')  #
            _remove_glyphs(s3, 'cum_meet_plot')  # Clear any cumulative plot if present

            selected_indices = s1_source.selected.indices  #
            if selected_indices:  #
                selected_meet_label = s1_source.data['x'][selected_indices[0]]  #

                df_meet_hist_filtered = pd.DataFrame()  #
                if is_generated_meeting_num:  #
                    # Original `meet = int(out_indicies[0]) + 1`
                    selected_meet_num = selected_indices[0] + 1  #
                    df_meet_hist_filtered = full_meet_hist_df[
                        full_meet_hist_df['meet_num'] == selected_meet_num].reset_index(drop=True)  #
                else:  #
                    df_meet_hist_filtered = full_meet_hist_df[
                        full_meet_hist_df['meet'] == selected_meet_label].reset_index(drop=True)  #

                if not df_meet_hist_filtered.empty:  #
                    source_hist = ColumnDataSource(data={  #
                        'x': df_meet_hist_filtered['date'],  #
                        'y': df_meet_hist_filtered['step']  #
                    })  #
                    s3.line(x='x', y='y', source=source_hist, color='tomato',  #
                            legend_label=f"Step for {selected_meet_label}", alpha=1.0, muted_alpha=0.1,
                            name='meet_plot')  #
                    s3.visible = True  #
                    s3.legend.location = 'bottom_left'  #
                    s3.legend.label_text_font_size = "8pt"  #
                    s3.legend.spacing = 1  #
                    s3.legend.background_fill_alpha = 0.0  #
                    s3.legend.click_policy = "mute"  #
                else:  #
                    s3.visible = False  #
            else:  #
                s3.visible = False  #

        s1.on_event(Tap, step_callback)  #

        def cum_callback(event: Tap) -> None:  #
            _remove_glyphs(s3, 'meet_plot')  #
            _remove_glyphs(s3, 'cum_meet_plot')  # Clear previous cum plot

            selected_indices = s1_source.selected.indices  #
            if len(selected_indices) == 1:  # Single selection for cumulative history
                selected_meet_label = s1_source.data['x'][selected_indices[0]]  #

                df_meet_hist_filtered = pd.DataFrame()  #
                if is_generated_meeting_num:  #
                    selected_meet_num = selected_indices[0] + 1  #
                    df_meet_hist_filtered = full_meet_hist_df[
                        full_meet_hist_df['meet_num'] == selected_meet_num].reset_index(drop=True)  #
                else:  #
                    df_meet_hist_filtered = full_meet_hist_df[
                        full_meet_hist_df['meet'] == selected_meet_label].reset_index(drop=True)  #

                if not df_meet_hist_filtered.empty:  #
                    source_hist = ColumnDataSource(data={  #
                        'x': df_meet_hist_filtered['date'],  #
                        'y': df_meet_hist_filtered['cum']  #
                    })  #
                    s3.line(x='x', y='y', source=source_hist, color='mediumseagreen',  #
                            legend_label=f"Cum for {selected_meet_label}", alpha=1.0, muted_alpha=0.1,
                            name='cum_meet_plot')  #
                    s3.visible = True  #
                    s3.legend.location = 'bottom_left'  #
                    s3.legend.label_text_font_size = "8pt"  #
                    s3.legend.spacing = 1  #
                    s3.legend.background_fill_alpha = 0.0  #
                    s3.legend.click_policy = "mute"  #
                else:  #
                    s3.visible = False  #
            elif len(selected_indices) == 2:  # Dual selection for spread history
                selected_meet_labels = [s1_source.data['x'][idx] for idx in selected_indices]  #

                df_filtered_hist_1 = pd.DataFrame()  #
                df_filtered_hist_2 = pd.DataFrame()  #

                if is_generated_meeting_num:  #
                    meet_num_1 = selected_indices[0] + 1  #
                    meet_num_2 = selected_indices[1] + 1  #
                    df_filtered_hist_1 = full_meet_hist_df[full_meet_hist_df['meet_num'] == meet_num_1]  #
                    df_filtered_hist_2 = full_meet_hist_df[full_meet_hist_df['meet_num'] == meet_num_2]  #
                else:  #
                    df_filtered_hist_1 = full_meet_hist_df[full_meet_hist_df['meet'] == selected_meet_labels[0]]  #
                    df_filtered_hist_2 = full_meet_hist_df[full_meet_hist_df['meet'] == selected_meet_labels[1]]  #

                if not df_filtered_hist_1.empty and not df_filtered_hist_2.empty:  #
                    # Merge on date to calculate spread
                    df_merged = pd.merge(df_filtered_hist_1[['date', 'cum']], df_filtered_hist_2[['date', 'cum']],
                                         on='date', suffixes=('_1', '_2'))  #
                    df_merged['sprd'] = df_merged['cum_1'] - df_merged['cum_2']  #

                    source_spread_hist = ColumnDataSource(data={  #
                        'x': df_merged['date'],  #
                        'y': df_merged['sprd']  #
                    })  #
                    s3.line(x='x', y='y', source=source_spread_hist, color='darkgoldenrod',  #
                            legend_label=f"Spread {selected_meet_labels[0]} - {selected_meet_labels[1]}", alpha=1.0,
                            muted_alpha=0.1, name='cum_meet_plot')  #
                    s3.visible = True  #
                    s3.legend.location = 'bottom_left'  #
                    s3.legend.label_text_font_size = "8pt"  #
                    s3.legend.spacing = 1  #
                    s3.legend.background_fill_alpha = 0.0  #
                    s3.legend.click_policy = "mute"  #
                else:  #
                    s3.visible = False  #
            else:  #
                s3.visible = False  #

        s2.on_event(Tap, cum_callback)  #

        return [tabs, s3]  #

    def plot_multi_wirp_bokeh(self,
                              wirp_data: Dict[str, Any],  # Output from WirpProcessor.get_wirp_data for multiple ccy
                              show_change: bool = False,  # Original `chg`
                              is_generated_meeting_num: bool = False,  # Original `gen`
                              valuation_dates: Optional[List[datetime.date]] = None
                              # Original `dates` (for change label)
                              ) -> layout:
        """
        Plots WIRP (World Interest Rate Probability) data for multiple currencies,
        with optional historical changes and interactive historical meeting data.

        Args:
            wirp_data (Dict[str, Any]): Dictionary containing 'wirp_data' and 'historical_cb_rates'
                                        from `WirpProcessor.get_wirp_data` for multiple currencies.
            show_change (bool): If True, plots changes between the first valuation date and subsequent dates.
            is_generated_meeting_num (bool): If True, uses meeting number as x-axis for historical.
            valuation_dates (Optional[List[datetime.date]]): List of valuation dates provided to `get_wirp_data`.

        Returns:
            layout: A Bokeh layout object containing the grid of plots.
        """
        ccy_codes = list(wirp_data['wirp_data'].keys())  #
        n_currencies = len(ccy_codes)  #
        n_valuation_dates = len(wirp_data['wirp_data'][ccy_codes[0]])  # # Number of valuation dates for each currency

        # Load historical meeting data for interactive plots
        full_meet_hist_dfs: Dict[str, pd.DataFrame] = {}  #
        for ccy_code in ccy_codes:  #
            ccy_config = config.get_currency_config(ccy_code)  #
            ois_meet_hist_file = ccy_config.get('ois_meet_hist_file')  #
            try:  #
                hist_df = data_loader.load_pickle(ois_meet_hist_file)  #
                if 'meet_num' not in hist_df.columns:  #
                    hist_df['meet_num'] = hist_df['meet'].apply(lambda x: int(re.search(r'\d+', x).group()))  #
                full_meet_hist_dfs[ccy_code] = hist_df  #
            except Exception as e:  #
                print(
                    f"Warning: Could not load historical WIRP meetings for {ccy_code}: {e}. Interactive historical plot may not work.")  #
                full_meet_hist_dfs[ccy_code] = pd.DataFrame()  #

        # --- Data Preparation ---
        # Initialize data structures for Bokeh sources
        step_data_curr: Dict[str, List[float]] = {ccy: [] for ccy in ccy_codes}  #
        cum_data_curr: Dict[str, List[float]] = {ccy: [] for ccy in ccy_codes}  #
        fwd_base_data_curr: Dict[str, List[float]] = {ccy: [] for ccy in ccy_codes}  #

        # Populate current data (first valuation date)
        max_len_curr_data = 0  #
        for ccy_code in ccy_codes:  #
            if wirp_data['wirp_data'][ccy_code]:  #
                df = wirp_data['wirp_data'][ccy_code][0]  # First valuation date
                step_data_curr[ccy_code] = df['step'].tolist()  #
                cum_data_curr[ccy_code] = df['cum'].tolist()  #
                fwd_base_data_curr[ccy_code] = df['cb'].tolist()  #
                if len(df) > max_len_curr_data:  #
                    max_len_curr_data = len(df)  #

        # Pad shorter lists to max_len_curr_data with NaNs or zeros for consistent plotting
        for ccy_code in ccy_codes:  #
            step_data_curr[ccy_code].extend([0] * (max_len_curr_data - len(step_data_curr[ccy_code])))  #
            cum_data_curr[ccy_code].extend(
                [cum_data_curr[ccy_code][-1]] * (max_len_curr_data - len(cum_data_curr[ccy_code])))  #
            fwd_base_data_curr[ccy_code].extend(
                [fwd_base_data_curr[ccy_code][-1]] * (max_len_curr_data - len(fwd_base_data_curr[ccy_code])))  #

        # Generate x-axis labels for current plots
        grps_curr = np.arange(1, max_len_curr_data + 1)  #
        x_lab_curr = flat_lst([[(str(g),
                                 f"{ccy_code[:-3].ljust(10)} {wirp_data['wirp_data'][ccy_code][0]['meet'].iloc[g - 1].rjust(10)}")
                                for g in grps_curr] for ccy_code in ccy_codes])  #

        # Prepare data for Bokeh source for current step/cum plots
        s1_source_curr = ColumnDataSource(data=dict(  #
            x=x_lab_curr,  #
            counts=flat_lst([step_data_curr[ccy] for ccy in ccy_codes]),  #
            cum_counts=flat_lst([cum_data_curr[ccy] for ccy in ccy_codes]),  #
            fwd_base_counts=flat_lst([fwd_base_data_curr[ccy] for ccy in ccy_codes]),  #
            colors=[Category10[n_currencies][i % n_currencies] for i in range(len(x_lab_curr))],  #
            legend_name=flat_lst([[ccy[:-3]] * max_len_curr_data for ccy in ccy_codes])  #
        ))  #

        # --- Plot 1: Current Step Plot (Grouped Bar Chart) ---
        s1_curr = figure(x_range=FactorRange(*x_lab_curr), width=800, height=450,  #
                         tools=["pan", "tap", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                         toolbar_location='right', title="WIRP Step Changes (Current)")  #
        s1_curr.xgrid.visible = False  #
        s1_curr.ygrid.visible = False  #
        s1_curr.vbar(x='x', top='counts', width=0.7, source=s1_source_curr, fill_color='colors', line_color='colors',
                     legend_field='legend_name')  #
        labels_1_curr = LabelSet(x='x', y='counts', text='counts', level='glyph', text_align='center', y_offset=-14,
                                 source=s1_source_curr, text_font_size='10px', text_color='midnightblue')  #
        s1_curr.add_layout((labels_1_curr))  #
        zero_line = Span(location=0, dimension='width', line_color='darkseagreen', line_width=1)  #
        s1_curr.renderers.extend([zero_line])  #
        s1_curr.legend.location = 'bottom_right'  #
        s1_curr.legend.label_text_font = "calibri"  #
        s1_curr.legend.label_text_font_size = "9pt"  #
        s1_curr.legend.spacing = 1  #
        s1_curr.legend.background_fill_alpha = 0.0  #
        s1_curr.legend.click_policy = "mute"  #
        s1_curr.xaxis.major_label_orientation = np.pi / 2  #
        s1_curr.xaxis.axis_label_text_font_size = "2pt"  #
        s1_curr.min_border_bottom = 50  #

        s1_curr.add_tools(HoverTool(tooltips=[  #
            ("Meeting", "@x"),  #
            ("Step", "@counts{0.0}bps")  #
        ]))  #

        # --- Plot 2: Current Cumulative Plot (Grouped Bar Chart) ---
        x_lab_cum_curr = flat_lst([[(str(g),
                                     f"{ccy_code[:-3].ljust(10)} {fwd_base_data_curr[ccy_code][g - 1]:.2f} {wirp_data['wirp_data'][ccy_code][0]['meet'].iloc[g - 1].rjust(10)}")
                                    for g in grps_curr] for ccy_code in ccy_codes])  #

        s2_curr = figure(x_range=FactorRange(*x_lab_cum_curr), width=800, height=450,  #
                         tools=["pan", "tap", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                         toolbar_location='right', title="WIRP Cumulative Changes (Current)")  #
        s2_curr.xgrid.visible = False  #
        s2_curr.ygrid.visible = False  #
        s2_curr.vbar(x='x', top='cum_counts', width=0.7, source=s1_source_curr, fill_color='colors',
                     line_color='colors', legend_field='legend_name')  #
        labels_2_curr = LabelSet(x='x', y='cum_counts', text='cum_counts', level='glyph', text_align='center',
                                 y_offset=-14, x_offset=-3, source=s1_source_curr, text_font_size='8px',
                                 text_color='midnightblue')  #
        s2_curr.add_layout((labels_2_curr))  #
        s2_curr.renderers.extend([zero_line])  #
        s2_curr.y_range = Range1d(min(s1_source_curr.data['cum_counts']) - 15,
                                  max(s1_source_curr.data['cum_counts']) + 15)  #
        s2_curr.legend.location = 'bottom_left'  #
        s2_curr.legend.label_text_font = "calibri"  #
        s2_curr.legend.label_text_font_size = "9pt"  #
        s2_curr.legend.spacing = 1  #
        s2_curr.legend.background_fill_alpha = 0.0  #
        s2_curr.legend.click_policy = "mute"  #
        s2_curr.xaxis.major_label_orientation = np.pi / 2  #
        s2_curr.xaxis.axis_label_text_font_size = "2pt"  #
        s2_curr.min_border_bottom = 50  #

        s2_curr.add_tools(HoverTool(tooltips=[  #
            ("Meeting", "@x"),  #
            ("Cumulative", "@cum_counts{0.0}bps"),  #
            ("Fwd Base", "@fwd_base_counts{0.00}")  #
        ]))  #

        # --- Plot 3 & 4: Historical Meeting Data (Interactive) ---
        s3_hist_step = figure(width=525, height=450,  #
                              tools=["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                              toolbar_location='right', title="Historical Step Data (Click on bar to view)")  #
        s3_hist_step.xaxis.formatter = DatetimeTickFormatter(days="%d-%b-%y", months="%d-%b-%y")  #
        s3_hist_step.xgrid.visible = False  #
        s3_hist_step.ygrid.visible = False  #
        s3_hist_step.visible = False  # Hidden by default
        s3_hist_step.renderers.extend([zero_line])  #

        s3_hist_step.add_tools(HoverTool(tooltips=[  #
            ("Date", "@x{%d-%b-%y}"),  #
            ("Value", "@y{0.0}bps")  #
        ], formatters={'$x': 'datetime'}))  #

        s4_hist_cum = figure(width=525, height=450,  #
                             tools=["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                             toolbar_location='right', title="Historical Cumulative Data (Click on bar to view)")  #
        s4_hist_cum.xaxis.formatter = DatetimeTickFormatter(days="%d-%b-%y", months="%d-%b-%y")  #
        s4_hist_cum.xgrid.visible = False  #
        s4_hist_cum.ygrid.visible = False  #
        s4_hist_cum.visible = False  # Hidden by default
        s4_hist_cum.renderers.extend([zero_line])  #

        s4_hist_cum.add_tools(HoverTool(tooltips=[  #
            ("Date", "@x{%d-%b-%y}"),  #
            ("Value", "@y{0.0}bps")  #
        ], formatters={'$x': 'datetime'}))  #

        # --- Tap Callbacks for interactivity ---
        # Helper function to remove glyphs
        def _remove_glyphs_bokeh(figure_obj: figure, glyph_name_substring: str) -> None:  #
            renderers = figure_obj.select(dict(type=GlyphRenderer))  #
            for r in renderers:  #
                if glyph_name_substring in r.name:  #
                    # To remove, set data to empty or nan
                    r.data_source.data['x'] = []  #
                    r.data_source.data['y'] = []  #
            figure_obj.legend.items = []  #

        def step_callback_multi(event: Tap) -> None:  #
            _remove_glyphs_bokeh(s3_hist_step, 'meet_plot')  #

            selected_indices = s1_source_curr.selected.indices  #
            if selected_indices:  #
                # Determine which meeting (e.g., 'May-25') and currency were clicked
                clicked_x_label = s1_source_curr.data['x'][selected_indices[0]]  #
                meeting_label = clicked_x_label[1].strip()  # e.g., 'May-25'
                ccy_code = clicked_x_label[1].split(' ')[0].strip() + 'M'  # e.g., 'USDM' -> 'USD'

                df_meet_hist_filtered = pd.DataFrame()  #
                if full_meet_hist_dfs.get(ccy_code) is not None:  #
                    if is_generated_meeting_num:  #
                        # Original `meet = int(s1_source.data['x'][int(out_indicies[0])][0])`
                        selected_meet_num = int(clicked_x_label[0])  #
                        df_meet_hist_filtered = full_meet_hist_dfs[ccy_code][
                            full_meet_hist_dfs[ccy_code]['meet_num'] == selected_meet_num].reset_index(drop=True)  #
                    else:  #
                        df_meet_hist_filtered = full_meet_hist_dfs[ccy_code][
                            full_meet_hist_dfs[ccy_code]['meet'] == meeting_label].reset_index(drop=True)  #

                if not df_meet_hist_filtered.empty:  #
                    source_hist = ColumnDataSource(data={  #
                        'x': df_meet_hist_filtered['date'],  #
                        'y': df_meet_hist_filtered['step']  #
                    })  #
                    s3_hist_step.line(x='x', y='y', source=source_hist, color='tomato',  #
                                      legend_label=f"{ccy_code}: {meeting_label}", alpha=1.0, muted_alpha=0.1,
                                      name=f'{ccy_code} meet_plot')  #
                    s3_hist_step.visible = True  #
                    s3_hist_step.legend.location = 'bottom_left'  #
                    s3_hist_step.legend.label_text_font_size = "7pt"  #
                    s3_hist_step.legend.spacing = 1  #
                    s3_hist_step.legend.background_fill_alpha = 0.0  #
                    s3_hist_step.legend.click_policy = "mute"  #
                else:  #
                    s3_hist_step.visible = False  #
            else:  #
                s3_hist_step.visible = False  #

        s1_curr.on_event(Tap, step_callback_multi)  #

        def cum_callback_multi(event: Tap) -> None:  #
            _remove_glyphs_bokeh(s4_hist_cum, 'cum_meet_plot')  #

            selected_indices = s1_source_curr.selected.indices  #
            if selected_indices:  #
                clicked_x_label = s1_source_curr.data['x'][selected_indices[0]]  #
                meeting_label = clicked_x_label[1].split(' ')[-1].strip()  #
                ccy_code = clicked_x_label[1].split(' ')[0].strip() + 'M'  #

                df_meet_hist_filtered = pd.DataFrame()  #
                if full_meet_hist_dfs.get(ccy_code) is not None:  #
                    if is_generated_meeting_num:  #
                        selected_meet_num = int(clicked_x_label[0])  #
                        df_meet_hist_filtered = full_meet_hist_dfs[ccy_code][
                            full_meet_hist_dfs[ccy_code]['meet_num'] == selected_meet_num].reset_index(drop=True)  #
                    else:  #
                        df_meet_hist_filtered = full_meet_hist_dfs[ccy_code][
                            full_meet_hist_dfs[ccy_code]['meet'] == meeting_label].reset_index(drop=True)  #

                if not df_meet_hist_filtered.empty:  #
                    source_hist = ColumnDataSource(data={  #
                        'x': df_meet_hist_filtered['date'],  #
                        'y': df_meet_hist_filtered['cum']  #
                    })  #
                    s4_hist_cum.line(x='x', y='y', source=source_hist, color='mediumseagreen',  #
                                     legend_label=f"{ccy_code}: {meeting_label}", alpha=1.0, muted_alpha=0.1,
                                     name=f'{ccy_code} cum_meet_plot')  #
                    s4_hist_cum.visible = True  #
                    s4_hist_cum.legend.location = 'bottom_left'  #
                    s4_hist_cum.legend.label_text_font_size = "7pt"  #
                    s4_hist_cum.legend.spacing = 1  #
                    s4_hist_cum.legend.background_fill_alpha = 0.0  #
                    s4_hist_cum.legend.click_policy = "mute"  #
                else:  #
                    s4_hist_cum.visible = False  #
            else:  #
                s4_hist_cum.visible = False  #

        s2_curr.on_event(Tap, cum_callback_multi)  #

        grid_plots_current = gridplot([[s1_curr, s3_hist_step], [s2_curr, s4_hist_cum]])  #
        grid_plots_current.toolbar_location = 'left'  #

        # --- Plotting Change Data (if show_change is True) ---
        if show_change and n_valuation_dates > 1:  #
            step_data_chg: Dict[str, List[float]] = {ccy: [] for ccy in ccy_codes}  #
            cum_data_chg: Dict[str, List[float]] = {ccy: [] for ccy in ccy_codes}  #

            # Populate change data
            max_len_chg_data = 0  #
            for ccy_code in ccy_codes:  #
                # Compare first valuation date data with second valuation date data
                df_curr = wirp_data['wirp_data'][ccy_code][0]  #
                df_prev = wirp_data['wirp_data'][ccy_code][1]  #

                # Handle potential length differences by taking min length
                min_len_comp = min(len(df_curr), len(df_prev))  #

                step_data_chg[ccy_code] = (np.array(df_curr['step'][:min_len_comp]) - np.array(
                    df_prev['step'][:min_len_comp])).tolist()  #
                cum_data_chg[ccy_code] = (np.array(df_curr['cum'][:min_len_comp]) - np.array(
                    df_prev['cum'][:min_len_comp])).tolist()  #

                if min_len_comp > max_len_chg_data:  #
                    max_len_chg_data = min_len_comp  #

            # Generate x-axis labels for change plots
            grps_chg = np.arange(1, max_len_chg_data + 1)  #
            x_lab_chg = flat_lst([[(str(g),
                                    f"{ccy_code[:-3].ljust(10)} {wirp_data['wirp_data'][ccy_code][0]['meet'].iloc[g - 1].rjust(10)}")
                                   for g in grps_chg] for ccy_code in ccy_codes])  #

            # Prepare data for Bokeh source for change plots
            s_change_source = ColumnDataSource(data=dict(  #
                x=x_lab_chg,  #
                counts=flat_lst([step_data_chg[ccy] for ccy in ccy_codes]),  #
                cum_counts=flat_lst([cum_data_chg[ccy] for ccy in ccy_codes]),  #
                colors=[Category10[n_currencies][i % n_currencies] for i in range(len(x_lab_chg))],  #
                legend_name=flat_lst([[ccy[:-3]] * max_len_chg_data for ccy in ccy_codes])  #
            ))  #

            # --- Plot 5: Step Change Plot (Grouped Scatter) ---
            s5_chg_step = figure(x_range=FactorRange(*x_lab_chg), width=800, height=450,  #
                                 tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],  #
                                 toolbar_location='right',
                                 title=f"WIRP Step Changes ({valuation_dates[0].strftime('%Y-%m-%d')} vs {valuation_dates[1].strftime('%Y-%m-%d')})")  #
            s5_chg_step.xgrid.visible = False  #
            s5_chg_step.ygrid.visible = False  #

            for i, ccy_code in enumerate(ccy_codes):  #
                # Filter source data for this currency for scatter plot
                ccy_specific_x = [x for x, name in zip(s_change_source.data['x'], s_change_source.data['legend_name'])
                                  if name == ccy_code[:-3]]  #
                ccy_specific_y = [y for y, name in
                                  zip(s_change_source.data['counts'], s_change_source.data['legend_name']) if
                                  name == ccy_code[:-3]]  #

                s5_chg_step.scatter(x=ccy_specific_x, y=ccy_specific_y,  #
                                    marker='circle_dot', size=8, color=Category10[n_currencies][i % n_currencies],  #
                                    legend_label=ccy_code[:-3], alpha=1.0, muted_alpha=0.1)  #

            s5_chg_step.renderers.extend([zero_line])  #
            s5_chg_step.legend.location = 'bottom_right'  #
            s5_chg_step.legend.label_text_font = "calibri"  #
            s5_chg_step.legend.label_text_font_size = "9pt"  #
            s5_chg_step.legend.spacing = 1  #
            s5_chg_step.legend.background_fill_alpha = 0.0  #
            s5_chg_step.legend.click_policy = "mute"  #
            s5_chg_step.xaxis.major_label_orientation = np.pi / 2  #
            s5_chg_step.xaxis.axis_label_text_font_size = "2pt"  #
            s5_chg_step.min_border_bottom = 50  #

            s5_chg_step.add_tools(HoverTool(tooltips=[  #
                ("Meeting", "@x"),  #
                ("Step Change", "@y{0.0}bps")  #
            ]))  #

            # --- Plot 6: Cumulative Change Plot (Grouped Scatter) ---
            s6_chg_cum = figure(x_range=FactorRange(*x_lab_chg), width=800, height=450,  #
                                tools=["pan", "hover", "wheel_zoom", "box_zoom", "save", "reset", "help"],  #
                                toolbar_location='right',
                                title=f"WIRP Cumulative Changes ({valuation_dates[0].strftime('%Y-%m-%d')} vs {valuation_dates[1].strftime('%Y-%m-%d')})")  #
            s6_chg_cum.xgrid.visible = False  #
            s6_chg_cum.ygrid.visible = False  #

            for i, ccy_code in enumerate(ccy_codes):  #
                ccy_specific_x = [x for x, name in zip(s_change_source.data['x'], s_change_source.data['legend_name'])
                                  if name == ccy_code[:-3]]  #
                ccy_specific_y = [y for y, name in
                                  zip(s_change_source.data['cum_counts'], s_change_source.data['legend_name']) if
                                  name == ccy_code[:-3]]  #

                s6_chg_cum.scatter(x=ccy_specific_x, y=ccy_specific_y,  #
                                   marker='circle_dot', size=8, color=Category10[n_currencies][i % n_currencies],  #
                                   legend_label=f"{ccy_code[:-3]}: {valuation_dates[1].strftime('%d-%m-%Y')}",
                                   alpha=1.0, muted_alpha=0.1)  #

            s6_chg_cum.renderers.extend([zero_line])  #
            s6_chg_cum.legend.location = 'bottom_left'  #
            s6_chg_cum.legend.label_text_font = "calibri"  #
            s6_chg_cum.legend.label_text_font_size = "7.5pt"  #
            s6_chg_cum.legend.spacing = 1  #
            s6_chg_cum.legend.background_fill_alpha = 0.0  #
            s6_chg_cum.legend.click_policy = "mute"  #
            s6_chg_cum.xaxis.major_label_orientation = np.pi / 2  #
            s6_chg_cum.xaxis.axis_label_text_font_size = "2pt"  #
            s6_chg_cum.min_border_bottom = 50  #

            s6_chg_cum.add_tools(HoverTool(tooltips=[  #
                ("Meeting", "@x"),  #
                ("Cumulative Change", "@y{0.0}bps")  #
            ]))  #

            grid_plots_change = gridplot(
                [[s5_chg_step, s3_hist_step], [s6_chg_cum, s4_hist_cum]])  # Reusing historical plots
            grid_plots_change.toolbar_location = 'left'  #

            return grid_plots_change  #

        else:  # If no change is to be shown
            return grid_plots_current  #

    def plot_option_vol_surface_bokeh(self,
                                      vol_surface_builders: List[Union[BondFutureVolSurface, StirVolSurface]]
                                      # Original `v`
                                      ) -> List[Tabs]:
        """
        Plots interactive option volatility surfaces using Bokeh.
        Displays tabs for call/put options against strike and delta.

        Args:
            vol_surface_builders (List[Union[BondFutureVolSurface, StirVolSurface]]):
                List of built volatility surface builder objects (e.g., for different dates).

        Returns:
            List[Tabs]: A list containing a Bokeh Tabs object for the volatility surface plots.
        """
        if not vol_surface_builders:  #
            print("No volatility surface builders provided.")  #
            return [Tabs(tabs=[TabPanel(child=figure(title="No Data"), title="Calls")])]  #

        # Get initial underlying price from the first builder (assumed consistent)
        live_px = vol_surface_builders[0].underlying_price  #

        # Prepare data for plotting
        df_call_list: List[pd.DataFrame] = []  #
        df_put_list: List[pd.DataFrame] = []  #

        for builder in vol_surface_builders:  #
            builder.build_surface(builder.underlying_ticker.split(' ')[0][:4],
                                  builder.chain_length)  # Ensure surface is built
            df_full = builder.surface_data  #

            df_call_list.append(df_full[df_full['option_type'] == 'C'])  #
            df_put_list.append(df_full[df_full['option_type'] == 'P'])  #

        # --- Common Plotting Settings ---
        common_tools = ["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"]  #
        common_hover_tooltips = [('Strike', '@strikes'), ('IV', '@iv{0.00}%'), ('Delta', '@delta{0.0}'),
                                 ('Price', '@px_mid{0.00}')]  #

        def _create_vol_plot(df_list: List[pd.DataFrame], plot_title: str, line_color: str, marker_color: str,
                             x_axis_col: str, x_axis_label: str, hover_extra_col: str = None) -> figure:
            p = figure(width=700, height=300, tools=common_tools, toolbar_location='right', title=plot_title)  #
            p.xgrid.visible = False  #
            p.ygrid.visible = False  #

            # Add specific hover tooltip for this plot if needed
            hover_tooltips_current = list(common_hover_tooltips)  #
            if hover_extra_col == 'delta':  # Add specific delta tooltip for delta plots
                hover_tooltips_current.append(('ATM_K', '@atm_k{0.0}'))  #

            p.add_tools(HoverTool(tooltips=hover_tooltips_current))  #

            for i, df in enumerate(df_list):  #
                source = ColumnDataSource(data={  #
                    'x': df[x_axis_col],  #
                    'y': df['iv'],  #
                    'strikes': df['strike'],  #
                    'delta': df['delta'],  #
                    'px_mid': df['px'],  # Original 'px'
                    'atm_k': df['ATM_K']  #
                })  #
                p.line(x='x', y='y', source=source,  #
                       legend_label=f"{ql_to_datetime(vol_surface_builders[i].reference_date).strftime('%Y-%m-%d')}",  #
                       line_width=0.7, color=line_color)  #
                p.scatter(x='x', y='y', source=source,  #
                          width=0.7, color=marker_color, size=10, marker="dot")  #

            # Add live price/delta line
            if x_axis_col == 'strikes':  #
                live_line = Span(location=live_px, dimension='height', line_color='goldenrod', line_width=1)  #
            elif x_axis_col == 'delta':  #
                live_line = Span(location=50 if 'call' in plot_title.lower() else -50, dimension='height',
                                 line_color='goldenrod', line_width=1)  #
            else:
                live_line = None  #

            if live_line:  #
                p.add_layout(live_line)  #

            p.legend.location = 'bottom_right'  #
            p.legend.label_text_font = "calibri"  #
            p.legend.label_text_font_size = "9pt"  #
            p.legend.spacing = 1  #
            p.legend.click_policy = "mute"  #
            p.legend.background_fill_alpha = 0.0  #
            p.yaxis.axis_label = 'Implied Volatility (%)'  #
            p.xaxis.axis_label = x_axis_label  #
            return p  #

        # --- Create Plots for Tabs ---
        s1_call_strikes = _create_vol_plot(df_call_list, "Call Options vs. Strike", 'royalblue', 'navy', 'strike',
                                           'Strike Price')  #
        s2_call_delta = _create_vol_plot(df_call_list, "Call Options vs. Delta", 'royalblue', 'navy', 'delta', 'Delta',
                                         'delta')  #
        s3_put_delta = _create_vol_plot(df_put_list, "Put Options vs. Delta", 'palevioletred', 'red', 'delta', 'Delta',
                                        'delta')  #
        s4_put_strikes = _create_vol_plot(df_put_list, "Put Options vs. Strike", 'palevioletred', 'red', 'strike',
                                          'Strike Price')  #

        tab1 = TabPanel(child=s1_call_strikes, title="Call Strikes")  #
        tab2 = TabPanel(child=s2_call_delta, title="Call Delta")  #
        tab3 = TabPanel(child=s3_put_delta, title="Put Delta")  #
        tab4 = TabPanel(child=s4_put_strikes, title="Put Strikes")  #

        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])  #
        return [tabs]  #

    def plot_option_strategy_bokeh(self,
                                   strategy: OptionStrategy,  # Original `st`
                                   add_delta_hedge_notional: Optional[float] = None,  # Original `add_delta`
                                   payoff_increment_calc: int = 100  #
                                   ) -> List[figure]:
        """
        Plots the interactive payoff and delta profile for an option strategy using Bokeh.

        Args:
            strategy (OptionStrategy): An OptionStrategy object containing simulation results.
            add_delta_hedge_notional (Optional[float]): If provided, adds a linear delta hedge to the strategy payoff/delta.
                                                         This is typically the notional or contracts of the futures hedge.
            payoff_increment_calc (int): Number of points for calculating the expiry payoff curve.

        Returns:
            List[figure]: A list of Bokeh figure objects (P&L and Delta plots).
        """
        strategy_details = strategy.strategy_details  #
        strategy_df = strategy_details['strategy_simulation_df']  #

        spot_idx = strategy_df.loc[strategy_df['ATM_K'] == 0].index  #
        if spot_idx.empty:  #
            # If ATM_K=0 is not exactly in index, find closest
            spot_idx = (strategy_df['ATM_K']).abs().argsort()[:1].iloc[0]  #
        else:  #
            spot_idx = spot_idx[0]  #

        x_spot = strategy_df['fut_px'][spot_idx]  #

        # Determine price label format based on strategy type (USD for bond, stir for STIR)
        px_label: str  #
        y_strat_px: Union[pd.Series, np.ndarray]  #

        if strategy_details['strategy_type'] == 'USD':  #
            px_label = px_opt_ticks(strategy_details['strategy_current_px'])  #
            # Convert decimal prices to 64ths for Y-axis and plotting
            y_strat_px = strategy_df['strat_px_fmt'].apply(convert_to_64)  #
            y_axis_label_px = 'Strategy P&L (64ths)'  #
        elif strategy_details['strategy_type'] == 'stir':  #
            px_label = str(np.round(100 * strategy_details['strategy_current_px'], 2))  #
            y_strat_px = 100 * strategy_df['strat_px']  # Convert to basis points
            y_axis_label_px = 'Strategy P&L (bps)'  #
        else:  #
            px_label = str(np.round(strategy_details['strategy_current_px'], 3))  #
            y_strat_px = strategy_df['strat_px']  #
            y_axis_label_px = 'Strategy P&L'  #

        # --- Payoff at Expiry Calculation ---
        x_expiry_sim = np.linspace(strategy_df['fut_px'].min(), strategy_df['fut_px'].max(),
                                   num=payoff_increment_calc)  #

        expiry_payoff_vals = np.zeros_like(x_expiry_sim, dtype=float)  #

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
            else:  #
                payoff = np.zeros_like(x_expiry_sim)  # Handle invalid type gracefully

            expiry_payoff_vals += payoff * weight  #

        expiry_payoff_vals -= strategy_details['strategy_current_px']  # Subtract initial strategy cost

        if add_delta_hedge_notional is not None:  #
            expiry_payoff_vals += add_delta_hedge_notional * (x_expiry_sim - x_spot)  # Linear hedge component

        # Convert expiry payoff to 64ths if USD bond strategy
        if strategy_details['strategy_type'] == 'USD':  #
            y1_expiry_fmt = np.array([convert_to_64(px_opt_ticks(p)) for p in expiry_payoff_vals])  #
        elif strategy_details['strategy_type'] == 'stir':  #
            y1_expiry_fmt = expiry_payoff_vals * 100  # Convert to bps for STIR
        else:  #
            y1_expiry_fmt = expiry_payoff_vals  #

        # --- Plot 1: Payoff & P&L ---
        p1 = figure(width=700, height=350,
                    tools=["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                    toolbar_location='right',
                    title=f"{self.strategy_name} | Px: {px_label} | Delta: {np.round(strategy_df['strat_delta'][spot_idx], 1)}")  #

        # Tooltips
        tooltip_formatters = {'$x': '0.00' if strategy_details['strategy_type'] == 'USD' else '0.000'}  #
        if strategy_details['strategy_type'] == 'USD':  #
            _tooltips = [('Fut Px', '$x{0.00}'), ('P&L', '@y_px_fmt'), ('ATM_K', '@atm_k{0.0}')]  #
        elif strategy_details['strategy_type'] == 'stir':  #
            _tooltips = [('Fut Px', '$x{0.000}'), ('P&L', '@y_px_fmt{0.0}bps'), ('ATM_K', '@atm_k{0.0}')]  #
        else:  #
            _tooltips = [('Fut Px', '$x{0.00}'), ('P&L', '@y_px{0.00}'), ('ATM_K', '@atm_k{0.0}')]  #

        main_source = ColumnDataSource(data={  #
            'x_sim': strategy_df['fut_px'],  # Simulated future prices
            'y_px': y_strat_px,  # Simulated strategy P&L values
            'atm_k': strategy_df['ATM_K'],  #
            'y_px_fmt': strategy_df['strat_px_fmt']  # Formatted prices (e.g., 64ths)
        })  #

        p1.add_tools(HoverTool(tooltips=_tooltips, formatters=tooltip_formatters))  #
        p1.xgrid.visible = False  #
        p1.ygrid.visible = False  #

        p1.line(x=x_expiry_sim, y=y1_expiry_fmt, width=0.7, color='black', alpha=1.0, muted_alpha=0.1,
                legend_label="Payoff at Expiry")  #
        p1.line(x='x_sim', y='y_px', source=main_source, width=0.7, color='navy', alpha=1.0, muted_alpha=0.1,
                legend_label="Strategy P&L")  #

        live_line = Span(location=x_spot, dimension='height', line_color='goldenrod', line_width=1, line_dash='dashed',
                         name='Current Spot')  #
        zero_line = Span(location=0, dimension='width', line_color='darkseagreen', line_width=1, line_dash='dotted',
                         name='Break-even')  #
        p1.add_layout(live_line)  #
        p1.add_layout(zero_line)  #

        # ATM_K labels on plot
        # Original code used `strategy_df['ATM_K']` directly, assuming a subset for labels.
        # Here, select a subset for labels to avoid clutter.
        labels_indices = np.linspace(0, len(strategy_df) - 1, 6, dtype=int)  # Select 6 equidistant points
        p1_labels_source = ColumnDataSource(data={  #
            'x_labels': strategy_df['fut_px'].iloc[labels_indices],  #
            'z_labels': np.round(strategy_df['ATM_K'].iloc[labels_indices], 1)  #
        })  #
        labels_atm_k = LabelSet(x='x_labels', y=5, y_units='screen', text='z_labels', source=p1_labels_source,  #
                                text_font='calibri', text_color='firebrick', text_font_size='9px', text_align='center',
                                x_offset=0)  #
        p1.add_layout(labels_atm_k)  #

        p1.xaxis.ticker = strategy_df['fut_px'].iloc[labels_indices].tolist()  # Set specific tickers
        p1.xaxis.major_label_overrides = {val: f"{val:.2f}\n{atm:.1f}" for val, atm in
                                          zip(strategy_df['fut_px'].iloc[labels_indices],
                                              strategy_df['ATM_K'].iloc[labels_indices])}  #

        p1.legend.location = 'top_left'  #
        p1.legend.label_text_font = "calibri"  #
        p1.legend.label_text_font_size = "9pt"  #
        p1.legend.spacing = 1  #
        p1.legend.click_policy = "mute"  #
        p1.legend.background_fill_alpha = 0.0  #
        p1.yaxis.axis_label = y_axis_label_px  #

        # --- Plot 2: Delta ---
        p2 = figure(width=700, height=225,
                    tools=["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                    toolbar_location='right')  #

        # Delta tooltips
        p2_tooltips = [('Fut Px', '$x{0.00}'), ('Delta', '@y{0.00}')]  #
        p2.add_tools(HoverTool(tooltips=p2_tooltips, formatters=tooltip_formatters))  #
        p2.xgrid.visible = False  #
        p2.ygrid.visible = False  #

        y_strat_delta = strategy_df['strat_delta'] / 100  # Convert to decimal delta

        if add_delta_hedge_notional is not None:  #
            y_strat_delta_hedged = y_strat_delta + add_delta_hedge_notional  #
            p2_source_hedged = ColumnDataSource(data={'x': strategy_df['fut_px'], 'y': y_strat_delta_hedged})  #
            p2.line(x='x', y='y', source=p2_source_hedged, width=0.7, color='k', line_dash='dashed',
                    legend_label="Delta + Hedge")  #
            # Update strategy_df for `strat_delta_updt` if this is used by caller
            strategy_df['strat_delta_updt'] = y_strat_delta_hedged  #

        p2_source_delta = ColumnDataSource(data={'x': strategy_df['fut_px'], 'y': y_strat_delta})  #
        p2.line(x='x', y='y', source=p2_source_delta, width=0.7, color='green', alpha=1.0, muted_alpha=0.1,
                legend_label="Delta")  #

        p2.axvline(x_spot, lw=0.5, color='y', linestyle='--', name='Current Spot')  #

        # Delta labels on plot
        p2_labels_source = ColumnDataSource(data={  #
            'x_labels': strategy_df['fut_px'].iloc[labels_indices],  #
            'z_labels': np.round(strategy_df['strat_delta'].iloc[labels_indices] / 100, 2)  #
        })  #
        labels_delta_plot = LabelSet(x='x_labels', y=5, y_units='screen', text='z_labels', source=p2_labels_source,  #
                                     text_font='calibri', text_color='firebrick', text_font_size='9px',
                                     text_align='right', x_offset=0)  #
        p2.add_layout(labels_delta_plot)  #

        p2.xaxis.ticker = strategy_df['fut_px'].iloc[labels_indices].tolist()  #
        p2.legend.location = 'top_left'  #
        p2.legend.label_text_font = "calibri"  #
        p2.legend.label_text_font_size = "9pt"  #
        p2.legend.spacing = 1  #
        p2.legend.click_policy = "mute"  #
        p2.legend.background_fill_alpha = 0.0  #
        p2.yaxis.axis_label = 'Strategy Delta'  #

        return [p1, p2]  #

    def plot_listed_options_timeseries_bokeh(self,
                                             vol_surface_builder: Union[BondFutureVolSurface, StirVolSurface],
                                             # Original `v2`
                                             underlying_future_ticker: str,  # Original `fut_ticker`
                                             option_weights: List[float]  # Original `opt_w`
                                             ) -> figure:
        """
        Plots historical prices of a listed option strategy (e.g., using bond/STIR options)
        alongside its underlying future price.

        Args:
            vol_surface_builder (Union[BondFutureVolSurface, StirVolSurface]):
                An instantiated and built volatility surface builder (to get option tickers).
            underlying_future_ticker (str): Bloomberg ticker of the underlying future.
            option_weights (List[float]): Weights for each option in the strategy.

        Returns:
            figure: The Bokeh figure object.
        """
        # Ensure surface is built to get option tickers
        vol_surface_builder.build_surface(vol_surface_builder.underlying_ticker.split(' ')[0][:4],
                                          vol_surface_builder.chain_length)  #

        option_data_from_surface = vol_surface_builder.surface_data  #
        # Get full Bloomberg tickers for options in the strategy
        option_tickers = option_data_from_surface['FullTicker'].tolist()  #

        # Combine underlying future ticker and option tickers for bulk fetch
        all_tickers_to_fetch = [underlying_future_ticker] + option_tickers  #

        # Start date for historical data. Original code used '20230801'.
        start_date_bbg = '20230801'  #
        end_date_bbg = to_bbg_date_str(datetime.datetime.now(), ql_date=0)  #

        try:  #
            df_hist_prices = b_con.bdh(all_tickers_to_fetch, 'PX_LAST', start_date_bbg, end_date_bbg)  #
            df_hist_prices = df_hist_prices.pivot_table(index='date', columns='ticker', values='value')  #
            df_hist_prices = df_hist_prices.dropna()  #

            if df_hist_prices.empty:  #
                print("No historical price data found for listed options/future.")  #
                return figure(width=700, height=350, title="No Data Found")  #

            # Calculate strategy price series
            strategy_px_series = np.sum(
                [np.array(df_hist_prices[option_tickers[i]]) * option_weights[i] for i in range(len(option_weights))],
                axis=0)  #
            strategy_px_series = np.round(100 * strategy_px_series, 1)  # Convert to bps and round

            # Prepare ColumnDataSource
            source = ColumnDataSource(data={  #
                'date': df_hist_prices.index,  #
                'strategy_px': strategy_px_series,  #
                'underlying_px': df_hist_prices[underlying_future_ticker]  #
            })  #

            # Create Bokeh figure
            h1 = figure(width=700, height=350,
                        tools=["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                        toolbar_location='right', x_axis_type='datetime')  #
            h1.xaxis.formatter = DatetimeTickFormatter(days="%d-%b-%y", months="%d-%b-%y")  #
            h1.add_tools(HoverTool(tooltips=[('Date', '@date{%d-%b-%y}'), ('Strategy Px', '@strategy_px{0.0}bps'),
                                             ('Underlying Px', '@underlying_px{0.00}')],
                                   formatters={'$x': 'datetime'}))  #
            h1.xgrid.visible = False  #
            h1.ygrid.visible = False  #

            # Plot strategy price
            h1.line(x='date', y='strategy_px', source=source, width=0.7, color='navy', legend_label="Strategy P&L",
                    alpha=1.0)  #

            h1.legend.location = 'top_right'  #
            h1.legend.label_text_font = "calibri"  #
            h1.legend.label_text_font_size = "9pt"  #
            h1.legend.spacing = 1  #
            h1.legend.click_policy = "mute"  #
            h1.legend.background_fill_alpha = 0.0  #
            h1.yaxis.axis_label = 'Strategy P&L (bps)'  #
            h1.xaxis.axis_label = 'Date'  #

            # Set y-range for strategy price
            h1.y_range = Range1d(strategy_px_series.min() - 2, strategy_px_series.max() + 2)  #

            # Add secondary Y-axis for underlying future price
            h1.extra_y_ranges = {
                "underlying_y_range": Range1d(start=df_hist_prices[underlying_future_ticker].min() - 0.05,
                                              end=df_hist_prices[underlying_future_ticker].max() + 0.05)}  #
            h1.add_layout(LinearAxis(y_range_name="underlying_y_range", axis_label="Underlying Price"), "right")  #
            h1.line(x='date', y='underlying_px', source=source, width=0.7, color='green',
                    legend_label="Underlying Future", alpha=0.2, y_range_name='underlying_y_range')  #

            return h1  #

        except Exception as e:
            raise RuntimeError(f"Error plotting listed options timeseries with Bokeh: {e}")  #

    def plot_inflation_fixings_bokeh(self,
                                     inflation_curve_builders: List[InflationCurveBuilder],  # Original `crv`
                                     is_generated_month_num: bool = False  # Original `gen`
                                     ) -> List[Tabs]:
        """
        Plots projected inflation fixings and their historical changes, with interactive historical data lookup.

        Args:
            inflation_curve_builders (List[InflationCurveBuilder]): List of `InflationCurveBuilder` objects
                                                                    (e.g., current and historical dates).
            is_generated_month_num (bool): If True, uses generated month number for historical x-axis.

        Returns:
            List[Tabs]: A list containing a Bokeh Tabs object for fixings plots and a separate historical plot.
        """
        if not inflation_curve_builders:  #
            print("No inflation curve builders provided.")  #
            return [Tabs(tabs=[TabPanel(child=figure(title="No Data"), title="Fixings")])]  #

        # Ensure curves are built
        for builder in inflation_curve_builders:  #
            builder.build_curve()  #

        # Get historical fixings from the first builder's `historical_fixings`
        # And ensure 'yoy' column is present for plotting

        # Populate 'yoy' in historical fixings if not present, as per original `INF_ZC_BUILD.py`
        for builder in inflation_curve_builders:  #
            if 'yoy' not in builder.historical_fixings.columns:  #
                builder.historical_fixings['yoy'] = builder.historical_fixings['index'].pct_change(periods=12) * 100  #
            if 'yoy' not in builder.curve[1].columns:  # market fixing curve
                builder.curve[1]['yoy'] = builder.curve[1]['index'].pct_change(periods=12) * 100  #
            if 'yoy' not in builder.curve[2].columns:  # forecast curve
                builder.curve[2]['yoy'] = builder.curve[2]['index'].pct_change(periods=12) * 100  #

        # Get the currency code for context
        inflation_index_name = inflation_curve_builders[0]._inflation_index_name  #

        # Extract projected fixings for the next 12 months from the market curve (index 1)
        # and barcap forecast curve (index 2).
        # Original `inf_swap_table` used `x_fixing = [crvs[0].last_pm + ql.Period(str(i)+"M") for i in np.arange(13)]`
        # Then it fetched 1y rates ending on these months.
        #
        # Here we extract direct yoy from the projected monthly series within the builder's curve.

        # This implies `curve[1]` (market fixings) and `curve[2]` (forecast) should have a 'yoy' column.
        # This is added above for now.

        # Get next 12 months from last market fixing month for comparison
        last_mkt_fixing_month_ql = inflation_curve_builders[0].last_market_fixing_month  #
        projected_months_ql = [last_mkt_fixing_month_ql + ql.Period(i, ql.Months) for i in
                               range(1, 13)]  # Next 12 months

        projected_months_labels = [ql_to_datetime(d).strftime('%b-%y') for d in projected_months_ql]  #

        # Get market fixings (from `curve[1]`)
        market_fixings_yoy = []  #
        for month_ql in projected_months_ql:  #
            # Find the month in the curve's projected data and get its YOY.
            # Assumes `curve[1]` has 'months' and 'yoy'.
            yoy_val = inflation_curve_builders[0].curve[1][inflation_curve_builders[0].curve[1]['months'] == month_ql][
                'yoy'].iloc[0]  #
            market_fixings_yoy.append(np.round(yoy_val, 2))  #

        # Get Barcap forecasts (from `curve[2]`)
        barcap_forecasts_yoy = []  #
        for month_ql in projected_months_ql:  #
            yoy_val = inflation_curve_builders[0].curve[2][inflation_curve_builders[0].curve[2]['months'] == month_ql][
                'yoy'].iloc[0]  #
            barcap_forecasts_yoy.append(np.round(yoy_val, 2))  #

        # Calculate changes in fixings if multiple builders provided
        fixings_change_yoy: List[float] = []  #
        if len(inflation_curve_builders) > 1:  #
            # Assuming `inflation_curve_builders[1]` is the historical comparison curve
            prev_market_fixings_yoy = []  #
            for month_ql in projected_months_ql:  #
                yoy_val = \
                inflation_curve_builders[1].curve[1][inflation_curve_builders[1].curve[1]['months'] == month_ql][
                    'yoy'].iloc[0]  #
                prev_market_fixings_yoy.append(np.round(yoy_val, 2))  #

            fixings_change_yoy = [100 * (mkt - prev_mkt) for mkt, prev_mkt in
                                  zip(market_fixings_yoy, prev_market_fixings_yoy)]  #
            fixings_change_yoy = np.round(fixings_change_yoy, 1).tolist()  #

        # Prepare ColumnDataSource
        s1_source = ColumnDataSource(data=dict(  #
            x=projected_months_labels,  #
            y_mkt=market_fixings_yoy,  #
            y_barx=barcap_forecasts_yoy,  #
            y_chg=fixings_change_yoy  #
        ))  #

        # --- Plot 1: Projected Fixings ---
        s1 = figure(x_range=projected_months_labels, width=550, height=300,  #
                    tools=["pan", "tap", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                    toolbar_location='right', title=f"{inflation_index_name} Projected Fixings")  #
        s1.xgrid.visible = False  #
        s1.ygrid.visible = False  #
        s1.vbar(x='x', top='y_mkt', width=0.7, source=s1_source, color='lightsteelblue',
                legend_label="Market Projection")  #
        s1.scatter(x='x', y='y_barx', source=s1_source, marker='circle', size=8, color='darkgreen',
                   legend_label="Barcap Forecast", alpha=0.7)  #

        # Y-range adjustment
        min_y_val = min(min(market_fixings_yoy), min(barcap_forecasts_yoy))  #
        max_y_val = max(max(market_fixings_yoy), max(barcap_forecasts_yoy))  #
        s1.y_range = Range1d(min_y_val - 0.5, max_y_val + 0.5)  #

        labels_mkt = LabelSet(x='x', y='y_mkt', text='y_mkt', level='glyph', text_align='center', y_offset=-10,
                              source=s1_source, text_font_size='10px', text_color='midnightblue')  #
        labels_barx = LabelSet(x='x', y='y_barx', text='y_barx', level='glyph', text_align='center', y_offset=8,
                               source=s1_source, text_font_size='10px', text_color='darkgreen')  #
        s1.add_layout((labels_mkt))  #
        s1.add_layout((labels_barx))  #
        s1.xaxis.major_label_orientation = np.pi / 2  #
        s1.yaxis.axis_label = 'Fixings (YoY %)'  #
        s1.legend.location = 'top_left'  #
        s1.legend.label_text_font = "calibri"  #
        s1.legend.label_text_font_size = "8pt"  #
        s1.legend.spacing = 1  #
        s1.legend.background_fill_alpha = 0.0  #
        s1.legend.click_policy = "mute"  #

        s1.add_tools(HoverTool(tooltips=[  #
            ("Month", "@x"),  #
            ("Market", "@y_mkt{0.00}%"),  #
            ("Barcap", "@y_barx{0.00}%")  #
        ]))  #

        # --- Plot 2: Fixings Change ---
        s2 = figure(x_range=projected_months_labels, width=550, height=125,  #
                    tools=["pan", "tap", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                    toolbar_location='right', title="1-Day Change in Fixings")  #
        s2.xgrid.visible = False  #
        s2.ygrid.visible = False  #
        s2.circle(x='x', y='y_chg', size=8, source=s1_source, color='firebrick', alpha=0.8)  #

        zero_line = Span(location=0, dimension='width', line_color='darkseagreen', line_width=1)  #
        s2.renderers.extend([zero_line])  #

        labels_chg = LabelSet(x='x', y='y_chg', text='y_chg', level='glyph', text_align='center', y_offset=-14,
                              source=s1_source, text_font_size='8px', text_color='firebrick')  #
        s2.add_layout((labels_chg))  #

        # Y-range adjustment for change plot
        min_chg = np.min(fixings_change_yoy) if fixings_change_yoy else 0  #
        max_chg = np.max(fixings_change_yoy) if fixings_change_yoy else 0  #
        s2.y_range = Range1d((min_chg * (1 - 0.2 * np.sign(min_chg))) - 2,
                             (max_chg * (1 + 0.2 * np.sign(max_chg))) + 2)  #

        s2.yaxis.axis_label = 'Change (bps)'  #
        s2.yaxis.major_label_text_font_size = '7pt'  #
        s2.xaxis.visible = False  #

        s2.add_tools(HoverTool(tooltips=[  #
            ("Month", "@x"),  #
            ("Change", "@y_chg{0.0}bps")  #
        ]))  #

        tabs_main_plots = Tabs(tabs=[TabPanel(child=layout([[s1], [s2]]), title="Projected Fixings")])  #

        # --- Historical Fixings Data Plot (Interactive) ---
        s3_hist = figure(width=570, height=450,  #
                         tools=["pan", "crosshair", "wheel_zoom", "box_zoom", "save", "reset", "help", "hover"],  #
                         toolbar_location='right', title="Historical Fixings Data (Click on bar to view)")  #
        s3_hist.xaxis.formatter = DatetimeTickFormatter(days="%d-%b-%y", months="%d-%b-%y")  #
        s3_hist.xgrid.visible = False  #
        s3_hist.ygrid.visible = False  #
        s3_hist.visible = False  # Hidden by default
        s3_hist.renderers.extend([zero_line])  #
        s3_hist.min_border_top = 30  #
        s3_hist.min_border_left = 50  #

        s3_hist.add_tools(HoverTool(tooltips=[  #
            ("Date", "@x{%d-%b-%y}"),  #
            ("Value", "@y{0.00}%")  #
        ], formatters={'$x': 'datetime'}))  #

        # --- Tap Callback for interactivity ---
        def _remove_glyphs_bokeh(figure_obj: figure, glyph_name_substring: str) -> None:  #
            renderers = figure_obj.select(dict(type=GlyphRenderer))  #
            for r in renderers:  #
                if glyph_name_substring in r.name:  #
                    r.data_source.data['x'] = []  #
                    r.data_source.data['y'] = []  #
            figure_obj.legend.items = []  #

        def fixings_callback(event: Tap) -> None:  #
            _remove_glyphs_bokeh(s3_hist, 'fix_plot')  #

            selected_indices = s1_source.selected.indices  #
            if len(selected_indices) == 1:  # Single selection for historical fixing plot
                selected_month_label = s1_source.data['x'][selected_indices[0]]  #

                df_hist_fix_filtered = pd.DataFrame()  #
                if is_generated_month_num:  #
                    # Original `fix = int(out_indicies[0]) + 1`
                    selected_month_num = selected_indices[0] + 1  #
                    df_hist_fix_filtered = inflation_curve_builders[0].historical_fixings[
                        inflation_curve_builders[0].historical_fixings['gen_month'] == selected_month_num].reset_index(
                        drop=True)  #
                else:  #
                    df_hist_fix_filtered = inflation_curve_builders[0].historical_fixings[
                        inflation_curve_builders[0].historical_fixings[
                            'fix_month2'] == selected_month_label].reset_index(drop=True)  #

                if not df_hist_fix_filtered.empty:  #
                    df_hist_fix_filtered = df_hist_fix_filtered.sort_values('date')  # Ensure sorted by date
                    source_hist = ColumnDataSource(data={  #
                        'x': df_hist_fix_filtered['date'],  #
                        'y': df_hist_fix_filtered['fixing']  #
                    })  #
                    s3_hist.line(x='x', y='y', source=source_hist, color='tomato',  #
                                 legend_label=f"Fixing for {selected_month_label}", alpha=1.0, muted_alpha=0.1,
                                 name='fix_plot')  #
                    s3_hist.visible = True  #
                    s3_hist.legend.location = 'bottom_left'  #
                    s3_hist.legend.label_text_font_size = "8pt"  #
                    s3_hist.legend.spacing = 1  #
                    s3_hist.legend.background_fill_alpha = 0.0  #
                    s3_hist.legend.click_policy = "mute"  #
                else:  #
                    s3_hist.visible = False  #
            elif len(selected_indices) == 2:  # Dual selection for spread history
                selected_month_labels = [s1_source.data['x'][idx] for idx in selected_indices]  #

                df_hist_fix_filtered_1 = pd.DataFrame()  #
                df_hist_fix_filtered_2 = pd.DataFrame()  #

                if is_generated_month_num:  #
                    month_num_1 = selected_indices[0] + 1  #
                    month_num_2 = selected_indices[1] + 1  #
                    df_hist_fix_filtered_1 = inflation_curve_builders[0].historical_fixings[
                        inflation_curve_builders[0].historical_fixings['gen_month'] == month_num_1]  #
                    df_hist_fix_filtered_2 = inflation_curve_builders[0].historical_fixings[
                        inflation_curve_builders[0].historical_fixings['gen_month'] == month_num_2]  #
                else:  #
                    df_hist_fix_filtered_1 = inflation_curve_builders[0].historical_fixings[
                        inflation_curve_builders[0].historical_fixings['fix_month2'] == selected_month_labels[0]]  #
                    df_hist_fix_filtered_2 = inflation_curve_builders[0].historical_fixings[
                        inflation_curve_builders[0].historical_fixings['fix_month2'] == selected_month_labels[1]]  #

                if not df_hist_fix_filtered_1.empty and not df_hist_fix_filtered_2.empty:  #
                    df_merged = pd.merge(df_hist_fix_filtered_1[['date', 'fixing']],
                                         df_hist_fix_filtered_2[['date', 'fixing']], on='date',
                                         suffixes=('_1', '_2'))  #
                    df_merged['sprd'] = 100 * (df_merged['fixing_1'] - df_merged['fixing_2'])  #
                    df_merged = df_merged.sort_values('date').dropna()  #

                    source_spread_hist = ColumnDataSource(data={  #
                        'x': df_merged['date'],  #
                        'y': df_merged['sprd']  #
                    })  #
                    s3_hist.line(x='x', y='y', source=source_spread_hist, color='darkgoldenrod',  #
                                 legend_label=f"Spread {selected_month_labels[0]} - {selected_month_labels[1]}",
                                 alpha=1.0, muted_alpha=0.1, name='fix_plot')  #
                    s3_hist.visible = True  #
                    s3_hist.legend.location = 'bottom_left'  #
                    s3_hist.legend.label_text_font_size = "8pt"  #
                    s3_hist.legend.spacing = 1  #
                    s3_hist.legend.background_fill_alpha = 0.0  #
                    s3_hist.legend.click_policy = "mute"  #
                else:  #
                    s3_hist.visible = False  #
            else:  #
                s3_hist.visible = False  #

        s1.on_event(Tap, fixings_callback)  #

        return [tabs_main_plots, s3_hist]  #

    def plot_swap_heatmap_bokeh(self,
                                curve_hmap_data: Dict[str, pd.DataFrame],
                                # Output from SwapTableGenerator.generate_swap_tables
                                change_offset_key: str = '-1',
                                # Key for the desired change offset (e.g., '-1' for 1-day change)
                                roll_keys: List[str] = ['3M', '6M'],  # Keys for the desired roll data
                                p_dim_width_base: int = 100  # Base width for each currency column in heatmap
                                ) -> layout:
        """
        Plots interactive heatmaps for yield curve rates, changes, forwards,
        curve steepness/spreads, and rolls using Bokeh.

        This function consolidates `curve_hm` and `rates_hm` from the original PLOT.py,
        and `swap_heatmap` from PLOT_BOKEH.py.

        Args:
            curve_hmap_data (Dict[str, pd.DataFrame]): Dictionary containing curve analysis DataFrames:
                                                      'rates', 'rates_chg', 'curves', 'chg', 'steep', 'steep_chg', 'roll'.
            change_offset_key (str): Key identifying the specific change offset.
            roll_keys (List[str]): List of keys for the specific roll data.
            p_dim_width_base (int): Base width to scale heatmap plots by number of currencies.

        Returns:
            layout: A Bokeh layout object containing the grid of heatmaps.
        """
        rates_df = curve_hmap_data['rates']  #
        rates_chg_dict = curve_hmap_data['rates_chg']  #
        curves_df = curve_hmap_data['curves']  #
        chg_dict = curve_hmap_data['chg']  #
        steep_df = curve_hmap_data['steep']  #
        steep_chg_dict = curve_hmap_data['steep_chg']  #
        roll_dict = curve_hmap_data['roll']  #

        n_currencies = rates_df.shape[1]  #

        # Filter out '11Y', '13Y', '40Y' as per original `rates_hm`
        filter_out_tenors = ['11Y', '13Y', '40Y']  #
        rates_df_filtered = rates_df.drop(index=[t for t in filter_out_tenors if t in rates_df.index],
                                          errors='ignore')  #

        rates_chg_df_filtered = rates_chg_dict.get(change_offset_key, pd.DataFrame()).drop(
            index=[t for t in filter_out_tenors if t in rates_chg_dict.get(change_offset_key, pd.DataFrame()).index],
            errors='ignore')  #

        # Common colormaps
        m_coolwarm_rgb = (255 * cm.coolwarm(range(256))).astype('int')  #
        coolwarm_palette = [RGB(*tuple(rgb)).to_hex() for rgb in m_coolwarm_rgb]  #

        m_purples_r_rgb = (255 * cm.Purples_r(range(256))).astype('int')  #
        purples_r_palette = [RGB(*tuple(rgb)).to_hex() for rgb in m_purples_r_rgb]  #

        # --- Data Preparation for Heatmaps ---
        # Rates and Changes
        df1_rates_source_data = rates_df_filtered.stack().reset_index()  #
        df1_rates_source_data.columns = ['Tenor', 'Curve', 'Rate']  #
        df1_rates_source_data['Chg'] = rates_chg_df_filtered.stack().reset_index()[0]  #
        s1_source = ColumnDataSource(df1_rates_source_data)  #

        # Forwards and Changes
        df2_fwds_source_data = curves_df.stack().reset_index()  #
        df2_fwds_source_data.columns = ['Fwds', 'Curve', 'Rate']  #
        df2_fwds_source_data['Chg'] = chg_dict.get(change_offset_key, pd.DataFrame()).stack().reset_index()[0]  #
        s2_source = ColumnDataSource(df2_fwds_source_data)  #

        # Steepness and Changes
        steep_df_T = steep_df.transpose()  #
        steep_chg_df_T = steep_chg_dict.get(change_offset_key, pd.DataFrame()).transpose()  #

        df3_steep_source_data = steep_df_T.stack().reset_index()  #
        df3_steep_source_data.columns = ['Curve', 'Steep', 'Spread']  #
        df3_steep_source_data['Chg'] = steep_chg_df_T.stack().reset_index()[0]  #
        s3_source = ColumnDataSource(df3_steep_source_data)  #

        # Rolls
        roll1_df_T = roll_dict.get(roll_keys[0], pd.DataFrame()).transpose()  #
        roll2_df_T = roll_dict.get(roll_keys[1], pd.DataFrame()).transpose()  #

        df4_roll_source_data = roll1_df_T.stack().reset_index()  #
        df4_roll_source_data.columns = ['Curve', 'Steep', 'Roll_1']  #
        df4_roll_source_data['Roll_2'] = roll2_df_T.stack().reset_index()[0]  #
        s4_source = ColumnDataSource(df4_roll_source_data)  #

        # Determine color mappers dynamically based on data ranges
        all_rates_vals = df1_rates_source_data['Rate'].tolist() + df2_fwds_source_data['Rate'].tolist()  #
        mapper_rates = LinearColorMapper(palette=coolwarm_palette, low=min(all_rates_vals), high=max(all_rates_vals))  #

        all_chg_vals = df1_rates_source_data['Chg'].tolist() + df2_fwds_source_data['Chg'].tolist()  #
        mapper_chg = LinearColorMapper(palette=purples_r_palette, low=min(all_chg_vals), high=max(all_chg_vals))  #

        all_steep_vals = df3_steep_source_data['Spread'].tolist()  #
        mapper_curve = LinearColorMapper(palette=coolwarm_palette, low=min(all_steep_vals), high=max(all_steep_vals))  #

        all_steep_chg_vals = df3_steep_source_data['Chg'].tolist()  #
        mapper_curve_chg = LinearColorMapper(palette=purples_r_palette, low=min(all_steep_chg_vals),
                                             high=max(all_steep_chg_vals))  #

        all_roll_vals = df4_roll_source_data['Roll_1'].tolist() + df4_roll_source_data['Roll_2'].tolist()  #
        mapper_roll = LinearColorMapper(palette=coolwarm_palette, low=min(all_roll_vals), high=max(all_roll_vals))  #

        # --- Heatmap Plotting ---

        # Plot 1: Rates
        s1 = figure(title="Rates", x_range=rates_df.columns.tolist(), y_range=rates_df_filtered.index.tolist()[::-1],
                    # Reverse y_range for plotting
                    x_axis_location="below", width=p_dim_width_base * n_currencies, height=400,
                    toolbar_location=None)  #
        s1.grid.grid_line_color = None  #
        s1.axis.axis_line_color = None  #
        s1.axis.major_tick_line_color = None  #
        s1.axis.major_label_text_font_size = "12px"  #
        s1.yaxis.axis_label = 'Tenor'  #

        s1.rect(x="Curve", y="Tenor", width=1, height=1, source=s1_source,  #
                fill_color={'field': 'Rate', 'transform': mapper_rates}, line_color=None)  #
        labels1 = LabelSet(x='Curve', y='Tenor', text='Rate', source=s1_source, level='glyph',  #
                           text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s1.add_layout(labels1)  #

        # Plot 2: Changes in Rates
        s2 = figure(title='Chg: ' + change_offset_key, x_range=rates_df.columns.tolist(),
                    y_range=rates_df_filtered.index.tolist()[::-1],  #
                    x_axis_location="below", y_axis_location="right", width=p_dim_width_base * n_currencies - 75,
                    height=400, toolbar_location=None)  #
        s2.grid.grid_line_color = None  #
        s2.axis.axis_line_color = None  #
        s2.axis.major_tick_line_color = None  #
        s2.axis.major_label_text_font_size = "12px"  #
        s2.yaxis.visible = False  #

        s2.rect(x="Curve", y="Tenor", width=1, height=1, source=s1_source,  #
                fill_color={'field': 'Chg', 'transform': mapper_chg}, line_color=None)  #
        labels2_chg = LabelSet(x='Curve', y='Tenor', text='Chg', source=s1_source, level='glyph',  #
                               text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s2.add_layout(labels2_chg)  #

        # Plot 3: Forwards
        s3 = figure(title="Forwards", x_range=rates_df.columns.tolist(), y_range=curves_df.index.tolist()[::-1],  #
                    x_axis_location="below", width=p_dim_width_base * n_currencies - 50, height=400,
                    toolbar_location=None)  #
        s3.grid.grid_line_color = None  #
        s3.axis.axis_line_color = None  #
        s3.axis.major_tick_line_color = None  #
        s3.axis.major_label_text_font_size = "12px"  #

        s3.rect(x="Curve", y="Fwds", width=1, height=1, source=s2_source,  #
                fill_color={'field': 'Rate', 'transform': mapper_rates}, line_color=None)  #
        labels3 = LabelSet(x='Curve', y='Fwds', text='Rate', source=s2_source, level='glyph',  #
                           text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s3.add_layout(labels3)  #

        # Plot 4: Changes in Forwards
        s4 = figure(title="Chg", x_range=rates_df.columns.tolist(), y_range=curves_df.index.tolist()[::-1],  #
                    x_axis_location="below", y_axis_location="right", width=p_dim_width_base * n_currencies - 75,
                    height=400, toolbar_location=None)  #
        s4.grid.grid_line_color = None  #
        s4.axis.axis_line_color = None  #
        s4.axis.major_tick_line_color = None  #
        s4.axis.major_label_text_font_size = "12px"  #
        s4.yaxis.visible = False  #

        s4.rect(x="Curve", y="Fwds", width=1, height=1, source=s2_source,  #
                fill_color={'field': 'Chg', 'transform': mapper_chg}, line_color=None)  #
        labels4_chg = LabelSet(x='Curve', y='Fwds', text='Chg', source=s2_source, level='glyph',  #
                               text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s4.add_layout(labels4_chg)  #

        # Plot 5: Curve Steepness
        s5 = figure(title="Curve", x_range=rates_df.columns.tolist(), y_range=steep_df.index.tolist()[::-1],  #
                    x_axis_location="below", width=p_dim_width_base * n_currencies, height=400,
                    toolbar_location=None)  #
        s5.grid.grid_line_color = None  #
        s5.axis.axis_line_color = None  #
        s5.axis.major_tick_line_color = None  #
        s5.axis.major_label_text_font_size = "12px"  #
        s5.yaxis.axis_label = 'Curve'  #

        s5.rect(x="Curve", y="Steep", width=1, height=1, source=s3_source,  #
                fill_color={'field': 'Spread', 'transform': mapper_curve}, line_color=None)  #
        labels5 = LabelSet(x='Curve', y='Steep', text='Spread', source=s3_source, level='glyph',  #
                           text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s5.add_layout(labels5)  #

        # Plot 6: Changes in Curve Steepness
        s6 = figure(title="Chg", x_range=rates_df.columns.tolist(), y_range=steep_df.index.tolist()[::-1],  #
                    x_axis_location="below", y_axis_location="right", width=p_dim_width_base * n_currencies - 75,
                    height=400, toolbar_location=None)  #
        s6.grid.grid_line_color = None  #
        s6.axis.axis_line_color = None  #
        s6.axis.major_tick_line_color = None  #
        s6.axis.major_label_text_font_size = "12px"  #
        s6.yaxis.visible = False  #

        s6.rect(x="Curve", y="Steep", width=1, height=1, source=s3_source,  #
                fill_color={'field': 'Chg', 'transform': mapper_curve_chg}, line_color=None)  #
        labels6_chg = LabelSet(x='Curve', y='Steep', text='Chg', source=s3_source, level='glyph',  #
                               text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s6.add_layout(labels6_chg)  #

        # Plot 7: Roll 1
        s7 = figure(title=f'Roll: {roll_keys[0]}', x_range=rates_df.columns.tolist(),
                    y_range=steep_df.index.tolist()[::-1],  #
                    x_axis_location="below", width=p_dim_width_base * n_currencies - 50, height=400,
                    toolbar_location=None)  #
        s7.grid.grid_line_color = None  #
        s7.axis.axis_line_color = None  #
        s7.axis.major_tick_line_color = None  #
        s7.axis.major_label_text_font_size = "12px"  #

        s7.rect(x="Curve", y="Steep", width=1, height=1, source=s4_source,  #
                fill_color={'field': 'Roll_1', 'transform': mapper_roll}, line_color=None)  #
        labels7_roll1 = LabelSet(x='Curve', y='Steep', text='Roll_1', source=s4_source, level='glyph',  #
                                 text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s7.add_layout(labels7_roll1)  #

        # Plot 8: Roll 2
        s8 = figure(title=f'Roll: {roll_keys[1]}', x_range=rates_df.columns.tolist(),
                    y_range=steep_df.index.tolist()[::-1],  #
                    x_axis_location="below", y_axis_location="right", width=p_dim_width_base * n_currencies - 75,
                    height=400, toolbar_location=None)  #
        s8.grid.grid_line_color = None  #
        s8.axis.axis_line_color = None  #
        s8.axis.major_tick_line_color = None  #
        s8.axis.major_label_text_font_size = "12px"  #
        s8.yaxis.visible = False  #

        s8.rect(x="Curve", y="Steep", width=1, height=1, source=s4_source,  #
                fill_color={'field': 'Roll_2', 'transform': mapper_roll}, line_color=None)  #
        labels8_roll2 = LabelSet(x='Curve', y='Steep', text='Roll_2', source=s4_source, level='glyph',  #
                                 text_align='center', y_offset=-7, text_color='black', text_font_size='9pt')  #
        s8.add_layout(labels8_roll2)  #

        p = layout(children=[[s1, s2, s3, s4], [s5, s6, s7, s8]])  #
        return p  #