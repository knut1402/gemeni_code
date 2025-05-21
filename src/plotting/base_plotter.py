# src/plotting/base_plotter.py

from abc import ABC, abstractmethod
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, Optional, Tuple, List, Union


class BasePlotter(ABC):
    """
    Abstract base class for all plotting functionalities.

    Defines common plotting configurations, such as matplotlib rcParams,
    and provides helper methods for setting up plot layouts.
    """

    def __init__(self):
        """
        Initializes the BasePlotter and applies default matplotlib settings.
        """
        self._apply_default_plot_settings()

    def _apply_default_plot_settings(self) -> None:
        """
        Applies a set of default matplotlib rcParams for consistent plot aesthetics.
        """
        # Reset to default first to ensure a clean slate, then apply custom settings
        mpl.rcParams.update(mpl.rcParamsDefault)

        # Apply common settings observed in original PLOT.py and PLOT_BOKEH.py
        mpl.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.labelpad'] = 10.0  # Adjusted from original's high values for better default aesthetics

    @abstractmethod
    def plot(self, data: Any, **kwargs) -> Any:
        """
        Abstract method to generate a plot. Subclasses must implement this.

        Args:
            data (Any): The data to be plotted.
            **kwargs: Additional plotting parameters.

        Returns:
            Any: The plot object (e.g., matplotlib Figure or Axes, Bokeh plot).
        """
        pass

    def _setup_subplots(self, num_plots: int, height_ratios: Optional[List[float]] = None,
                        figsize: Tuple[float, float] = (12, 8), hspace: float = 0.0) -> Tuple[
        plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Helper method to set up matplotlib subplots.

        Args:
            num_plots (int): The number of subplots to create.
            height_ratios (Optional[List[float]]): List of height ratios for subplots.
            figsize (Tuple[float, float]): Figure size (width, height) in inches.
            hspace (float): Height space between subplots.

        Returns:
            Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]: The matplotlib Figure and Axes object(s).
        """
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, axs = plt.subplots(num_plots, 1, figsize=figsize,
                                    gridspec_kw={'height_ratios': height_ratios, 'hspace': hspace})
            ax = axs  # Assign axs to ax for consistent return type usage in subclasses if only one subplot expected.

        return fig, ax

    def _add_grid_and_labels(self, ax: plt.Axes, x_label: str = '', y_label: str = '', title: str = '') -> None:
        """
        Helper method to add grid, labels, and title to a matplotlib Axes object.

        Args:
            ax (plt.Axes): The matplotlib Axes object.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            title (str): Title for the plot.
        """
        ax.grid(True, 'major', 'both', linestyle=':')
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)