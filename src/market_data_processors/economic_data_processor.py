# src/market_data_processors/economic_data_processor.py

import pandas as pd
import numpy as np
import datetime
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
from typing import Any, Dict, List, Optional, Union
from src.data_manager import b_con, data_loader  #
from src.date_utils import to_bbg_date_str, ql_to_datetime  #
from src.config import config  #


class EconomicDataProcessor:
    """
    Processes economic data by fetching it from Bloomberg, applying various transformations
    (e.g., YoY changes, ROC, Z-scores), and preparing it for further analysis or visualization.
    """

    def __init__(self):
        """
        Initializes the EconomicDataProcessor.
        Loads the economic data master file from the data lake.
        """
        self._eco_master_df = data_loader.load_excel(
            config.general_settings.get('eco_master_file', 'eco_master.xlsx'),  #
            sheet_name="EcoData"  #
        )
        if self._eco_master_df.empty:
            raise RuntimeError("Economic data master file could not be loaded or is empty.")  #

    def get_economic_data(self,
                          country: str,
                          start_date: str,  # YYYYMMDD format
                          end_date: Union[str, int] = -1,  # YYYYMMDD or -1 for today
                          category1: str = "all",  # 'all' or specific Cat1
                          category2: str = "all",  # 'all' or specific Cat2
                          change_period: int = 6,  # Period for diff()
                          roc_period: int = 3,  # Period for Rate of Change (diff of diff)
                          zscore_period: int = 0  # Period in years for Z-score calculation
                          ) -> Dict[str, pd.DataFrame]:
        """
        Fetches and processes economic time series data from Bloomberg.

        Applies transformations like year-over-year change, rate of change, and z-scores.

        Args:
            country (str): Country code for filtering economic data (e.g., 'AU', 'US').
            start_date (str): Start date for data fetching in 'YYYYMMDD' format.
            end_date (Union[str, int]): End date for data fetching in 'YYYYMMDD' format,
                                        or -1 for today's date.
            category1 (str): Filter by 'Cat1' in the economic master data ('all' for no filter).
            category2 (str): Filter by 'Cat2' in the economic master data ('all' for no filter).
            change_period (int): Number of periods for calculating the difference (e.g., 6 for 6-month change).
            roc_period (int): Number of periods for calculating Rate of Change (difference of `change_period` diff).
            zscore_period (int): Period in years for calculating Z-scores. 0 means no Z-score.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing various transformed DataFrames:
                                     'raw', 'df_pct' (YoY % Change), 'df_chg' (Raw Change),
                                     'df_roc' (Raw ROC), 'df_pct_roc' (YoY % ROC),
                                     'zs_raw', 'zs_pct', 'zs_chg', 'zs_roc', 'zs_pct_roc'.
        """
        filtered_eco_df = self._eco_master_df[self._eco_master_df['Country'] == country]  #
        if category1 != "all":  #
            filtered_eco_df = filtered_eco_df[filtered_eco_df['Cat1'] == category1]  #
        if category2 != "all":  #
            filtered_eco_df = filtered_eco_df[filtered_eco_df['Cat2'] == category2]  #

        if filtered_eco_df.empty:
            print(f"No economic data found for Country: {country}, Cat1: {category1}, Cat2: {category2}.")  #
            return {}  #

        tickers = filtered_eco_df['Ticker'].tolist()  #
        labels = filtered_eco_df['Label'].tolist()  #

        # Determine end date for Bloomberg query
        end_date_str = to_bbg_date_str(datetime.datetime.now(), ql_date=0) if end_date == -1 else str(end_date)  #

        # Determine start date for DataFrame index
        start_dt_obj = datetime.datetime.strptime(start_date, '%Y%m%d')  #
        today_dt_obj = datetime.datetime.now()  #

        # Initialize raw DataFrame with monthly frequency
        raw_df = pd.DataFrame(columns=labels, index=pd.date_range(start=start_dt_obj, end=today_dt_obj, freq='M'))  #
        raw_df.name = "Raw"  #

        # Fetch data for all tickers
        # Using a single bdh call for efficiency, then pivoting
        try:
            all_data_long = b_con.bdh(tickers, 'PX_LAST', start_date, end_date_str, longdata=True)  #
            if all_data_long.empty:
                print(f"No Bloomberg data returned for tickers: {tickers}")  #
                return {}  #

            # Process monthly data
            monthly_tickers = filtered_eco_df[filtered_eco_df['Freq'] == 'M']['Ticker'].tolist()  #
            monthly_data_pivot = all_data_long[all_data_long['ticker'].isin(monthly_tickers)].pivot_table(
                index='date', columns='ticker', values='value'
            )  #
            raw_df.update(monthly_data_pivot.rename(
                columns=filtered_eco_df.set_index('Ticker')['Label']))  # Update using Label as column name

            # Process weekly data (resample to monthly last)
            weekly_tickers = filtered_eco_df[filtered_eco_df['Freq'] == 'W']['Ticker'].tolist()  #
            if weekly_tickers:  #
                weekly_data_pivot = all_data_long[all_data_long['ticker'].isin(weekly_tickers)].pivot_table(
                    index='date', columns='ticker', values='value'
                )  #
                weekly_data_resampled = weekly_data_pivot.resample('M').last()  #
                raw_df.update(weekly_data_resampled.rename(columns=filtered_eco_df.set_index('Ticker')['Label']))  #

        except Exception as e:
            print(f"Error fetching or processing Bloomberg data: {e}")  #
            return {}  #

        # Drop rows that are entirely NaN after merging/resampling for cleaner analysis start
        raw_df = raw_df.dropna(how='all')  #

        # --- NaN Summary ---
        nan_summary_df = pd.DataFrame(index=labels)  #
        nan_summary_df['#'] = raw_df.isna().sum()  #
        nan_summary_df['First'] = raw_df.apply(lambda x: x.first_valid_index())  #
        nan_summary_df['Last'] = raw_df.apply(lambda x: x.last_valid_index())  #

        # --- YoY Percentage Change ---
        pct_change_df = raw_df.copy()  #
        pct_change_df.name = "Pct"  #

        is_index_labels = filtered_eco_df[filtered_eco_df['IsIndex'] == 1]['Label'].tolist()  #
        for label in is_index_labels:  #
            if label in pct_change_df.columns:  #
                pct_change_df[label] = np.round(100 * raw_df[label].pct_change(periods=12), 2)  #

        # --- Raw Change and Rate of Change (ROC) ---
        raw_change_df = raw_df.diff(periods=change_period)  #
        raw_change_df.name = "Chg"  #

        raw_roc_df = raw_change_df.diff(periods=roc_period)  #
        raw_roc_df.name = "ROC"  #

        pct_roc_df = pct_change_df.diff(periods=roc_period)  #
        pct_roc_df.name = "Pct_ROC"  #

        # --- Z-Scores ---
        output_data = {
            'raw': raw_df,
            'nan_summary': nan_summary_df,
            'df_pct': pct_change_df,
            'df_chg': raw_change_df,
            'df_roc': raw_roc_df,
            'df_pct_roc': pct_roc_df
        }

        if zscore_period > 0:  #
            zscore_start_idx = -12 * zscore_period  #

            # Labels to exclude from Z-score calculation (based on ZS_excl = 1)
            zscore_exclude_labels = filtered_eco_df[filtered_eco_df['ZS_excl'] == 1]['Label'].tolist()  #

            # Z-score calculation for each DataFrame, excluding specified labels
            # NOTE: Original comment: `####### check this z-score calc to confirm row wise operation + understand how we can make it row wise first/last non-nan`
            # `scipy.stats.zscore` by default computes column-wise. `nan_policy='omit'` handles NaNs.
            # To ensure proper Z-scoring for potentially sparse data, it's typically done column-wise.
            # If row-wise behavior is desired (unusual for time series), axis=1 would be used.
            # Given the original code's structure, column-wise (default) is likely intended.

            zscore_dfs = {}
            for df_name, df in output_data.items():
                if '_df' in df_name:  # Only apply to DataFrames containing time series data
                    z_df = df.iloc[zscore_start_idx:].copy()  # Apply Z-score only to the relevant period

                    for col in z_df.columns:  #
                        if col not in zscore_exclude_labels:  #
                            z_df[col] = zscore(z_df[col], nan_policy='omit')  #

                    z_df['Avg'] = z_df.drop(columns=zscore_exclude_labels, errors='ignore').mean(
                        axis=1)  # Exclude specified labels for average
                    zscore_dfs[f"zs_{df_name.replace('df_', '')}"] = z_df  #

            output_data.update(zscore_dfs)  #

        return output_data