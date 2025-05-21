# src/market_tables/bond_tables.py

import pandas as pd
import datetime
import numpy as np
import QuantLib as ql
from typing import Any, Dict, List, Optional, Union
from src.instruments.bond import Bond  #
from src.pricers.linker_carry_calculator import LinkerCarryCalculator  #
from src.date_utils import to_bbg_date_str, datetime_to_ql  #
from src.config import config  #
from src.data_manager import b_con  #


class LinkerTableGenerator:
    """
    Generates a comprehensive monitor table for inflation-linked bonds (linkers).

    This table includes various metrics such as current price, yield, nominal yield,
    Breakeven Inflation (BEI), and carry components.
    """

    def __init__(self):
        """
        Initializes the LinkerTableGenerator.
        """
        self.carry_calculator = LinkerCarryCalculator()  #

    def generate_linker_table(self,
                              bond_db: pd.DataFrame,
                              # Original `bond_db` input, assumed to have 'linker_isin', 'compar_isin', 'linker', 'nominal', 'country', 'index'
                              repo_rate: float,
                              reference_date_input: Union[str, int, datetime.datetime, ql.Date] = 0,  # Original `b`
                              change_reference_date_input: Union[str, int, datetime.datetime, ql.Date] = -1,
                              # Original `chg`
                              country_filter: str = '',
                              index_filter: str = '',
                              fixing_curve_type: str = 'BARX',  # Original `fixing_curve`
                              forward_dates: Optional[List[Union[str, datetime.datetime, ql.Date]]] = None
                              # Original `fwd_date`
                              ) -> pd.DataFrame:
        """
        Generates a linker monitor table with current and historical yield data,
        BEI, and carry.

        Args:
            bond_db (pd.DataFrame): DataFrame containing bond metadata (e.g., ISINs, nominals).
                                    Expected columns: 'linker_isin', 'compar_isin', 'linker', 'nominal',
                                    'country', 'index'.
            repo_rate (float): The repo rate to use for carry calculations (as a decimal).
            reference_date_input (Union[str, int, datetime.datetime, ql.Date]):
                The primary reference date for the table.
            change_reference_date_input (Union[str, int, datetime.datetime, ql.Date]):
                The historical reference date for calculating changes (e.g., 1-day change).
            country_filter (str): Optional filter for bond 'country'.
            index_filter (str): Optional filter for bond 'index'.
            fixing_curve_type (str): Type of inflation fixing curve to use for carry.
                                     Options: 'BARX', 'Market', 'Seasonals'.
            forward_dates (Optional[List[Union[str, datetime.datetime, ql.Date]]]):
                Optional list of dates for forward carry calculations.

        Returns:
            pd.DataFrame: A DataFrame representing the linker monitor table.
        """
        # Filter bond_db based on country and index
        filtered_bond_db = bond_db.reset_index(drop=True)
        if country_filter:  #
            filtered_bond_db = filtered_bond_db[filtered_bond_db['country'] == country_filter].reset_index(drop=True)
        if index_filter:  #
            filtered_bond_db = filtered_bond_db[filtered_bond_db['index'] == index_filter].reset_index(drop=True)

        if filtered_bond_db.empty:
            print("No bonds found after filtering.")
            return pd.DataFrame()

        # Set QuantLib evaluation date for the primary reference date
        ql.Settings.instance().evaluationDate = datetime_to_ql(datetime.datetime.now())  #
        current_ref_date_ql = ql.Settings.instance().evaluationDate

        # Determine reference dates for Bloomberg fetches
        ref_date_current_ql = current_ref_date_ql  # This is the "today" in original, then adjusted by `b`
        if isinstance(reference_date_input, int):
            ref_date_current_ql = ql.Settings.instance().evaluationDate + ql.Period(reference_date_input, ql.Days)
        else:
            ref_date_current_ql = datetime_to_ql(reference_date_input) if isinstance(reference_date_input, (
            datetime.datetime, ql.Date)) else current_ref_date_ql  #

        ref_date_change_ql = ref_date_current_ql  # This is the "ref_date" in original, then adjusted by `chg`
        if isinstance(change_reference_date_input, int):
            ref_date_change_ql = ref_date_current_ql + ql.Period(change_reference_date_input, ql.Days)
        else:
            ref_date_change_ql = datetime_to_ql(change_reference_date_input) if isinstance(change_reference_date_input,
                                                                                           (datetime.datetime,
                                                                                            ql.Date)) else ref_date_current_ql  #

        # Format dates for Bloomberg API calls
        bbg_date_current = to_bbg_date_str(ref_date_current_ql, ql_date=1)
        bbg_date_change = to_bbg_date_str(ref_date_change_ql, ql_date=1)

        linker_monitor_df = pd.DataFrame()
        linker_monitor_df['linker_isin'] = filtered_bond_db['linker_isin']
        linker_monitor_df['compar_isin'] = filtered_bond_db['compar_isin']
        linker_monitor_df['Linker'] = filtered_bond_db['linker']
        linker_monitor_df['Nominal'] = filtered_bond_db['nominal']

        unique_isins = linker_monitor_df['linker_isin'].tolist() + linker_monitor_df['compar_isin'].tolist()
        unique_isins = list(set(unique_isins))  # Ensure unique tickers for bulk fetch

        # Fetch Maturity for all linkers
        try:
            maturity_data = b_con.ref(linker_monitor_df['linker_isin'].tolist(), ['MATURITY'])
            maturity_dict = maturity_data.set_index('ticker')['value'].to_dict()  #
            linker_monitor_df['Maturity'] = linker_monitor_df['linker_isin'].map(maturity_dict)
        except Exception as e:
            print(f"Error fetching bond maturities: {e}")
            linker_monitor_df['Maturity'] = np.nan

        # Fetch current and historical prices/yields for linkers and nominals
        fields_current = ['PX_LAST', 'YLD_YTM_MID']
        fields_change = ['YLD_YTM_MID']

        # Use bdh for date-specific fetches
        current_data = b_con.bdh(unique_isins, fields_current, bbg_date_current, bbg_date_current, longdata=True)
        change_data = b_con.bdh(unique_isins, fields_change, bbg_date_change, bbg_date_change, longdata=True)

        # Pivot to make fields columns
        current_data_pivot = current_data.pivot_table(index='ticker', columns='field', values='value')
        change_data_pivot = change_data.pivot_table(index='ticker', columns='field', values='value')

        # Map data back to linker_monitor_df
        linker_monitor_df['Px'] = linker_monitor_df['linker_isin'].map(current_data_pivot['PX_LAST'])
        linker_monitor_df['Yield'] = linker_monitor_df['linker_isin'].map(current_data_pivot['YLD_YTM_MID'])
        linker_monitor_df['Nom_Yld'] = linker_monitor_df['compar_isin'].map(current_data_pivot['YLD_YTM_MID'])
        linker_monitor_df['Yld_1'] = linker_monitor_df['linker_isin'].map(change_data_pivot['YLD_YTM_MID'])
        linker_monitor_df['Nom_Yld_1'] = linker_monitor_df['compar_isin'].map(change_data_pivot['YLD_YTM_MID'])

        # Calculate Carry for each linker
        carry_results: Dict[str, List[float]] = {}
        forward_carry_cols: List[str] = []

        for idx, row in linker_monitor_df.iterrows():
            linker_isin = row['linker_isin']
            inflation_index = row['index']  # Assuming 'index' column in bond_db holds inflation index name

            # Call the new LinkerCarryCalculator
            carry_components = self.carry_calculator.calculate_linker_carry(
                isin=linker_isin,
                inflation_index_name=inflation_index,
                currency_code=row['Nominal'],  # Assuming 'Nominal' holds currency code (e.g., 'USD', 'EUR')
                repo_rate=repo_rate,
                fixing_curve_type=fixing_curve_type,
                forward_dates=forward_dates
            )

            # Store carry components for this linker
            for key, val in carry_components.items():
                if key not in carry_results:  #
                    carry_results[key] = []  #
                carry_results[key].append(val)  #

            # Collect forward carry column names for the first linker (assuming consistent for all)
            if not forward_carry_cols:  #
                forward_carry_cols = [k for k in carry_components.keys() if k.startswith('Carry_')]  #

        # Add carry columns to linker_monitor_df
        linker_monitor_df['Carry'] = carry_results.get('Carry', [np.nan] * len(linker_monitor_df))
        for col in forward_carry_cols:
            linker_monitor_df[col] = carry_results.get(col, [np.nan] * len(linker_monitor_df))

        # Calculate derived metrics
        linker_monitor_df['Δ_adj'] = 100 * (linker_monitor_df['Yield'] - linker_monitor_df['Yld_1']) - \
                                     linker_monitor_df['Carry']
        linker_monitor_df['Δ_Nom'] = 100 * (linker_monitor_df['Nom_Yld'] - linker_monitor_df['Nom_Yld_1'])
        linker_monitor_df['BEI'] = 100 * (linker_monitor_df['Nom_Yld'] - linker_monitor_df['Yield']) + \
                                   linker_monitor_df['Carry']
        linker_monitor_df['Δ_BEI_adj'] = linker_monitor_df['Δ_Nom'] - linker_monitor_df['Δ_adj']

        # Rounding as per original code
        linker_monitor_df['Yield'] = linker_monitor_df['Yield'].round(3)
        linker_monitor_df['Δ_adj'] = linker_monitor_df['Δ_adj'].round(1)
        linker_monitor_df['BEI'] = linker_monitor_df['BEI'].round(1)
        linker_monitor_df['Nom_Yld'] = linker_monitor_df['Nom_Yld'].round(3)
        linker_monitor_df['Δ_Nom'] = linker_monitor_df['Δ_Nom'].round(1)
        linker_monitor_df['Δ_BEI_adj'] = linker_monitor_df['Δ_BEI_adj'].round(1)
        for col in ['Carry'] + forward_carry_cols:
            linker_monitor_df[col] = linker_monitor_df[col].round(1)

        # Reorder columns as per original output
        final_cols = ['Linker', 'Maturity', 'Px', 'Yield', 'Δ_adj', 'BEI', 'Δ_BEI_adj', 'Nom_Yld', 'Δ_Nom',
                      'Nominal'] + ['Carry'] + forward_carry_cols
        return linker_monitor_df[final_cols]