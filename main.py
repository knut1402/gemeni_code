# main.py

import datetime
import QuantLib as ql
import pandas as pd
import numpy as np
from bokeh.io import show
from bokeh.layouts import column, row

# Import core modules
from src.config import config  # Global configuration object
from src.data_manager import b_con, data_loader  # Bloomberg connection and historical data loader
from src.quantlib_utils import setup_ql_settings  # For setting QuantLib evaluation date
from src.date_utils import ql_to_datetime  # To convert QL dates for display

# Import curve builders
from src.curves.ois_curve_builder import OISCurveBuilder
from src.curves.swap_curve_builder import SwapCurveBuilder
from src.curves.inflation_curve_builder import InflationCurveBuilder
from src.curves.bond_curve_builder import BondCurveBuilder  # Placeholder

# Import instruments
from src.instruments.swap import Swap, SwapParameters
from src.instruments.bond import Bond
from src.instruments.option import Option

# Import pricers
from src.pricers.swap_pricer import SwapPricer
from src.pricers.inflation_pricer import InflationPricer
from src.pricers.linker_carry_calculator import LinkerCarryCalculator
from src.pricers.option_pricer import OptionPricer


# Import volatility surface builders
from src.volatility_surface.bond_future_vol_surface import BondFutureVolSurface
from src.volatility_surface.stir_vol_surface import StirVolSurface

# Import strategy analytics
from src.strategy_analytics.option_strategy import OptionStrategy

# Import market tables
from src.market_tables.bond_tables import LinkerTableGenerator
from src.market_tables.inflation_tables import InflationTableGenerator
from src.market_tables.swap_tables import SwapTableGenerator  # Placeholder

# Import data processors
from src.market_data_processors.economic_data_processor import EconomicDataProcessor
from src.market_data_processors.wirp_processor import WirpProcessor

# Import plotting modules
from src.plotting.matplotlib_plots import MatplotlibPlotter
from src.plotting.bokeh_plots import BokehPlotter


def run_example_workflow():
    """
    Demonstrates key functionalities of the refactored financial infrastructure.
    """
    print("--- Starting Example Workflow ---")

    # 1. Initialize QuantLib Settings
    print("\n1. Initializing QuantLib Settings...")
    setup_ql_settings(datetime.datetime(2024, 5, 20))  # Set a specific valuation date
    current_ql_date = ql.Settings.instance().evaluationDate
    print(f"QuantLib Evaluation Date: {ql_to_datetime(current_ql_date).strftime('%Y-%m-%d')}")

    # 2. Build and Inspect an OIS Curve
    print("\n2. Building and Inspecting an OIS Curve (SOFR)...")
    try:
        sofr_builder = OISCurveBuilder('USD', current_ql_date)  # 'USD' is currency code
        sofr_builder.build_curve()
        print(f"SOFR Curve Built. Reference Date: {ql_to_datetime(sofr_builder.reference_date).strftime('%Y-%m-%d')}")
        print("\nSOFR Market Instrument Table:")
        print(sofr_builder.market_instrument_table)
        print("\nSOFR Curve Nodes (first 5):")
        # Ensure QuantLib curve nodes are accessible and convertible
        if sofr_builder.curve and hasattr(sofr_builder.curve, 'nodes'):
            nodes = sofr_builder.curve.nodes()
            for date, discount in nodes[:5]:
                print(f"  Date: {ql_to_datetime(date).strftime('%Y-%m-%d')}, Discount Factor: {discount:.6f}")
        else:
            print("  Curve nodes not available or not standard QuantLib type.")

    except Exception as e:
        print(f"Error building SOFR OIS curve: {e}")

    # 3. Build and Inspect a Swap Curve (USD_3M)
    print("\n3. Building and Inspecting a Swap Curve (USD_3M)...")
    try:
        usd3m_builder = SwapCurveBuilder('USD', current_ql_date)  # 'USD' currency
        usd3m_builder.build_curve()
        print(
            f"USD_3M Swap Curve Built. Reference Date: {ql_to_datetime(usd3m_builder.reference_date).strftime('%Y-%m-%d')}")
        print("\nUSD_3M Swap Rates Table:")
        print(usd3m_builder.market_data.get('swap_rates_df', 'N/A'))
    except Exception as e:
        print(f"Error building USD_3M swap curve: {e}")

    # 4. Price a Swap
    print("\n4. Pricing a Swap (5Y USD Spot Starting)...")
    try:
        swap_pricer = SwapPricer()
        usd_5y_swap_params = SwapParameters(
            currency_code='USD',
            start_tenor=0,  # Spot starting
            maturity_tenor=5,  # 5 years
            notional=1_000_000  # 1 million USD
        )
        swap_metrics = swap_pricer.calculate_swap_metrics([usd_5y_swap_params])
        print("\n5Y USD Swap Metrics:")
        print(swap_metrics['table'])
    except Exception as e:
        print(f"Error pricing USD 5Y swap: {e}")

    # 5. Build and Inspect an Inflation Curve (USCPI)
    print("\n5. Building and Inspecting an Inflation Curve (USCPI)...")
    try:
        uscpi_builder = InflationCurveBuilder('USCPI', current_ql_date)
        uscpi_builder.build_curve()
        print(
            f"USCPI Inflation Curve Built. Reference Date: {ql_to_datetime(uscpi_builder.reference_date).strftime('%Y-%m-%d')}")
        print("\nUSCPI Inflation Swap Rates:")
        print(uscpi_builder.inflation_zc_rates)
        print("\nUSCPI Historical Fixings (last 5):")
        print(uscpi_builder.historical_fixings.tail())
    except Exception as e:
        print(f"Error building USCPI inflation curve: {e}")

    # 6. Generate a Linker Table
    print("\n6. Generating a Linker Table (Example)...")
    try:
        linker_generator = LinkerTableGenerator()
        # Dummy bond_db DataFrame for demonstration
        dummy_bond_db = pd.DataFrame({
            'linker_isin': ['US912810PU60 Govt'],  # Example ISIN
            'compar_isin': ['US91282CGP08 Govt'],  # Example comparable nominal bond
            'linker': ['US TII 2.375 08/2042'],
            'nominal': ['USD'],
            'country': ['US'],
            'index': ['USCPI']  # Associated inflation index
        })
        repo_rate = 0.001  # 0.1%

        linker_table_df = linker_generator.generate_linker_table(
            bond_db=dummy_bond_db,
            repo_rate=repo_rate,
            reference_date_input=current_ql_date,
            change_reference_date_input=current_ql_date - ql.Period(1, ql.Days),  # 1 day prior
            fixing_curve_type='Market'
        )
        print("\nLinker Monitor Table:")
        print(linker_table_df)
    except Exception as e:
        print(f"Error generating linker table: {e}")

    # 7. Analyze an Option Strategy (Bond Future Butterfly)
    print("\n7. Analyzing an Option Strategy (Bond Future Butterfly)...")
    try:
        # NOTE: Using a placeholder ticker for bond futures (USU1 Comdty)
        # Ensure it exists in your Bloomberg setup and has option chain data.
        bond_future_ticker = 'USU1 Comdty'  #
        option_types = ['C', 'C', 'C']
        strikes = [165.0, 168.0, 171.0]
        weights = [1.0, -2.0, 1.0]  # Butterfly

        bond_vol_surface_builder = BondFutureVolSurface('USD', bond_future_ticker, current_ql_date)

        butterfly_strategy = OptionStrategy(
            currency_code='USD',
            underlying_ticker=bond_future_ticker,
            option_types=option_types,
            strikes=strikes,
            weights=weights,
            strategy_name="USU1 Butterfly",
            vol_surface_builder=bond_vol_surface_builder
        )

        strategy_summary = butterfly_strategy.strategy_details
        print(f"\nStrategy: {strategy_summary['strategy_name']}")
        print(f"Current P&L: {strategy_summary['strategy_current_px']:.4f}")
        print("\nStrategy Simulation (first 5 rows):")
        print(strategy_summary['strategy_simulation_df'].head())
    except Exception as e:
        print(f"Error analyzing option strategy: {e}")

    # 8. Plot an OIS Curve using Matplotlib
    print("\n8. Plotting an OIS Curve (SOFR) using Matplotlib...")
    try:
        mpl_plotter = MatplotlibPlotter()
        # Plot current SOFR curve and SOFR curve from 30 days ago
        sofr_builder_today = OISCurveBuilder('USD', current_ql_date)
        sofr_builder_today.build_curve()
        sofr_builder_30d_ago = OISCurveBuilder('USD', current_ql_date - ql.Period(30, ql.Days))
        sofr_builder_30d_ago.build_curve()

        fig = mpl_plotter.plot(
            plot_type='swap_curve',
            data=[sofr_builder_today, sofr_builder_30d_ago],
            max_tenor_years=10,
            bar_changes=True,
            plot_title="SOFR OIS Curve & 30-Day Change"
        )
        # fig.show() # Uncomment to display plot if running in interactive environment
        print("Matplotlib SOFR OIS curve plot generated.")
    except Exception as e:
        print(f"Error plotting Matplotlib OIS curve: {e}")

    # 9. Plot an Inflation Curve using Bokeh
    print("\n9. Plotting an Inflation Curve (UKRPI) using Bokeh...")
    try:
        bokeh_plotter = BokehPlotter()
        ukrpi_builder_today = InflationCurveBuilder('UKRPI', current_ql_date)
        ukrpi_builder_today.build_curve()
        ukrpi_builder_90d_ago = InflationCurveBuilder('UKRPI', current_ql_date - ql.Period(90, ql.Days))
        ukrpi_builder_90d_ago.build_curve()

        bokeh_plots = bokeh_plotter.plot(
            plot_type='inflation_curve',
            data=[ukrpi_builder_today, ukrpi_builder_90d_ago],
            max_tenor_years=10,
            bar_changes=True,
            plot_title="UKRPI Inflation Curve & 90-Day Change"
        )
        # show(column(*bokeh_plots)) # Uncomment to display plot
        print("Bokeh UKRPI Inflation curve plot generated.")
    except Exception as e:
        print(f"Error plotting Bokeh Inflation curve: {e}")

    # 10. Run Economic Data Processor
    print("\n10. Running Economic Data Processor (US CPI YoY)...")
    try:
        eco_processor = EconomicDataProcessor()
        us_cpi_data = eco_processor.get_economic_data(
            country='US',
            start_date='20000101',
            category1='Prices',
            category2='CPI',
            zscore_period=20  # 20 years for Z-score
        )
        print("\nUS CPI Data (Z-Score Pct Change, last 5 rows):")
        if 'zs_pct' in us_cpi_data:
            print(us_cpi_data['zs_pct'].tail())
        else:
            print("  Z-score data not generated.")
    except Exception as e:
        print(f"Error processing economic data: {e}")

    # 11. Update and Plot WIRP History
    print("\n11. Updating and Plotting WIRP History (USD)...")
    try:
        wirp_processor = WirpProcessor()
        # Update history (will fetch new data from Bloomberg and save to pickle)
        updated_wirp_hist = wirp_processor.update_wirp_history('USD', write_to_disk=True, update_mode=True)
        print("\nUSD WIRP History (last 5 rows):")
        if 'USD' in updated_wirp_hist:
            print(updated_wirp_hist['USD'].tail())
        else:
            print("  USD WIRP history not updated or found.")

        # Plot single day WIRP for USD
        wirp_data_today = wirp_processor.get_wirp_data(
            currency_codes='USD',
            valuation_dates=[ql_to_datetime(current_ql_date)]
        )
        bokeh_wirp_plots = bokeh_plotter.plot(
            plot_type='simple_wirp',
            data=wirp_data_today
        )
        # show(column(*bokeh_wirp_plots)) # Uncomment to display plot
        print("Bokeh USD WIRP plot generated.")

    except Exception as e:
        print(f"Error updating or plotting WIRP history: {e}")

    # 12. Close Bloomberg Connection
    print("\n12. Closing Bloomberg Connection...")
    try:
        b_con.close()
        print("Bloomberg connection closed.")
    except Exception as e:
        print(f"Error closing Bloomberg connection: {e}")

    print("\nAttempting to price a single option using OptionPricer...")
    try:
        option_pricer = OptionPricer()
        single_option_metrics = option_pricer.get_option_price_and_greeks(
            currency_code='USD',
            option_type='C',
            strike_price=170.0,
            expiry_date_input='2024-09-20',  # Example expiry
            underlying_ticker='USU1 Comdty',
            implied_volatility_pct=15.0  # Example vol
        )
        print("Single Option Metrics:", single_option_metrics)
    except Exception as e:
        print(f"Error pricing single option: {e}")

    print("\n--- Example Workflow Completed ---")


if __name__ == "__main__":
    # Ensure Bloomberg is running and you have data_lake populated with some .pkl files
    # (e.g., SOFR_DC_OIS_MEETING_HIST.pkl, HICPxT_fixing_hist.pkl etc. if available)
    # and eco_master.xlsx in data/data_lake/

    # You might need to run the `update_wirp_history` and `update_inflation_fixing_history`
    # functions a few times with `write_to_disk=True` in a separate script or interactively
    # to populate your data_lake before running this main example.

    run_example_workflow()