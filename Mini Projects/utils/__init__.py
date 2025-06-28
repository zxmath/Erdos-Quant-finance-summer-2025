"""
Utility modules for Quantitative Finance Mini Projects
=====================================================

This package contains utility functions for:
- Data downloading and processing
- Portfolio metrics and risk analysis  
- Statistical normality testing
- Data visualization

Modules:
- data_utils: Financial data downloading and processing
- portfolio_metrics: Portfolio analysis and risk calculations
- normality_tests: Statistical testing for normal distributions
- visualization_utils: Plotting and visualization functions
"""

__version__ = "1.0.0"
__author__ = "Quant Finance Summer 2025"

# Import main utility functions for easy access
try:
    from .data_utils import download_real_data, process_returns_data, save_portfolio_data, load_portfolio_data
    from .portfolio_metrics import calculate_var_95, calculate_portfolio_metrics, calculate_individual_stock_metrics
    from .normality_tests import comprehensive_normality_test, test_time_periods, test_outlier_impact
    from .visualization_utils import plot_portfolio_comparison, plot_normality_results
    
    __all__ = [
        'download_real_data',
        'process_returns_data',
        'save_portfolio_data', 
        'load_portfolio_data',
        'calculate_var_95',
        'calculate_portfolio_metrics',
        'calculate_individual_stock_metrics',
        'comprehensive_normality_test',
        'test_time_periods',
        'test_outlier_impact',
        'plot_portfolio_comparison',
        'plot_normality_results'
    ]
except ImportError as e:
    print(f"Warning: Some utility functions may not be available due to missing dependencies: {e}")
    __all__ = []
