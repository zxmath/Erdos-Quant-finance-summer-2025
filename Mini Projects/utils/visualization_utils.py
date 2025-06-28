"""
Visualization Utilities for Financial Analysis
=============================================

This module provides plotting and visualization functions
for portfolio analysis and normality testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_portfolio_comparison(portfolio_metrics_dict, title="Portfolio Comparison"):
    """
    Create comparison plots for multiple portfolios
    
    Parameters:
    -----------
    portfolio_metrics_dict : dict
        Dictionary of {portfolio_name: metrics_dict}
    title : str
        Title for the plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    portfolio_names = list(portfolio_metrics_dict.keys())
    
    # Risk vs Return scatter plot
    returns = [portfolio_metrics_dict[name]['Annual Return (%)'] for name in portfolio_names]
    volatilities = [portfolio_metrics_dict[name]['Annual Volatility (%)'] for name in portfolio_names]
    sharpe_ratios = [portfolio_metrics_dict[name]['Sharpe Ratio'] for name in portfolio_names]
    
    scatter = axes[0, 0].scatter(volatilities, returns, c=sharpe_ratios, s=200, alpha=0.7, cmap='RdYlGn')
    axes[0, 0].set_xlabel('Annual Volatility (%)')
    axes[0, 0].set_ylabel('Annual Return (%)')
    axes[0, 0].set_title('Risk vs Return (Color = Sharpe Ratio)')
    
    # Add portfolio labels
    for i, name in enumerate(portfolio_names):
        axes[0, 0].annotate(name, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.colorbar(scatter, ax=axes[0, 0], label='Sharpe Ratio')
    
    # Risk metrics bar chart
    risk_metrics = ['VaR 95% (%)', 'Max Drawdown (%)']
    x = np.arange(len(portfolio_names))
    width = 0.35
    
    for i, metric in enumerate(risk_metrics):
        values = [abs(portfolio_metrics_dict[name][metric]) for name in portfolio_names]
        axes[0, 1].bar(x + i * width, values, width, label=metric, alpha=0.7)
    
    axes[0, 1].set_xlabel('Portfolio')
    axes[0, 1].set_ylabel('Risk Metric (%)')
    axes[0, 1].set_title('Risk Metrics Comparison')
    axes[0, 1].set_xticks(x + width / 2)
    axes[0, 1].set_xticklabels(portfolio_names, rotation=45)
    axes[0, 1].legend()
    
    # Return distribution comparison
    for name in portfolio_names:
        if 'Portfolio Returns' in portfolio_metrics_dict[name]:
            returns_data = portfolio_metrics_dict[name]['Portfolio Returns']
            axes[1, 0].hist(returns_data * 100, bins=50, alpha=0.6, label=name, density=True)
    
    axes[1, 0].set_xlabel('Daily Returns (%)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Return Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics radar chart (simplified bar chart)
    metrics = ['Annual Return (%)', 'Sharpe Ratio', 'Sortino Ratio']
    metric_values = []
    
    for metric in metrics:
        values = []
        for name in portfolio_names:
            if metric in portfolio_metrics_dict[name]:
                val = portfolio_metrics_dict[name][metric]
                # Normalize for better visualization
                if metric == 'Annual Return (%)':
                    val = val / 10  # Scale to 0-2 range roughly
                values.append(val)
            else:
                values.append(0)
        metric_values.append(values)
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, name in enumerate(portfolio_names):
        values = [metric_values[j][i] for j in range(len(metrics))]
        axes[1, 1].bar(x + i * width, values, width, label=name, alpha=0.7)
    
    axes[1, 1].set_xlabel('Performance Metrics')
    axes[1, 1].set_ylabel('Normalized Values')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_xticks(x + width / 2)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_normality_results(test_results, returns_data=None):
    """
    Create comprehensive normality test visualization
    
    Parameters:
    -----------
    test_results : dict
        Results from comprehensive_normality_test
    returns_data : array-like, optional
        Original returns data for plotting
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Normality Analysis: {test_results["name"]}', fontsize=16, fontweight='bold')
    
    if returns_data is not None:
        # Clean data
        clean_data = np.array(returns_data).flatten()
        clean_data = clean_data[~np.isnan(clean_data)]
        
        # Histogram with normal overlay
        axes[0, 0].hist(clean_data, bins=50, density=True, alpha=0.7, color='skyblue', label='Data')
        
        # Overlay normal distribution
        mu, sigma = np.mean(clean_data), np.std(clean_data)
        x = np.linspace(clean_data.min(), clean_data.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        axes[0, 0].plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        
        axes[0, 0].set_xlabel('Returns')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Distribution vs Normal')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(clean_data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series plot (if data has time structure)
        if hasattr(returns_data, 'index') and len(returns_data.index) == len(clean_data):
            axes[1, 0].plot(returns_data.index, clean_data, alpha=0.7, linewidth=0.8)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Returns')
            axes[1, 0].set_title('Time Series of Returns')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # Just plot as sequence
            axes[1, 0].plot(clean_data, alpha=0.7, linewidth=0.8)
            axes[1, 0].set_xlabel('Observation')
            axes[1, 0].set_ylabel('Returns')
            axes[1, 0].set_title('Returns Sequence')
            axes[1, 0].grid(True, alpha=0.3)
    
    # Test results summary
    test_names = []
    p_values = []
    normal_status = []
    
    for test_name, test_result in test_results['tests'].items():
        if isinstance(test_result, dict) and 'normal' in test_result:
            test_names.append(test_name)
            p_values.append(test_result.get('p_value', 0))
            normal_status.append(test_result['normal'])
    
    # Bar chart of p-values
    colors = ['green' if normal else 'red' for normal in normal_status]
    bars = axes[1, 1].bar(range(len(test_names)), p_values, color=colors, alpha=0.7)
    
    # Add significance line
    axes[1, 1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    
    axes[1, 1].set_xlabel('Statistical Tests')
    axes[1, 1].set_ylabel('p-value')
    axes[1, 1].set_title('Normality Test Results')
    axes[1, 1].set_xticks(range(len(test_names)))
    axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add text annotations for statistics
    stats_text = f"""Sample Size: {test_results['sample_size']}
Mean: {test_results['mean']:.6f}
Std Dev: {test_results['std']:.6f}
Skewness: {test_results['skewness']:.3f}
Kurtosis: {test_results['kurtosis']:.3f}

Tests Passed: {test_results['consensus']['normal_tests']}/{test_results['consensus']['total_tests']}
Consensus: {'NORMAL' if test_results['consensus']['consensus_normal'] else 'NOT NORMAL'}"""
    
    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_rolling_normality(returns_data, window_sizes=None, ticker="Stock"):
    """
    Plot rolling normality test results over different window sizes
    
    Parameters:
    -----------
    returns_data : pandas.Series
        Returns data with datetime index
    window_sizes : list, optional
        List of rolling window sizes to test
    ticker : str
        Name of the stock/asset
    """
    if window_sizes is None:
        window_sizes = [30, 60, 90, 180, 252]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Rolling Normality Analysis: {ticker}', fontsize=16, fontweight='bold')
    
    # Calculate rolling statistics
    rolling_stats = pd.DataFrame(index=returns_data.index)
    
    for window in window_sizes:
        rolling_stats[f'Mean_{window}d'] = returns_data.rolling(window).mean()
        rolling_stats[f'Std_{window}d'] = returns_data.rolling(window).std()
        rolling_stats[f'Skew_{window}d'] = returns_data.rolling(window).skew()
        rolling_stats[f'Kurt_{window}d'] = returns_data.rolling(window).kurt()
    
    # Plot rolling mean and std
    for window in window_sizes:
        axes[0, 0].plot(rolling_stats.index, rolling_stats[f'Mean_{window}d'], 
                       label=f'{window}d window', alpha=0.7)
    axes[0, 0].set_title('Rolling Mean')
    axes[0, 0].set_ylabel('Mean Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for window in window_sizes:
        axes[0, 1].plot(rolling_stats.index, rolling_stats[f'Std_{window}d'], 
                       label=f'{window}d window', alpha=0.7)
    axes[0, 1].set_title('Rolling Standard Deviation')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot rolling skewness
    for window in window_sizes:
        axes[1, 0].plot(rolling_stats.index, rolling_stats[f'Skew_{window}d'], 
                       label=f'{window}d window', alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Rolling Skewness')
    axes[1, 0].set_ylabel('Skewness')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot rolling kurtosis
    for window in window_sizes:
        axes[1, 1].plot(rolling_stats.index, rolling_stats[f'Kurt_{window}d'], 
                       label=f'{window}d window', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Rolling Kurtosis')
    axes[1, 1].set_ylabel('Excess Kurtosis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_outlier_impact(outlier_results):
    """
    Plot the impact of outlier removal on normality
    
    Parameters:
    -----------
    outlier_results : pandas.DataFrame
        Results from test_outlier_impact function
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Impact of Outlier Removal on Normality', fontsize=16, fontweight='bold')
    
    # CRITICAL FIX: Sort data by percentage removed to avoid connecting lines inappropriately
    sorted_results = outlier_results.sort_values('Data Removed (%)')
    
    # Plot normality strength vs data removed (sorted) - scatter plot without lines
    axes[0, 0].scatter(sorted_results['Data Removed (%)'], sorted_results['Strength'], 
                      s=100, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Data Removed (%)')
    axes[0, 0].set_ylabel('Normality Strength')
    axes[0, 0].set_title('Normality Strength vs Outlier Removal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot number of normal tests passed (use original order for categorical data)
    axes[0, 1].bar(range(len(outlier_results)), outlier_results['Normal Tests Passed'], 
                  alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Outlier Threshold')
    axes[0, 1].set_ylabel('Normal Tests Passed')
    axes[0, 1].set_title('Tests Passed vs Outlier Threshold')
    axes[0, 1].set_xticks(range(len(outlier_results)))
    axes[0, 1].set_xticklabels(outlier_results['Outlier Threshold'], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot skewness and kurtosis changes (sorted by data removed) - scatter plots without lines
    axes[1, 0].scatter(sorted_results['Data Removed (%)'], sorted_results['Skewness'], 
                      s=80, alpha=0.7, color='red')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Data Removed (%)')
    axes[1, 0].set_ylabel('Skewness')
    axes[1, 0].set_title('Skewness vs Outlier Removal')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(sorted_results['Data Removed (%)'], sorted_results['Kurtosis'], 
                      s=80, alpha=0.7, color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Data Removed (%)')
    axes[1, 1].set_ylabel('Excess Kurtosis')
    axes[1, 1].set_title('Kurtosis vs Outlier Removal')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(correlation_matrix, title="Correlation Matrix"):
    """
    Create a correlation heatmap with annotations
    
    Parameters:
    -----------
    correlation_matrix : pandas.DataFrame
        Correlation matrix to plot
    title : str
        Title for the plot
    """
    # Don't create a new figure - let the calling code handle figure management
    
    # Create heatmap without masking - show full correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f',
                vmin=-1, vmax=1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
