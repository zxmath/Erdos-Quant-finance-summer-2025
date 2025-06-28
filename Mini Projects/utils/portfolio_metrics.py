"""
Portfolio Metrics and Risk Analysis
==================================

This module provides functions for calculating portfolio metrics,
risk measures, and performance analytics.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


def calculate_var_95(returns):
    """
    Calculate Value at Risk at 95% confidence level
    
    Parameters:
    -----------
    returns : array-like
        Return data (daily returns)
        
    Returns:
    --------
    float
        VaR at 95% confidence level (as percentage)
    """
    sorted_returns = np.sort(returns)
    index = int(0.05 * len(sorted_returns))
    return sorted_returns[index] * 100


def calculate_cvar_95(returns):
    """
    Calculate Conditional Value at Risk (Expected Shortfall) at 95% confidence
    
    Parameters:
    -----------
    returns : array-like
        Return data (daily returns)
        
    Returns:
    --------
    float
        CVaR at 95% confidence level (as percentage)
    """
    sorted_returns = np.sort(returns)
    index = int(0.05 * len(sorted_returns))
    return sorted_returns[:index].mean() * 100


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from returns
    
    Parameters:
    -----------
    returns : array-like
        Return data (daily returns)
        
    Returns:
    --------
    float
        Maximum drawdown as percentage
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return drawdowns.min() * 100


def calculate_portfolio_metrics(returns, weights=None, risk_free_rate=0.05):
    """
    Calculate comprehensive portfolio metrics
    
    Parameters:
    -----------
    returns : pandas.DataFrame or pandas.Series
        Return data (daily returns)
    weights : array-like, optional
        Portfolio weights (default: equal weighting)
    risk_free_rate : float
        Annual risk-free rate (default: 5%)
        
    Returns:
    --------
    dict
        Dictionary containing portfolio metrics
    """
    # Handle Series input
    if isinstance(returns, pd.Series):
        portfolio_returns = returns
    else:
        # Handle DataFrame with weights
        if weights is None:
            weights = np.array([1/len(returns.columns)] * len(returns.columns))
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
    
    # Annualized metrics (252 trading days)
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # Risk metrics
    var_95 = calculate_var_95(portfolio_returns)
    cvar_95 = calculate_cvar_95(portfolio_returns)
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    
    # Additional metrics
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis()
    
    # Sortino ratio (downside deviation)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if len(downside_returns) > 0 else np.inf
    
    return {
        'Annual Return (%)': annual_return * 100,
        'Annual Volatility (%)': annual_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'VaR 95% (%)': var_95,
        'CVaR 95% (%)': cvar_95,
        'Max Drawdown (%)': max_drawdown,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Portfolio Returns': portfolio_returns
    }


def calculate_individual_stock_metrics(returns_data, risk_free_rate=0.05):
    """
    Calculate metrics for individual stocks in a portfolio
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        Returns data with stocks as columns
    risk_free_rate : float
        Annual risk-free rate (default: 5%)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metrics for each stock
    """
    individual_metrics = {}
    
    for ticker in returns_data.columns:
        returns = returns_data[ticker].dropna()
        
        if len(returns) > 10:  # Ensure sufficient data
            # Basic metrics
            annual_return = returns.mean() * 252 * 100
            annual_vol = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))
            
            # Risk metrics
            var_95 = calculate_var_95(returns)
            max_dd = calculate_max_drawdown(returns)
            
            individual_metrics[ticker] = {
                'Annual Return (%)': annual_return,
                'Annual Volatility (%)': annual_vol,
                'Sharpe Ratio': sharpe,
                'VaR 95% (%)': var_95,
                'Max Drawdown (%)': max_dd,
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis()
            }
    
    return pd.DataFrame(individual_metrics).T


def optimize_portfolio_weights(returns_data, objective='sharpe', risk_free_rate=0.05):
    """
    Optimize portfolio weights based on different objectives
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        Returns data with stocks as columns
    objective : str
        Optimization objective ('sharpe', 'min_variance', 'max_return')
    risk_free_rate : float
        Annual risk-free rate
        
    Returns:
    --------
    dict
        Optimized weights and portfolio metrics
    """
    n_assets = len(returns_data.columns)
    
    # Calculate expected returns and covariance matrix
    mean_returns = returns_data.mean() * 252
    cov_matrix = returns_data.cov() * 252
    
    # Objective functions
    def portfolio_return(weights):
        return np.sum(mean_returns * weights)
    
    def portfolio_variance(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def negative_sharpe(weights):
        p_return = portfolio_return(weights)
        p_variance = portfolio_variance(weights)
        return -(p_return - risk_free_rate) / p_variance
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess (equal weights)
    x0 = np.array([1/n_assets] * n_assets)
    
    # Optimize based on objective
    if objective == 'sharpe':
        result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    elif objective == 'min_variance':
        result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    elif objective == 'max_return':
        result = minimize(lambda x: -portfolio_return(x), x0, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        raise ValueError("Objective must be 'sharpe', 'min_variance', or 'max_return'")
    
    if result.success:
        optimal_weights = result.x
        
        # Calculate metrics with optimal weights
        portfolio_metrics = calculate_portfolio_metrics(returns_data, optimal_weights, risk_free_rate)
        
        return {
            'weights': optimal_weights,
            'tickers': returns_data.columns.tolist(),
            'metrics': portfolio_metrics,
            'success': True
        }
    else:
        return {
            'weights': x0,
            'tickers': returns_data.columns.tolist(),
            'metrics': calculate_portfolio_metrics(returns_data, x0, risk_free_rate),
            'success': False,
            'message': 'Optimization failed'
        }


def calculate_correlation_matrix(returns_data):
    """
    Calculate and analyze correlation matrix
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        Returns data with stocks as columns
        
    Returns:
    --------
    dict
        Correlation analysis results
    """
    corr_matrix = returns_data.corr()
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)
    
    correlations = upper_triangle.stack()
    
    return {
        'correlation_matrix': corr_matrix,
        'mean_correlation': correlations.mean(),
        'max_correlation': correlations.max(),
        'min_correlation': correlations.min(),
        'std_correlation': correlations.std(),
        'correlations': correlations
    }


def calculate_beta(stock_returns, market_returns):
    """
    Calculate stock beta relative to market
    
    Parameters:
    -----------
    stock_returns : pandas.Series
        Individual stock returns
    market_returns : pandas.Series
        Market returns (e.g., S&P 500)
        
    Returns:
    --------
    float
        Beta coefficient
    """
    # Align the data
    aligned_data = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned_data) < 30:  # Need sufficient data
        return np.nan
    
    # Calculate covariance and variance
    covariance = np.cov(aligned_data['stock'], aligned_data['market'])[0, 1]
    market_variance = np.var(aligned_data['market'])
    
    return covariance / market_variance if market_variance != 0 else np.nan


def rolling_portfolio_metrics(returns_data, window=252, weights=None):
    """
    Calculate rolling portfolio metrics over time
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        Returns data with stocks as columns
    window : int
        Rolling window size in days (default: 252 = 1 year)
    weights : array-like, optional
        Portfolio weights
        
    Returns:
    --------
    pandas.DataFrame
        Rolling metrics over time
    """
    if weights is None:
        weights = np.array([1/len(returns_data.columns)] * len(returns_data.columns))
    
    # Calculate portfolio returns
    portfolio_returns = (returns_data * weights).sum(axis=1)
    
    # Rolling calculations
    rolling_return = portfolio_returns.rolling(window).mean() * 252 * 100
    rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
    rolling_sharpe = (rolling_return - 5) / rolling_vol  # Assuming 5% risk-free rate
    
    rolling_metrics = pd.DataFrame({
        'Rolling Return (%)': rolling_return,
        'Rolling Volatility (%)': rolling_vol,
        'Rolling Sharpe': rolling_sharpe
    }, index=returns_data.index)
    
    return rolling_metrics.dropna()
