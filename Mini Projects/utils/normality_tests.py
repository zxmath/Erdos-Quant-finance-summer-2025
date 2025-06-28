"""
Statistical Normality Testing Framework
======================================

This module provides comprehensive normality testing functions
for financial returns analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest, jarque_bera, anderson
import warnings

warnings.filterwarnings('ignore')


def comprehensive_normality_test(data, name="Dataset", alpha=0.05):
    """
    Perform comprehensive normality testing using multiple statistical tests
    
    Parameters:
    -----------
    data : array-like
        Data to test for normality
    name : str
        Name of the dataset for reporting
    alpha : float
        Significance level (default 0.05)
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # Remove NaN values
    clean_data = np.array(data).flatten()
    clean_data = clean_data[~np.isnan(clean_data)]
    
    if len(clean_data) < 3:
        return {
            'name': name,
            'sample_size': len(clean_data),
            'error': 'Insufficient data for testing'
        }
    
    results = {
        'name': name,
        'sample_size': len(clean_data),
        'mean': np.mean(clean_data),
        'std': np.std(clean_data),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data),
        'tests': {}
    }
    
    # 1. Shapiro-Wilk Test (best for small samples)
    if len(clean_data) <= 5000:  # Shapiro-Wilk limit
        try:
            shapiro_stat, shapiro_p = shapiro(clean_data)
            results['tests']['Shapiro-Wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'normal': shapiro_p >= alpha,
                'description': 'Most powerful test for small samples (<5000)'
            }
        except Exception as e:
            results['tests']['Shapiro-Wilk'] = {'error': str(e)}
    
    # 2. Kolmogorov-Smirnov Test
    try:
        # Test against standard normal distribution
        standardized_data = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        ks_stat, ks_p = kstest(standardized_data, 'norm')
        results['tests']['Kolmogorov-Smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'normal': ks_p >= alpha,
            'description': 'Good for larger samples, tests against normal CDF'
        }
    except Exception as e:
        results['tests']['Kolmogorov-Smirnov'] = {'error': str(e)}
    
    # 3. Jarque-Bera Test
    try:
        jb_stat, jb_p = jarque_bera(clean_data)
        results['tests']['Jarque-Bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'normal': jb_p >= alpha,
            'description': 'Tests normality through skewness and kurtosis'
        }
    except Exception as e:
        results['tests']['Jarque-Bera'] = {'error': str(e)}
    
    # 4. Anderson-Darling Test
    try:
        ad_result = anderson(clean_data, dist='norm')
        # Use 5% significance level (index 2)
        critical_value = ad_result.critical_values[2]
        significance_level = ad_result.significance_levels[2]
        
        results['tests']['Anderson-Darling'] = {
            'statistic': ad_result.statistic,
            'critical_value': critical_value,
            'normal': ad_result.statistic < critical_value,
            'significance_level': significance_level,
            'description': 'More sensitive to deviations in tails'
        }
    except Exception as e:
        results['tests']['Anderson-Darling'] = {'error': str(e)}
    
    # 5. D'Agostino's Normality Test
    try:
        if len(clean_data) >= 20:  # Minimum sample size for this test
            dag_stat, dag_p = normaltest(clean_data)
            results['tests']["D'Agostino"] = {
                'statistic': dag_stat,
                'p_value': dag_p,
                'normal': dag_p >= alpha,
                'description': 'Combines skewness and kurtosis tests'
            }
    except Exception as e:
        results['tests']["D'Agostino"] = {'error': str(e)}
    
    # Calculate consensus
    normal_count = sum(1 for test in results['tests'].values() 
                      if isinstance(test, dict) and test.get('normal', False))
    total_tests = len([test for test in results['tests'].values() 
                      if isinstance(test, dict) and 'normal' in test])
    
    results['consensus'] = {
        'normal_tests': normal_count,
        'total_tests': total_tests,
        'consensus_normal': normal_count > total_tests / 2 if total_tests > 0 else False,
        'strength': normal_count / total_tests if total_tests > 0 else 0
    }
    
    return results


def test_time_periods(returns_data, periods=None, test_name="Stock"):
    """
    Test normality across different time periods
    
    Parameters:
    -----------
    returns_data : pandas.Series or pandas.DataFrame
        Returns data with datetime index
    periods : list, optional
        List of period lengths in days to test
    test_name : str
        Name for the test series
        
    Returns:
    --------
    pandas.DataFrame
        Results of normality tests across different periods
    """
    if periods is None:
        periods = [30, 60, 90, 180, 252, 504]  # Default periods
    
    # Handle DataFrame input (take first column)
    if isinstance(returns_data, pd.DataFrame):
        returns_data = returns_data.iloc[:, 0]
    
    results = []
    
    for period in periods:
        if len(returns_data) >= period:
            # Test most recent period
            recent_data = returns_data.iloc[-period:]
            test_result = comprehensive_normality_test(
                recent_data, 
                name=f"{test_name} ({period} days)"
            )
            
            # Extract key metrics
            row = {
                'Period (days)': period,
                'Sample Size': test_result['sample_size'],
                'Mean': test_result['mean'],
                'Std Dev': test_result['std'],
                'Skewness': test_result['skewness'],
                'Kurtosis': test_result['kurtosis'],
                'Normal Tests Passed': test_result['consensus']['normal_tests'],
                'Total Tests': test_result['consensus']['total_tests'],
                'Consensus Normal': test_result['consensus']['consensus_normal'],
                'Strength': test_result['consensus']['strength']
            }
            
            # Add individual test results
            for test_name_inner, test_result_inner in test_result['tests'].items():
                if isinstance(test_result_inner, dict) and 'normal' in test_result_inner:
                    row[f'{test_name_inner} Normal'] = test_result_inner['normal']
                    row[f'{test_name_inner} p-value'] = test_result_inner.get('p_value', np.nan)
            
            results.append(row)
    
    return pd.DataFrame(results)


def test_outlier_impact(returns_data, outlier_thresholds=None, test_name="Dataset"):
    """
    Test impact of removing outliers on normality
    
    Parameters:
    -----------
    returns_data : array-like
        Returns data to test
    outlier_thresholds : list, optional
        List of standard deviation thresholds for outlier removal
    test_name : str
        Name for the test
        
    Returns:
    --------
    pandas.DataFrame
        Results showing impact of outlier removal
    """
    if outlier_thresholds is None:
        outlier_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5]
    
    # Convert to numpy array
    data = np.array(returns_data).flatten()
    data = data[~np.isnan(data)]
    
    results = []
    
    # Test original data
    original_result = comprehensive_normality_test(data, f"{test_name} (Original)")
    results.append({
        'Outlier Threshold': 'None',
        'Data Removed (%)': 0,
        'Sample Size': len(data),
        'Normal Tests Passed': original_result['consensus']['normal_tests'],
        'Total Tests': original_result['consensus']['total_tests'],
        'Consensus Normal': original_result['consensus']['consensus_normal'],
        'Strength': original_result['consensus']['strength'],
        'Skewness': original_result['skewness'],
        'Kurtosis': original_result['kurtosis']
    })
    
    # Test with different outlier removal thresholds
    for threshold in outlier_thresholds:
        # Remove outliers beyond threshold standard deviations
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        filtered_data = data[z_scores <= threshold]
        
        removed_pct = (1 - len(filtered_data) / len(data)) * 100
        
        if len(filtered_data) >= 10:  # Ensure sufficient data remains
            filtered_result = comprehensive_normality_test(
                filtered_data, 
                f"{test_name} (±{threshold}σ)"
            )
            
            results.append({
                'Outlier Threshold': f'±{threshold}σ',
                'Data Removed (%)': removed_pct,
                'Sample Size': len(filtered_data),
                'Normal Tests Passed': filtered_result['consensus']['normal_tests'],
                'Total Tests': filtered_result['consensus']['total_tests'],
                'Consensus Normal': filtered_result['consensus']['consensus_normal'],
                'Strength': filtered_result['consensus']['strength'],
                'Skewness': filtered_result['skewness'],
                'Kurtosis': filtered_result['kurtosis']
            })
    
    return pd.DataFrame(results)


def find_normal_stocks(returns_data_dict, min_normal_tests=3):
    """
    Find stocks with the most evidence of normal returns
    
    Parameters:
    -----------
    returns_data_dict : dict
        Dictionary of {ticker: returns_data}
    min_normal_tests : int
        Minimum number of normality tests that must pass
        
    Returns:
    --------
    pandas.DataFrame
        Ranked list of stocks by normality evidence
    """
    results = []
    
    for ticker, returns_data in returns_data_dict.items():
        if len(returns_data) > 30:  # Ensure sufficient data
            test_result = comprehensive_normality_test(returns_data, ticker)
            
            results.append({
                'Ticker': ticker,
                'Sample Size': test_result['sample_size'],
                'Normal Tests Passed': test_result['consensus']['normal_tests'],
                'Total Tests': test_result['consensus']['total_tests'],
                'Consensus Normal': test_result['consensus']['consensus_normal'],
                'Normality Strength': test_result['consensus']['strength'],
                'Skewness': test_result['skewness'],
                'Kurtosis': test_result['kurtosis'],
                'Annual Volatility (%)': returns_data.std() * np.sqrt(252) * 100
            })
    
    df = pd.DataFrame(results)
    
    # Filter for stocks with minimum normal tests passed
    normal_stocks = df[df['Normal Tests Passed'] >= min_normal_tests]
    
    # Sort by normality strength
    return normal_stocks.sort_values('Normality Strength', ascending=False)


def create_normality_summary(test_results):
    """
    Create a summary of normality test results
    
    Parameters:
    -----------
    test_results : dict
        Results from comprehensive_normality_test
        
    Returns:
    --------
    str
        Formatted summary string
    """
    summary = f"\n=== NORMALITY TEST SUMMARY: {test_results['name']} ===\n"
    summary += f"Sample Size: {test_results['sample_size']}\n"
    summary += f"Mean: {test_results['mean']:.6f}\n"
    summary += f"Std Dev: {test_results['std']:.6f}\n"
    summary += f"Skewness: {test_results['skewness']:.3f}\n"
    summary += f"Kurtosis: {test_results['kurtosis']:.3f}\n\n"
    
    summary += "INDIVIDUAL TEST RESULTS:\n"
    for test_name, test_result in test_results['tests'].items():
        if isinstance(test_result, dict) and 'normal' in test_result:
            status = "✓ NORMAL" if test_result['normal'] else "✗ NOT NORMAL"
            p_val = test_result.get('p_value', 'N/A')
            summary += f"  {test_name}: {status} (p={p_val:.6f})\n"
    
    consensus = test_results['consensus']
    summary += f"\nCONSENSUS: {consensus['normal_tests']}/{consensus['total_tests']} tests support normality\n"
    summary += f"Overall Assessment: {'NORMAL' if consensus['consensus_normal'] else 'NOT NORMAL'}\n"
    summary += f"Confidence: {consensus['strength']:.1%}\n"
    
    return summary


def rolling_window_normality_test(returns, window_size, alpha=0.05):
    """
    Perform rolling window normality testing on a time series.
    
    Parameters:
    returns (pd.Series): Time series of returns
    window_size (int): Size of the rolling window
    alpha (float): Significance level for the test
    
    Returns:
    dict: Summary of rolling window normality results
    """
    if len(returns) < window_size:
        return {
            'windows_tested': 0,
            'normal_windows': 0,
            'normality_rate': 0.0,
            'details': []
        }
    
    normal_count = 0
    total_windows = 0
    details = []
    
    for i in range(len(returns) - window_size + 1):
        window_data = returns.iloc[i:i+window_size]
        
        # Use Shapiro-Wilk test for the window
        stat, p_value = shapiro(window_data.dropna())
        is_normal = p_value > alpha
        
        if is_normal:
            normal_count += 1
        
        total_windows += 1
        
        details.append({
            'window_start': i,
            'window_end': i + window_size - 1,
            'p_value': p_value,
            'is_normal': is_normal
        })
    
    normality_rate = normal_count / total_windows if total_windows > 0 else 0.0
    
    return {
        'windows_tested': total_windows,
        'normal_windows': normal_count,
        'normality_rate': normality_rate,
        'details': details
    }
