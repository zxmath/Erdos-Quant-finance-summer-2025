# Quantitative Finance Mini Projects
**2025 Introduction to Quantitative Methods in Finance**  
**The Erdős Institute**

This repository contains four comprehensive mini projects exploring fundamental concepts in quantitative finance, from portfolio optimization to advanced options hedging strategies.

## 📈 Project Overview

### [Mini Project 1: Portfolio Construction and Risk Analysis](./Mini%20Project%201.ipynb)
**Objective**: Create and analyze high-risk and low-risk investment portfolios using real financial data.

**Key Features**:
- **Real Data Analysis**: Downloads live stock data using Yahoo Finance API
- **Portfolio Optimization**: Uses mean-variance optimization to construct optimal portfolios
- **Risk-Return Profiling**: Comprehensive analysis of portfolio characteristics
- **Quantitative Validation**: Validates portfolios against predefined risk criteria

**Portfolio Characteristics**:
- **High-Risk Portfolio**: Technology and growth stocks with >25% annual volatility
- **Low-Risk Portfolio**: Defensive stocks and utilities with <15% annual volatility
- **Metrics Analyzed**: Sharpe ratio, VaR (95%), maximum drawdown, correlation analysis

**Technical Implementation**:
- Custom utility modules for data processing and portfolio analysis
- Advanced visualization tools for risk-return comparison
- Professional data persistence for use in subsequent projects

---

### [Mini Project 2: Statistical Hypothesis Testing in Finance](./Mini%20Project%202.ipynb)
**Objective**: Test the fundamental assumption that log returns of financial assets follow normal distributions.

**Comprehensive Testing Framework**:
- **Multiple Statistical Tests**: Shapiro-Wilk, Kolmogorov-Smirnov, Jarque-Bera, D'Agostino
- **Rolling Window Analysis**: Time-varying normality assessment
- **Outlier Impact Studies**: Effect of extreme values on distributional assumptions
- **Portfolio vs Individual Stock Analysis**: Comparison of distributional properties

**Key Findings**:
- **Overwhelming rejection** of normality across most financial time series
- **Portfolio diversification** improves but doesn't achieve normality
- **Time-varying characteristics** of distributional properties
- **Outlier sensitivity** significantly impacts normality test results

**Methodological Insights**:
- Different tests show varying sensitivity to non-normality
- Kolmogorov-Smirnov less sensitive than Shapiro-Wilk for small deviations
- Rolling window analysis reveals period-dependent normality characteristics

---

### [Mini Project 3: Black-Scholes Options Analysis and Greeks](./Mini%20Project%203.ipynb)
**Objective**: Comprehensive analysis of Black-Scholes option pricing and sensitivity measures (Greeks).

**Analysis Components**:
- **Time Sensitivity (Theta)**: How option prices change as expiration approaches
- **Price Sensitivity (Delta)**: Rate of change with respect to underlying asset price
- **Call vs Put Comparison**: Symmetric analysis of both option types
- **Visual Greek Analysis**: Interactive plots showing sensitivity behaviors

**Key Insights**:
- **Delta Patterns**: S-curve behavior from 0 to 1 for calls, -1 to 0 for puts
- **Time Decay**: Accelerating theta as expiration approaches
- **At-the-Money Behavior**: Maximum sensitivity occurs near strike price
- **Put-Call Relationships**: Validation of theoretical parity relationships

**Financial Applications**:
- **Risk Management**: Understanding exposure to price and time changes
- **Hedging Strategies**: Greeks inform hedge ratio calculations
- **Trading Insights**: Optimal timing and positioning strategies

---

### [Mini Project 4: Advanced Volatility Modeling and Sigma Hedging](./Mini%20Project%204.ipynb)
**Objective**: Explore delta hedging performance under stochastic volatility and implement advanced hedging strategies.

**Volatility Models Implemented**:
- **Custom Stochastic**: Discrete volatility regime switches
- **Heston Model**: Mean-reverting stochastic volatility
- **GARCH(1,1)**: Volatility clustering with persistence
- **Constant Volatility**: Baseline Black-Scholes assumption

**Advanced Hedging Strategies**:
- **Pure Delta Hedging**: Traditional stock-based hedging
- **Delta-Vega Hedging**: Two-instrument hedging against price and volatility risk
- **Frequency Analysis**: Optimal rebalancing frequency under different volatility regimes

**Key Discoveries**:
- **GARCH model** produces highest hedging risk due to volatility clustering
- **Sigma hedging** provides substantial risk reduction, especially for GARCH
- **Hedging frequency** optimization varies by volatility model
- **Stochastic volatility** significantly increases hedging uncertainty

**Risk Management Implications**:
- Traditional delta hedging inadequate under realistic volatility conditions
- Multi-factor hedging strategies essential for sophisticated volatility models
- Transaction cost vs. tracking error trade-offs in rebalancing decisions

## 🛠️ Technical Architecture

### Utility Modules
The project includes comprehensive utility modules for professional quantitative analysis:

#### [`utils/data_utils.py`](./utils/data_utils.py)
- Real-time financial data downloading and processing
- Data validation and cleaning procedures
- Standardized return calculations (simple, log, adjusted)
- Professional data persistence and loading functions

#### [`utils/portfolio_metrics.py`](./utils/portfolio_metrics.py)
- Portfolio-level risk and return calculations
- Individual asset performance metrics
- Correlation and covariance analysis
- Value-at-Risk (VaR) and other risk measures

#### [`utils/normality_tests.py`](./utils/normality_tests.py)
- Comprehensive statistical testing framework
- Rolling window analysis tools
- Outlier impact assessment functions
- Consensus testing across multiple statistical methods

#### [`utils/visualization_utils.py`](./utils/visualization_utils.py)
- Professional financial plotting functions
- Risk-return scatter plots and efficient frontier visualization
- Correlation heatmaps and time series analysis
- Normality assessment visualization tools

### Data Pipeline
```
Raw Financial Data → Processing → Analysis → Visualization → Insights
     ↓                 ↓           ↓            ↓            ↓
Yahoo Finance API → data_utils → portfolio/ → visualization → Actionable
yfinance/yahooquery              normality/    professional   Financial
                                 options       charts         Intelligence
```

## 📊 Key Results and Insights

### Portfolio Construction (Project 1)
- **Successful Risk Differentiation**: Created portfolios with distinct risk profiles
- **Quantitative Validation**: High-risk portfolio achieved >25% volatility, low-risk <15%
- **Optimization Success**: Mean-variance optimization produced sensible allocations
- **Professional Standards**: Results suitable for institutional portfolio management

### Normality Testing (Project 2)
- **Fundamental Assumption Invalid**: Normal distribution assumption consistently violated
- **Universal Finding**: Both individual stocks and portfolios show non-normal behavior
- **Test Sensitivity Matters**: Different statistical tests reveal varying aspects of non-normality
- **Practical Implications**: Financial models must account for non-normal distributions

### Options Analysis (Project 3)
- **Greeks Behavior Validated**: Theoretical predictions confirmed through comprehensive analysis
- **Visual Understanding**: Clear patterns in Delta, Theta behavior across market conditions
- **Risk Management Tools**: Practical insights for options trading and hedging
- **Educational Value**: Strong foundation for advanced derivatives strategies

### Advanced Hedging (Project 4)
- **Volatility Model Impact**: GARCH model most challenging for traditional delta hedging
- **Sigma Hedging Value**: Delta-vega strategies significantly reduce hedging risk
- **Frequency Optimization**: Diminishing returns beyond daily rebalancing
- **Realistic Modeling**: Sophisticated volatility models essential for accurate risk assessment

## 🎯 Learning Outcomes

### Technical Skills Developed
- **Advanced Python Programming**: Object-oriented design with financial applications
- **Statistical Analysis**: Comprehensive hypothesis testing and distributional analysis
- **Financial Modeling**: Portfolio optimization, options pricing, stochastic processes
- **Data Visualization**: Professional-grade financial charts and analysis tools
- **Risk Management**: VaR, stress testing, and advanced hedging strategies

### Quantitative Finance Concepts
- **Modern Portfolio Theory**: Mean-variance optimization and efficient frontier analysis
- **Derivatives Pricing**: Black-Scholes model and Greeks sensitivity analysis
- **Statistical Finance**: Hypothesis testing applied to financial assumptions
- **Stochastic Modeling**: Heston, GARCH, and custom volatility processes
- **Risk Management**: Multi-factor hedging and performance optimization

### Professional Applications
- **Portfolio Management**: Institutional-quality portfolio construction and analysis
- **Risk Assessment**: Comprehensive statistical testing of financial assumptions
- **Options Trading**: Practical Greeks analysis and hedging strategy implementation
- **Model Validation**: Testing and comparing different financial modeling approaches

## 🚀 Usage Instructions

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scipy yfinance yahooquery tabulate
```

### Running the Projects
1. **Start with Project 1** to generate baseline portfolio data
2. **Project 2** can run independently or use Project 1 data for Task 4
3. **Project 3** is standalone options analysis
4. **Project 4** explores advanced hedging concepts

### Data Dependencies
- **Project 1**: Generates processed portfolio data for other projects
- **Project 2**: Can use either independent data or Project 1 portfolios
- **Projects 3 & 4**: Standalone analysis with synthetic/real data

## 📚 Educational Context

These projects form a comprehensive introduction to quantitative finance, covering:
- **Foundational Concepts**: Portfolio theory, statistical testing, options pricing
- **Advanced Methods**: Stochastic volatility, multi-factor hedging, risk management
- **Professional Tools**: Industry-standard libraries and methodologies
- **Practical Applications**: Real-world financial problems and solutions

## 🎓 Academic References
- Modern Portfolio Theory (Markowitz, 1952)
- Black-Scholes-Merton Options Pricing Model
- Statistical Finance and Econometrics
- Stochastic Volatility Models (Heston, GARCH)
- Advanced Derivatives and Risk Management

---

*This comprehensive analysis provides a solid foundation in quantitative finance methods, combining theoretical understanding with practical implementation skills essential for modern financial analysis and risk management.*
