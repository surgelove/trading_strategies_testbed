# Trading Strategy Testbed

A comprehensive Jupyter notebook-based platform for developing, testing, and visualizing algorithmic trading strategies using real-time market data, advanced analytics, and multiple visualization frameworks.

## Overview

This repository provides a complete trading strategy development environment with live OANDA API integration, SQLite data storage, real-time streaming analytics, and multiple visualization libraries. The platform supports both live market data and Monte Carlo simulations for comprehensive strategy testing and backtesting.

## Features

### 🌐 Live Market Data Integration
- **OANDA API Integration**: Real-time connection to OANDA Live FXTrade environment
- **Live Data Streaming**: Real-time price feeds with 5-second granularity
- **Multi-Instrument Support**: Support for major currency pairs (USD_CAD, EUR_USD, etc.)
- **Order Management**: Live order placement, modification, and cancellation
- **Position Tracking**: Real-time position monitoring and P&L tracking

### 💾 Data Management & Storage
- **SQLite Database**: Persistent storage for historical price data
- **Time-Based Queries**: Flexible timestamp queries with timezone offset support
- **Data Resampling**: Automatic resampling to multiple timeframes (5s, 1m, 5m, 15m, 1h)
- **Historical Analysis**: Query and analyze stored market data
- **Database Management**: Complete CRUD operations for trading data

### 📊 Advanced Visualization
- **Plotly Interactive Charts**: Zoom, pan, and hover functionality with dynamic scaling
- **Bokeh Visualization**: Multi-series interactive plots with distinct color coding
- **ECharts Integration**: Professional charting with range selectors and toolbox
- **Matplotlib Analytics**: Traditional plotting with moving averages and indicators
- **Real-time Updates**: Live chart updates during market data streaming

### 📈 Technical Analysis & Indicators
- **Moving Average Systems**: EMA, TEMA with customizable timeframes
- **Crossover Detection**: Automatic bullish/bearish signal identification
- **Time-Based Streaming**: Real-time indicator calculation during live data feeds
- **Multi-Timeframe Analysis**: Simultaneous analysis across multiple time periods
- **Signal Generation**: Automated trading signal detection and alerts

### 🔄 Trading Strategy Framework
- **Modular Architecture**: Separated data processing, analysis, and visualization
- **Strategy Development**: Framework for implementing custom trading algorithms
- **Backtesting Support**: Historical strategy performance evaluation
- **Risk Management**: Position sizing and risk assessment tools
- **Performance Metrics**: Strategy evaluation and optimization tools

### 🎲 Monte Carlo Simulation
- **Realistic Price Generation**: 24-hour stock price simulation with proper volatility
- **Volume Simulation**: Corresponding trading volume data generation
- **Multiple Timeframes**: Generate data at various intervals for strategy testing
- **Scenario Analysis**: Test strategies under different market conditions

## Backtesting

For comprehensive strategy testing and optimization, see the detailed methodology in [aia backtesting.pdf](aia%20backtesting.pdf).

### Steps
1. During the week, a gatherer running on a remote server gathers price data from OANDA for instruments of interest into database.
2. At end of week, Interactive Brokers is queried to get price data for other instruments of interest which are inserted into the database.
3. Databases are kept in a vault, one database per week.

### Backtesting Process
- For each week
  - For each instrument
    - For each strategy
      - Run algorithms
      - Generate signals
      - Make decisions
      - Calculate outcomes
      - Compare strategies
      - Compare outcomes
      - Tweak strategies for better outcomes

## Current Notebooks

### `test_frugalpurple.ipynb` - Comprehensive Trading Platform
The main notebook implementing a complete trading ecosystem:

#### Core Components:
1. **Database Operations**: SQLite integration for data persistence
2. **OANDA API Integration**: Live market data connection and order management
3. **Data Processing**: Real-time streaming analytics with multiple timeframes
4. **Visualization Suite**: Multiple charting libraries (Plotly, Bokeh, ECharts, Matplotlib)
5. **Technical Analysis**: Moving averages, crossover detection, and signal generation
6. **Monte Carlo Simulation**: Realistic market data generation for testing

#### Key Functions:
- `connect_to_database()`: SQLite database connection and management
- `get_oanda_data()`: Live OANDA API data retrieval
- `get_instrument_precision()`: Trading precision for different instruments
- `TimeBasedStreamingMA()`: Real-time moving average calculation
- `plot_bokeh_time_series()`: Interactive multi-series visualization
- `simple_echarts_plot()`: Professional charting with zoom and pan
- `plot_interactive_plotly()`: Dynamic interactive charts with hover tooltips

#### Trading Operations:
- **Order Management**: Place, modify, and cancel live orders
- **Position Tracking**: Monitor open positions and P&L
- **Risk Management**: Calculate position sizes and manage exposure
- **Signal Processing**: Automated detection of trading opportunities

## Getting Started

### Prerequisites
1. **OANDA Account**: Live trading account with API access
2. **API Credentials**: Store in `secrets.json` file:
   ```json
   {
     "api_key": "your_oanda_api_key",
     "account_id": "your_account_id",
     "url": "https://api-fxtrade.oanda.com"
   }
   ```

### Quick Start
1. **Setup Environment**: Install required dependencies
2. **Configure Database**: Initialize SQLite database for data storage
3. **Connect to OANDA**: Establish live API connection
4. **Load Historical Data**: Query and analyze stored market data
5. **Start Live Analysis**: Begin real-time strategy monitoring
6. **Visualize Results**: Use interactive charts for analysis

## Strategy Development

The modular design supports comprehensive strategy development:

### Data Layer
1. **Historical Analysis**: Query SQLite database for backtesting
2. **Live Data Integration**: Stream real-time market data from OANDA
3. **Multi-Timeframe Support**: Analyze across different time horizons
4. **Data Preprocessing**: Clean and resample data for analysis

### Indicator Development
1. **Technical Indicators**: Implement custom moving averages, oscillators
2. **Signal Generation**: Create automated buy/sell signal logic
3. **Cross-Timeframe Analysis**: Combine signals from multiple timeframes
4. **Performance Optimization**: Efficient real-time calculation methods

### Visualization & Analysis
1. **Interactive Charts**: Use Plotly, Bokeh, or ECharts for dynamic visualization
2. **Custom Dashboards**: Build comprehensive analysis interfaces
3. **Real-time Monitoring**: Live strategy performance tracking
4. **Historical Backtesting**: Evaluate strategy performance on historical data

### Trading Implementation
1. **Order Management**: Implement automated order placement logic
2. **Risk Controls**: Add position sizing and risk management rules
3. **Portfolio Management**: Multi-instrument strategy coordination
4. **Performance Tracking**: Monitor and optimize strategy results

## Technical Architecture

### Data Flow
```
OANDA API → SQLite Database → Data Processing → Technical Analysis → Visualization
    ↓              ↓                ↓               ↓              ↓
Live Orders ← Strategy Logic ← Signal Generation ← Indicators ← Interactive Charts
```

### Key Components
- **Data Frequency**: 5-second real-time data with multiple resampling options
- **Moving Averages**: Configurable EMA/TEMA with streaming calculation
- **Visualization**: Multiple libraries for different use cases
- **Database**: SQLite for persistent storage and historical analysis
- **API Integration**: Direct OANDA Live FXTrade connection

## Advanced Features

### Real-Time Analytics
- **Streaming Calculations**: Efficient real-time indicator updates
- **Event-Driven Architecture**: React to market changes instantly
- **Multi-Instrument Monitoring**: Track multiple currency pairs simultaneously
- **Custom Alert Systems**: Automated notifications for trading opportunities

### Visualization Capabilities
- **Plotly**: Interactive charts with zoom, pan, and hover functionality
- **Bokeh**: Multi-series plots with distinct color coding and legends
- **ECharts**: Professional charting with range selectors and toolbox
- **Matplotlib**: Traditional plotting for technical analysis and reports

### Database Operations
- **Time-Series Queries**: Efficient timestamp-based data retrieval
- **Data Aggregation**: Multi-timeframe analysis and reporting
- **Historical Backtesting**: Strategy performance on historical data
- **Data Export**: CSV and JSON export for external analysis

## Future Enhancements

### Trading Strategy Expansion
- **Advanced Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillators
- **Machine Learning**: Predictive models using scikit-learn and TensorFlow
- **Sentiment Analysis**: News and social media sentiment integration
- **Alternative Data**: Economic indicators and market sentiment feeds

### Platform Improvements
- **Web Interface**: Flask/Django web application for remote access
- **Real-Time Dashboards**: Live monitoring with WebSocket connections
- **Mobile Alerts**: Push notifications for trading signals
- **Cloud Deployment**: AWS/GCP deployment for 24/7 operation

### Advanced Analytics
- **Portfolio Optimization**: Multi-asset allocation strategies
- **Risk Analytics**: VaR, drawdown analysis, and stress testing
- **Performance Attribution**: Detailed strategy performance breakdown
- **Correlation Analysis**: Cross-asset and cross-strategy relationships

### Integration & Automation
- **Multiple Brokers**: Interactive Brokers, TD Ameritrade, Alpaca integration
- **Cryptocurrency**: Binance, Coinbase Pro API connections
- **Automated Trading**: Fully automated strategy execution
- **Regulatory Compliance**: Trade reporting and audit trail features

## Dependencies

### Core Libraries
```bash
pip install pandas numpy matplotlib jupyter
pip install plotly bokeh pyecharts
pip install requests sqlite3 json
pip install ipywidgets nbformat
```

### OANDA Integration
```bash
pip install oandapyV20  # Official OANDA Python API
```

### Optional Enhancements
```bash
pip install scikit-learn tensorflow  # Machine learning
pip install flask dash streamlit      # Web interfaces
pip install ta-lib                    # Technical analysis
pip install yfinance alpha_vantage    # Additional data sources
```

## Security & Risk Management

### API Security
- Store credentials in `secrets.json` (not in version control)
- Use environment variables for production deployment
- Implement API rate limiting and error handling
- Regular credential rotation and monitoring

### Trading Risk Controls
- Position sizing based on account equity
- Maximum daily loss limits
- Stop-loss and take-profit automation
- Real-time risk monitoring and alerts

---

*This repository serves as a comprehensive foundation for professional algorithmic trading strategy development, combining live market data, advanced analytics, and institutional-grade visualization tools. The platform is designed for both strategy research and live trading implementation.*