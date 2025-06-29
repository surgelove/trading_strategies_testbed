# Trading Strategy Testbed

A Jupyter notebook-based platform for developing, testing, and visualizing algorithmic trading strategies using real-time data simulation.

## Overview

This repository contains a series of trading strategy tests and simulations designed to help develop and evaluate algorithmic trading approaches. The primary focus is on providing a flexible, visual environment for strategy experimentation using live data streams.

## Features

### 📊 Data Simulation
- **Monte Carlo Stock Price Generation**: Creates realistic 24-hour stock price data with 1-minute intervals
- **Volume Simulation**: Generates corresponding trading volume data
- **Live Data Streaming**: Real-time data processing at configurable intervals (0.1s to 1s)

### 📈 Visualization
- **Real-time Charts**: Live updating price and volume charts using matplotlib
- **Dynamic Scaling**: Automatic y-axis adjustment for optimal viewing
- **Full History Display**: Complete price history with visual indicators
- **Signal Annotations**: Real-time display of trading signals and crossovers

### 🔄 Trading Strategies
- **Moving Average Crossovers**: Implementation of EMA (Exponential Moving Average) crossover strategies
- **Triple EMA System**: 15-minute triple EMA as crosser signal
- **Standard EMA**: 15-minute EMA as crossee baseline
- **Signal Detection**: Automatic bullish/bearish crossover identification

### 🏗️ Modular Architecture
- **Separated Concerns**: Independent data processing and visualization modules
- **Configurable Parameters**: Easy adjustment of timeframes, intervals, and strategy parameters
- **Extensible Design**: Framework ready for additional strategy implementations

## Current Notebooks

### `test1.ipynb` - Core Trading Strategy Simulator
The main notebook implementing:
1. **Data Generation Cell**: Monte Carlo simulation for realistic stock price movements
2. **Live Streaming Cell**: Real-time data processing with moving average calculations
3. **Strategy Processing Cell**: Modular functions for strategy implementation

#### Key Functions:
- `process_moving_averages_data()`: Calculates EMAs and detects crossover signals
- `display_live_chart()`: Real-time visualization of price action and indicators

## Getting Started

1. **Open the notebook**: Launch `test1.ipynb` in Jupyter
2. **Run data generation**: Execute the first cell to create sample stock data
3. **Start live simulation**: Run the streaming cell to begin real-time strategy testing
4. **Observe signals**: Watch for moving average crossovers and trading signals

## Strategy Development

The modular design allows for easy strategy development:

1. **Add new indicators**: Extend the `process_moving_averages_data()` function
2. **Implement new strategies**: Create additional processing functions
3. **Customize visualization**: Modify the `display_live_chart()` function
4. **Test parameters**: Adjust timeframes, thresholds, and signal logic

## Technical Details

- **Data Frequency**: 1-minute historical data, processed at 0.1-second intervals
- **Moving Averages**: 15-minute window for both triple EMA and standard EMA
- **Visualization**: Matplotlib with real-time updates and dynamic scaling
- **Signal Types**: Bullish crossovers (crosser above crossee), Bearish crossovers (crosser below crossee)

## Future Enhancements

- Additional strategy implementations (RSI, MACD, Bollinger Bands)
- Backtesting framework with performance metrics
- Portfolio management and risk assessment
- Multi-asset strategy testing
- Strategy comparison and optimization tools

## Dependencies

- pandas
- numpy
- matplotlib
- jupyter
- ipywidgets (for interactive displays)

---

*This repository serves as a foundation for algorithmic trading strategy development and testing. Each notebook represents a different approach or strategy variant, providing a comprehensive testing environment for quantitative trading research.*
