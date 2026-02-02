# Stock Analyser + Backtester

A comprehensive Streamlit application for technical analysis and strategy backtesting of stock data.

## Features

- 📤 **Upload Data**: Import CSV or Excel files with OHLCV data
- 🌐 **Live Data**: Fetch real-time data from Yahoo Finance
- 📊 **Technical Indicators**: EMA, RSI, MACD, Bollinger Bands, ADX, ATR
- 💰 **Backtesting**: Full strategy backtesting with realistic fees and slippage
- 🔄 **Walk-Forward Analysis**: Out-of-sample validation
- 🔍 **Parameter Sweep**: Grid search for optimal parameters
- 📈 **Visual Charts**: Interactive candlestick charts and equity curves

## Installation

### Local Development

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy!

## Usage

### Upload Mode
1. Prepare a CSV or Excel file with at minimum a `Close` column
2. Best format includes: Date, Open, High, Low, Close, Volume
3. Upload the file and the analysis will run automatically

### Live Mode
1. Enter a ticker symbol (e.g., AAPL, TSLA, MSFT, ^GSPC)
2. Select the period and interval
3. Click "Fetch & Analyze"

### Configuration

Use the sidebar to adjust:
- **Signal Settings**: ADX filter for choppy markets
- **Backtest Settings**: Initial cash, risk per trade, stop/take levels
- **Advanced Settings**: Fees, slippage, and execution settings
- **Walk-Forward**: Train/test split ratio

## Strategy Logic

The default strategy combines:
- **Trend**: EMA50 vs EMA200 crossover
- **Momentum**: MACD signal
- **Risk Management**: RSI overbought/oversold filters
- **Regime Filter**: ADX to avoid choppy markets

Entries and exits use ATR-based stops and take profits with realistic fee and slippage modeling.

## Disclaimer

⚠️ **This is an educational tool for learning purposes only. Not financial advice.**

Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

## Technical Details

- Built with Streamlit, Pandas, and Plotly
- Uses yfinance for market data
- Vectorized backtesting engine
- Support for long and short positions
- Gap-aware stop loss fills
- Realistic transaction cost modeling

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## License

Educational use only. Use at your own risk.
