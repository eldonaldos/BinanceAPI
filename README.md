# Binance API Trading Price Fetcher

A cryptocurrency trading tool that fetches real-time price data from Binance API for multiple trading pairs to support automated trading decision making.

## Overview

This repository fetches live cryptocurrency price data from the Binance API and stores it in JSON format for analysis and trading strategies. It supports multiple trading pairs including CLOUSDT, BLESSUSDT, and RIVERUSDT.

## Project Structure

- `binance_pack.py` - Main script to fetch cryptocurrency prices from Binance API
- `app.py` - Main application entry point
- `sentiment_analysis.py` - Sentiment analysis module for trading signals
- `fetch_tweets.py` - Twitter data fetching for market sentiment
- `clean_text.py` - Text preprocessing and cleaning
- `realtime_monitoring.py` - Real-time monitoring of price movements
- `visualize_results.py` - Results visualization and charts
- `run_pipeline.py` - Pipeline orchestration
- `start_all.py` - Start all services
- `config.py` - Configuration settings
- `*_pack.json` - Cryptocurrency price data packs from Binance (CLOUSDT, BLESSUSDT, RIVERUSDT, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eldonaldos/BinanceAPI.git
cd BinanceAPI
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Fetch cryptocurrency prices from Binance API:
```bash
python binance_pack.py
```

Run the full trading application:
```bash
python start_all.py
```

## Features

- **Real-time Price Fetching** - Fetch live cryptocurrency prices from Binance API
- **Multiple Trading Pairs** - Support for multiple cryptocurrency trading pairs
- **Data Storage** - Store price data in JSON format for analysis
- **Sentiment Analysis** - Analyze Twitter sentiment for trading signals
- **Real-time Monitoring** - Monitor price movements and market conditions
- **Visualization** - Generate charts and visualizations for trading decisions

## License

MIT

## Author

Created by Usame
