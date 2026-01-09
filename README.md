# Twitter Sentiment Analysis

A sentiment analysis application that analyzes Twitter data and Binance cryptocurrency price data for market sentiment insights.

## Project Structure

- `app.py` - Main application entry point
- `sentiment_analysis.py` - Sentiment analysis module
- `fetch_tweets.py` - Twitter data fetching
- `clean_text.py` - Text preprocessing and cleaning
- `realtime_monitoring.py` - Real-time monitoring functionality
- `visualize_results.py` - Results visualization
- `run_pipeline.py` - Pipeline orchestration
- `start_all.py` - Start all services
- `config.py` - Configuration settings
- `*_pack.json` - Cryptocurrency data packs (Binance, Bless, Clo, River)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Twitter-Sentiment.git
cd Twitter-Sentiment
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

Run the application:
```bash
python start_all.py
```

## Features

- Real-time Twitter sentiment analysis
- Cryptocurrency data integration
- Live monitoring dashboard
- Sentiment visualization

## License

MIT

## Author

Created by Usame
