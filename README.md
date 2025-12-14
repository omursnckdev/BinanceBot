# Binance Futures Trading Bot

A production-grade Python trading bot for Binance USDT-M Futures with a **testnet-first** approach.

## Features

- **Multi-Signal Fusion Strategy**: Combines technical indicators, whale activity detection, and news sentiment analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, EMA trend filter
- **Whale Detection**: Large trade flagging and order book imbalance analysis
- **News Sentiment**: Real-time news analysis with VADER sentiment scoring
- **Risk Management**: Kill-switches, position sizing, stop-loss/take-profit
- **Safety First**: Testnet default, dry-run mode, multiple safety toggles

## Safety Features

⚠️ **This bot trades with real money. Use at your own risk.**

Built-in safety measures:
- ✅ **Testnet by default** - No mainnet access without explicit configuration
- ✅ **Dry-run mode** - Simulates trades without executing
- ✅ **Triple safety toggle** - Requires ENV=mainnet, DRY_RUN=false, AND ALLOW_LIVE_TRADING=true
- ✅ **Daily loss kill-switch** - Stops trading if daily loss exceeds threshold
- ✅ **Consecutive loss protection** - Pauses after multiple losses
- ✅ **Position limits** - Caps total exposure and position count
- ✅ **Cooldown periods** - Enforces waiting period after losses

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Binance Futures Testnet account

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd BinanceBot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or your preferred editor
```

### 4. Create Testnet API Keys

1. Visit [Binance Futures Testnet](https://testnet.binancefuture.com/)
2. Log in or create account (uses Binance main account)
3. Go to API Management
4. Create a new API key
5. Copy the API Key and Secret to your `.env` file

### 5. Running the Bot

#### Dry-Run Mode (Recommended First)

```bash
# Run in dry-run mode (default)
python main.py
```

The bot will:
- Connect to testnet
- Fetch market data
- Generate trading signals
- Log what trades it WOULD make
- NOT place any actual orders

#### Testnet Live Mode

After you're confident the bot works:

```bash
# Edit .env
DRY_RUN=false
# Keep ENV=testnet and ALLOW_LIVE_TRADING=false
```

Now the bot will place real orders on testnet (test money only).

#### Mainnet Mode (⚠️ Real Money)

**Only proceed if you fully understand the risks!**

```bash
# Edit .env
ENV=mainnet
DRY_RUN=false
ALLOW_LIVE_TRADING=true

# Also update API keys to mainnet keys
BINANCE_API_KEY=your_mainnet_api_key
BINANCE_API_SECRET=your_mainnet_api_secret
```

## Project Structure

```
BinanceBot/
├── config.py              # Pydantic settings with defaults
├── main.py                # Main entry point and event loop
│
├── exchange/
│   └── binance_client.py  # Binance API wrapper
│
├── data/
│   └── market_data.py     # Candles, trades, order book
│
├── signals/
│   ├── indicators.py      # Technical indicators
│   ├── whales.py          # Whale detection
│   └── sentiment.py       # News sentiment analysis
│
├── strategy/
│   └── fusion.py          # Signal fusion and decisions
│
├── risk/
│   └── risk_manager.py    # Position sizing, kill-switches
│
├── execution/
│   └── executor.py        # Order execution with dry-run
│
├── state/
│   └── store.py           # Position and trade tracking
│
├── utils/
│   └── logger.py          # Structured JSON logging
│
├── tests/                  # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `testnet` | Environment: `testnet` or `mainnet` |
| `DRY_RUN` | `true` | Simulate trades without executing |
| `ALLOW_LIVE_TRADING` | `false` | Additional safety toggle for mainnet |
| `BINANCE_API_KEY` | - | Your Binance API key |
| `BINANCE_API_SECRET` | - | Your Binance API secret |
| `SYMBOLS` | BTCUSDT,... | Comma-separated trading pairs |
| `NEWS_API_KEY` | - | CryptoPanic or NewsAPI key |

### Risk Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_DAILY_LOSS_PCT` | 5.0 | Kill-switch: max daily loss % |
| `MAX_CONSECUTIVE_LOSSES` | 3 | Kill-switch: max consecutive losses |
| `MAX_OPEN_POSITIONS` | 3 | Maximum concurrent positions |
| `MAX_TOTAL_EXPOSURE_PCT` | 50.0 | Max exposure as % of balance |
| `COOLDOWN_AFTER_LOSS_SECONDS` | 300 | Wait time after a loss |

## Trading Strategy

### Signal Fusion

The bot combines three signal sources:

1. **Technical Indicators** (weight: 50%)
   - RSI for overbought/oversold
   - MACD for momentum
   - Bollinger Bands for volatility
   - EMA crossover for trend

2. **Whale Detection** (weight: 30%)
   - Large trade detection via aggregated trades
   - Order book imbalance analysis
   - Defensive close on opposing whale activity

3. **News Sentiment** (weight: 20%)
   - Real-time news fetching
   - VADER sentiment analysis
   - Coin-specific keyword mapping

### Decision Flow

1. Fetch market data for all timeframes
2. Calculate technical indicators
3. Detect whale activity
4. Analyze recent news sentiment
5. Compute weighted fusion score
6. Apply entry threshold and trend confirmation
7. Check risk limits and cooldowns
8. Execute trade with bracket orders (entry + SL + TP)

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_indicators.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Logging

Logs are written to:
- Console (human-readable format)
- File (JSON format for parsing)

Log levels:
- `DEBUG`: All details including indicator values
- `INFO`: Trade decisions and important events
- `WARNING`: Risk events and data issues
- `ERROR`: API errors and failures

## API Rate Limits

The bot respects Binance rate limits:
- Uses exponential backoff on errors
- Caches exchange info
- Batches requests where possible

## Disclaimer

⚠️ **IMPORTANT DISCLAIMER** ⚠️

- This bot is provided for educational purposes only
- Trading cryptocurrencies involves substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- The authors are not responsible for any financial losses
- Always test thoroughly on testnet before using real money
- This is not financial advice

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

## Support

For issues and feature requests, please open a GitHub issue.
