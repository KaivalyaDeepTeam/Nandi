# Forex Predictor - Real-time Trading POC

AI-powered forex prediction system using LSTM + XGBoost ensemble, integrated with MetaTrader 5 via socket bridge.

## Architecture

```
Python (any OS)                    MetaTrader 5
┌──────────────────────┐          ┌─────────────────────┐
│  Feature Engineering │◄────────►│  ForexPredictor.mq5 │
│  LSTM + XGBoost      │  TCP/IP  │  (Socket Bridge EA)  │
│  Risk Management     │          │  - Price data        │
│  Trade Execution     │────────►│  - Order execution   │
└──────────────────────┘          └─────────────────────┘
```

## Setup

### 1. MetaTrader 5 Demo Account

1. Download MetaTrader 5 from [metatrader5.com](https://www.metatrader5.com/en/download)
   - **Windows**: Install directly
   - **macOS**: Use Parallels, VMware, Wine, or a Windows VPS
2. Open MT5 and create a demo account:
   - File > Open an Account
   - Select broker (e.g., MetaQuotes-Demo, ICMarkets, Pepperstone)
   - Choose "Demo account" with $10,000+ virtual balance
3. Copy `mt5/ForexPredictor.mq5` to your MT5 data folder:
   - In MT5: File > Open Data Folder > MQL5 > Experts
   - Paste the file there
4. In MT5: Tools > Options > Expert Advisors:
   - Check "Allow algorithmic trading"
   - Check "Allow DLL imports"
5. Compile the EA: open it in MetaEditor (F4) and press Compile (F7)
6. Drag `ForexPredictor` onto any chart
7. In the EA settings, confirm port = 5555

### 2. Python Environment

```bash
cd forex_predictor
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Network (if MT5 is on a different machine)

- Change `MT5_HOST` in `config/settings.py` to the MT5 machine's IP
- Ensure port 5555 is open between the machines

## Usage

### Train the model
```bash
python train.py --symbol EURUSD --bars 5000
```

### Run backtest
```bash
python backtest.py --symbol EURUSD --bars 10000
```

### Start live trading (demo)
```bash
python main.py --symbol EURUSD
python main.py --symbol EURUSD --retrain  # retrain before starting
```

### Stop the bot
Press `Ctrl+C` for graceful shutdown.

## Configuration

Edit `config/settings.py` to customize:
- **SYMBOLS**: Trading pairs (add any pair your broker supports)
- **RISK_CONFIG**: Risk per trade, max positions, SL/TP
- **LSTM_CONFIG / XGBOOST_CONFIG**: Model hyperparameters
- **ENSEMBLE_CONFIG**: Model weights and confidence threshold

## Project Structure

```
forex_predictor/
├── config/settings.py          # All configuration
├── mt5/ForexPredictor.mq5      # MT5 Expert Advisor (socket bridge)
├── src/
│   ├── mt5_connector.py        # TCP connection to MT5
│   ├── feature_engineer.py     # Technical indicators
│   ├── models/
│   │   ├── lstm_model.py       # LSTM neural network
│   │   ├── xgboost_model.py    # XGBoost classifier
│   │   └── ensemble.py         # Combined predictor
│   ├── risk_manager.py         # Position sizing & risk rules
│   └── trade_executor.py       # Trade execution logic
├── train.py                    # Training pipeline
├── backtest.py                 # Historical backtesting
└── main.py                     # Live trading bot
```

## Disclaimer

This is a proof-of-concept for educational purposes. Use only with demo accounts. Past performance does not guarantee future results. Forex trading involves substantial risk of loss.
