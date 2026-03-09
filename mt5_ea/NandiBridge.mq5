//+------------------------------------------------------------------+
//|                                                  NandiBridge.mq5  |
//|                        Nandi RL Trading System — MT5 File Bridge  |
//|                                                                    |
//|  This EA runs inside MetaTrader 5 and communicates with Nandi     |
//|  (Python) via shared CSV files in MQL5/Files directory.           |
//|                                                                    |
//|  Setup: Attach to any chart. It handles ALL 8 pairs automatically.|
//|                                                                    |
//|  File Protocol:                                                    |
//|    MT5 → Python:                                                   |
//|      fx_tick.csv      — current bid/ask for all pairs             |
//|      fx_account.csv   — account balance, equity, margin           |
//|      fx_positions.csv — all open positions                         |
//|      fx_m5_PAIR.csv   — M5 OHLCV bars per pair                   |
//|    Python → MT5:                                                   |
//|      fx_command.csv   — trade commands                             |
//|    MT5 → Python:                                                   |
//|      fx_response.csv  — command execution results                  |
//+------------------------------------------------------------------+
#property copyright "Nandi Trading System"
#property version   "2.00"
#property strict

// ── Input Parameters ──────────────────────────────────────────────
input int    M5BarsToExport  = 5000;      // Number of M5 bars to export per pair
input int    UpdateIntervalMs = 1000;     // How often to update files (milliseconds)
input bool   ExportM5Data    = true;      // Export M5 OHLCV bars
input bool   PaperMode       = true;      // Paper mode (log but don't execute)
input double MaxLotSize      = 0.1;       // Maximum lot size per trade
input int    MaxSlippagePoints = 30;      // Max slippage in points

// ── Pairs to monitor ─────────────────────────────────────────────
string Pairs[] = {
   "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
   "NZDUSD", "USDCHF", "USDCAD", "EURJPY"
};

// OctaFX may use suffix like "m" or ".pro" — detect automatically
string PairSuffix = "";

// ── File paths ───────────────────────────────────────────────────
string TickFile     = "fx_tick.csv";
string AccountFile  = "fx_account.csv";
string PositionsFile = "fx_positions.csv";
string CommandFile  = "fx_command.csv";
string ResponseFile = "fx_response.csv";

// ── State ────────────────────────────────────────────────────────
datetime lastUpdate = 0;
datetime lastM5Export = 0;
int tickCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   // Detect OctaFX symbol suffix
   DetectSymbolSuffix();

   Print("╔══════════════════════════════════════════╗");
   Print("║     NANDI BRIDGE EA v2.0 — STARTED       ║");
   Print("╠══════════════════════════════════════════╣");
   PrintFormat("║  Pairs:      %d", ArraySize(Pairs));
   PrintFormat("║  Suffix:     '%s'", PairSuffix);
   PrintFormat("║  M5 Bars:    %d", M5BarsToExport);
   PrintFormat("║  Interval:   %d ms", UpdateIntervalMs);
   PrintFormat("║  Paper Mode: %s", PaperMode ? "YES" : "NO — LIVE TRADING");
   PrintFormat("║  Max Lot:    %.2f", MaxLotSize);
   Print("╚══════════════════════════════════════════╝");

   // Initial export of all data
   ExportAllTicks();
   ExportAccountInfo();
   ExportPositions();

   if(ExportM5Data)
      ExportAllM5Bars();

   // Set timer for periodic updates
   EventSetMillisecondTimer(UpdateIntervalMs);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("Nandi Bridge EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Timer event — periodic data export and command processing          |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Export market data
   ExportAllTicks();
   ExportAccountInfo();
   ExportPositions();

   // Export M5 bars every 5 minutes (300 seconds)
   if(ExportM5Data && TimeCurrent() - lastM5Export >= 300)
   {
      ExportAllM5Bars();
      lastM5Export = TimeCurrent();
   }

   // Check for commands from Nandi
   ProcessCommands();
}

//+------------------------------------------------------------------+
//| Tick event — also check commands on every tick                     |
//+------------------------------------------------------------------+
void OnTick()
{
   tickCount++;

   // Process commands on every tick for faster response
   if(tickCount % 5 == 0)  // Every 5th tick
      ProcessCommands();
}

//+------------------------------------------------------------------+
//| Detect OctaFX symbol suffix                                        |
//+------------------------------------------------------------------+
void DetectSymbolSuffix()
{
   // OctaFX symbols may be: EURUSD, EURUSDm, EURUSD.pro, etc.
   string testNames[] = {"", "m", ".pro", "-m", "_m", ".ecn"};

   for(int i = 0; i < ArraySize(testNames); i++)
   {
      string testSymbol = "EURUSD" + testNames[i];
      if(SymbolInfoInteger(testSymbol, SYMBOL_EXIST))
      {
         PairSuffix = testNames[i];
         PrintFormat("Detected symbol suffix: '%s' (test: %s)", PairSuffix, testSymbol);
         return;
      }
   }

   // Fallback: try the chart symbol
   string chartSymbol = Symbol();
   if(StringFind(chartSymbol, "EURUSD") >= 0)
   {
      PairSuffix = StringSubstr(chartSymbol, 6);
      PrintFormat("Using chart symbol suffix: '%s'", PairSuffix);
   }
}

//+------------------------------------------------------------------+
//| Get broker symbol name (pair + suffix)                             |
//+------------------------------------------------------------------+
string BrokerSymbol(string pair)
{
   return pair + PairSuffix;
}

//+------------------------------------------------------------------+
//| Export current bid/ask for all pairs                                |
//+------------------------------------------------------------------+
void ExportAllTicks()
{
   int handle = FileOpen(TickFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open ", TickFile);
      return;
   }

   FileWrite(handle, "symbol", "bid", "ask", "spread", "time", "point", "digits",
             "trade_tick_value", "trade_tick_size", "volume_min", "volume_max", "volume_step");

   for(int i = 0; i < ArraySize(Pairs); i++)
   {
      string sym = BrokerSymbol(Pairs[i]);
      if(!SymbolInfoInteger(sym, SYMBOL_EXIST))
         continue;

      // Ensure symbol is in Market Watch
      SymbolSelect(sym, true);

      double bid = SymbolInfoDouble(sym, SYMBOL_BID);
      double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
      int spread = (int)SymbolInfoInteger(sym, SYMBOL_SPREAD);
      double point = SymbolInfoDouble(sym, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
      double tickVal = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
      double tickSize = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
      double volMin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
      double volMax = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
      double volStep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);

      FileWrite(handle, Pairs[i], bid, ask, spread, TimeCurrent(),
                point, digits, tickVal, tickSize, volMin, volMax, volStep);
   }

   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Export account information                                         |
//+------------------------------------------------------------------+
void ExportAccountInfo()
{
   int handle = FileOpen(AccountFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE) return;

   FileWrite(handle, "field", "value");
   FileWrite(handle, "balance", AccountInfoDouble(ACCOUNT_BALANCE));
   FileWrite(handle, "equity", AccountInfoDouble(ACCOUNT_EQUITY));
   FileWrite(handle, "margin", AccountInfoDouble(ACCOUNT_MARGIN));
   FileWrite(handle, "free_margin", AccountInfoDouble(ACCOUNT_MARGIN_FREE));
   FileWrite(handle, "margin_level", AccountInfoDouble(ACCOUNT_MARGIN_LEVEL));
   FileWrite(handle, "profit", AccountInfoDouble(ACCOUNT_PROFIT));
   FileWrite(handle, "currency", AccountInfoString(ACCOUNT_CURRENCY));
   FileWrite(handle, "leverage", AccountInfoInteger(ACCOUNT_LEVERAGE));
   FileWrite(handle, "server", AccountInfoString(ACCOUNT_SERVER));
   FileWrite(handle, "company", AccountInfoString(ACCOUNT_COMPANY));
   FileWrite(handle, "paper_mode", PaperMode ? "1" : "0");
   FileWrite(handle, "timestamp", TimeCurrent());

   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Export all open positions                                           |
//+------------------------------------------------------------------+
void ExportPositions()
{
   int handle = FileOpen(PositionsFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE) return;

   FileWrite(handle, "ticket", "symbol", "type", "volume", "price_open",
             "sl", "tp", "profit", "swap", "time_open", "comment");

   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      // Map broker symbol back to standard name
      string sym = PositionGetString(POSITION_SYMBOL);
      string stdPair = sym;
      if(PairSuffix != "")
         StringReplace(stdPair, PairSuffix, "");

      FileWrite(handle,
         ticket,
         stdPair,
         PositionGetInteger(POSITION_TYPE),
         PositionGetDouble(POSITION_VOLUME),
         PositionGetDouble(POSITION_PRICE_OPEN),
         PositionGetDouble(POSITION_SL),
         PositionGetDouble(POSITION_TP),
         PositionGetDouble(POSITION_PROFIT),
         PositionGetDouble(POSITION_SWAP),
         PositionGetInteger(POSITION_TIME),
         PositionGetString(POSITION_COMMENT)
      );
   }

   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Export M5 OHLCV bars for all pairs                                 |
//+------------------------------------------------------------------+
void ExportAllM5Bars()
{
   for(int i = 0; i < ArraySize(Pairs); i++)
   {
      ExportM5ForPair(Pairs[i]);
   }
   PrintFormat("Exported M5 bars for %d pairs (%d bars each)", ArraySize(Pairs), M5BarsToExport);
}

//+------------------------------------------------------------------+
//| Export M5 bars for a single pair                                    |
//+------------------------------------------------------------------+
void ExportM5ForPair(string pair)
{
   string sym = BrokerSymbol(pair);
   if(!SymbolInfoInteger(sym, SYMBOL_EXIST))
      return;

   MqlRates rates[];
   int copied = CopyRates(sym, PERIOD_M5, 0, M5BarsToExport, rates);
   if(copied <= 0)
   {
      PrintFormat("WARNING: No M5 data for %s (copied=%d)", sym, copied);
      return;
   }

   string filename = "fx_m5_" + pair + ".csv";
   int handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("ERROR: Cannot open %s", filename);
      return;
   }

   FileWrite(handle, "time", "open", "high", "low", "close", "volume", "spread");

   for(int j = 0; j < copied; j++)
   {
      FileWrite(handle,
         (long)rates[j].time,
         rates[j].open,
         rates[j].high,
         rates[j].low,
         rates[j].close,
         rates[j].tick_volume,
         rates[j].spread
      );
   }

   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Process commands from Nandi (Python)                               |
//+------------------------------------------------------------------+
void ProcessCommands()
{
   if(!FileIsExist(CommandFile, FILE_COMMON))
      return;

   int handle = FileOpen(CommandFile, FILE_READ | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE) return;

   string parts[];
   string line = "";

   // Read the entire command line
   while(!FileIsEnding(handle))
   {
      string field = FileReadString(handle);
      if(StringLen(field) > 0)
      {
         if(StringLen(line) > 0) line += ",";
         line += field;
      }
   }
   FileClose(handle);

   // Delete command file after reading
   FileDelete(CommandFile, FILE_COMMON);

   if(StringLen(line) == 0) return;

   // Parse command
   StringSplit(line, ',', parts);
   if(ArraySize(parts) == 0) return;

   string cmd = parts[0];
   StringTrimRight(cmd);
   StringTrimLeft(cmd);

   // Execute command
   string response = "";

   if(cmd == "BUY" || cmd == "SELL")
      response = ExecuteOrder(parts);
   else if(cmd == "CLOSE")
      response = ClosePosition(parts);
   else if(cmd == "MODIFY")
      response = ModifyPosition(parts);
   else if(cmd == "CLOSE_ALL")
      response = CloseAllPositions();
   else if(cmd == "REFRESH_HISTORY")
      response = RefreshHistory(parts);
   else if(cmd == "PING")
      response = "OK,PONG," + IntegerToString(TimeCurrent());
   else
      response = "ERROR,Unknown command: " + cmd;

   // Write response
   WriteResponse(response);
}

//+------------------------------------------------------------------+
//| Execute a market order                                             |
//+------------------------------------------------------------------+
string ExecuteOrder(string &parts[])
{
   if(ArraySize(parts) < 3)
      return "ERROR,Insufficient parameters for order";

   string cmd = parts[0];
   string rawPair = parts[1];
   StringTrimRight(rawPair);
   StringTrimLeft(rawPair);

   string sym = BrokerSymbol(rawPair);
   double lots = StringToDouble(parts[2]);
   double sl = ArraySize(parts) > 3 ? StringToDouble(parts[3]) : 0;
   double tp = ArraySize(parts) > 4 ? StringToDouble(parts[4]) : 0;
   string comment = ArraySize(parts) > 5 ? parts[5] : "Nandi";

   // Enforce lot limits
   lots = MathMin(lots, MaxLotSize);
   double volMin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double volStep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   lots = MathMax(lots, volMin);
   lots = MathFloor(lots / volStep) * volStep;
   lots = NormalizeDouble(lots, 2);

   if(PaperMode)
   {
      double price = (cmd == "BUY") ? SymbolInfoDouble(sym, SYMBOL_ASK)
                                     : SymbolInfoDouble(sym, SYMBOL_BID);
      PrintFormat("[PAPER] %s %.2f lots %s @ %.5f | SL=%.5f TP=%.5f",
                  cmd, lots, sym, price, sl, tp);
      return StringFormat("OK,%d,%.5f", GetTickCount(), price);
   }

   // Real execution
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = sym;
   request.volume = lots;
   request.type = (cmd == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price = (cmd == "BUY") ? SymbolInfoDouble(sym, SYMBOL_ASK)
                                   : SymbolInfoDouble(sym, SYMBOL_BID);
   request.sl = sl;
   request.tp = tp;
   request.deviation = MaxSlippagePoints;
   request.magic = 20250308;  // Nandi magic number
   request.comment = comment;
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result))
   {
      return StringFormat("ERROR,OrderSend failed: %d (%s)",
                          result.retcode, GetRetcodeDescription(result.retcode));
   }

   if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
   {
      PrintFormat("EXECUTED: %s %.2f lots %s @ %.5f | ticket=%d",
                  cmd, lots, sym, result.price, result.deal);
      return StringFormat("OK,%d,%.5f", result.deal, result.price);
   }

   return StringFormat("ERROR,Retcode %d: %s", result.retcode, GetRetcodeDescription(result.retcode));
}

//+------------------------------------------------------------------+
//| Close a position by ticket                                         |
//+------------------------------------------------------------------+
string ClosePosition(string &parts[])
{
   if(ArraySize(parts) < 2)
      return "ERROR,No ticket specified";

   ulong ticket = (ulong)StringToInteger(parts[1]);

   if(PaperMode)
   {
      PrintFormat("[PAPER] CLOSE ticket=%d", ticket);
      return "OK,CLOSED";
   }

   if(!PositionSelectByTicket(ticket))
      return "ERROR,Position not found: " + IntegerToString(ticket);

   string sym = PositionGetString(POSITION_SYMBOL);
   double volume = PositionGetDouble(POSITION_VOLUME);
   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = sym;
   request.volume = volume;
   request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price = (posType == POSITION_TYPE_BUY) ? SymbolInfoDouble(sym, SYMBOL_BID)
                                                   : SymbolInfoDouble(sym, SYMBOL_ASK);
   request.deviation = MaxSlippagePoints;
   request.magic = 20250308;
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result))
      return StringFormat("ERROR,Close failed: %d", result.retcode);

   return "OK,CLOSED";
}

//+------------------------------------------------------------------+
//| Close all Nandi positions                                          |
//+------------------------------------------------------------------+
string CloseAllPositions()
{
   int closed = 0;
   int total = PositionsTotal();

   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetInteger(POSITION_MAGIC) == 20250308 ||
         StringFind(PositionGetString(POSITION_COMMENT), "Nandi") >= 0)
      {
         string dummyParts[];
         ArrayResize(dummyParts, 2);
         dummyParts[0] = "CLOSE";
         dummyParts[1] = IntegerToString(ticket);
         ClosePosition(dummyParts);
         closed++;
      }
   }

   return StringFormat("OK,CLOSED_ALL,%d", closed);
}

//+------------------------------------------------------------------+
//| Modify position SL/TP                                              |
//+------------------------------------------------------------------+
string ModifyPosition(string &parts[])
{
   if(ArraySize(parts) < 4)
      return "ERROR,Insufficient parameters for modify";

   ulong ticket = (ulong)StringToInteger(parts[1]);
   double newSL = StringToDouble(parts[2]);
   double newTP = StringToDouble(parts[3]);

   if(PaperMode)
   {
      PrintFormat("[PAPER] MODIFY ticket=%d SL=%.5f TP=%.5f", ticket, newSL, newTP);
      return "OK,MODIFIED";
   }

   if(!PositionSelectByTicket(ticket))
      return "ERROR,Position not found";

   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol = PositionGetString(POSITION_SYMBOL);
   request.sl = newSL;
   request.tp = newTP;

   if(!OrderSend(request, result))
      return StringFormat("ERROR,Modify failed: %d", result.retcode);

   return "OK,MODIFIED";
}

//+------------------------------------------------------------------+
//| Refresh historical data for a symbol                               |
//+------------------------------------------------------------------+
string RefreshHistory(string &parts[])
{
   if(ArraySize(parts) < 3)
      return "ERROR,Need symbol and bars count";

   string rawPair = parts[1];
   StringTrimRight(rawPair);
   StringTrimLeft(rawPair);
   string sym = BrokerSymbol(rawPair);
   int bars = (int)StringToInteger(parts[2]);

   ENUM_TIMEFRAMES tf = PERIOD_M5;
   if(ArraySize(parts) > 3)
   {
      string tfStr = parts[3];
      StringTrimRight(tfStr);
      if(tfStr == "D1") tf = PERIOD_D1;
      else if(tfStr == "H4") tf = PERIOD_H4;
      else if(tfStr == "H1") tf = PERIOD_H1;
      else if(tfStr == "M5") tf = PERIOD_M5;
      else if(tfStr == "M1") tf = PERIOD_M1;
   }

   MqlRates rates[];
   int copied = CopyRates(sym, tf, 0, bars, rates);
   if(copied <= 0)
      return "ERROR,No data for " + sym;

   // Write to fx_data.csv
   string dataFile = "fx_data.csv";
   int handle = FileOpen(dataFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return "ERROR,Cannot write data file";

   FileWrite(handle, "time", "open", "high", "low", "close", "volume");
   for(int i = 0; i < copied; i++)
   {
      FileWrite(handle,
         (long)rates[i].time,
         rates[i].open, rates[i].high, rates[i].low, rates[i].close,
         rates[i].tick_volume
      );
   }
   FileClose(handle);

   return StringFormat("OK,%d bars", copied);
}

//+------------------------------------------------------------------+
//| Write response file                                                |
//+------------------------------------------------------------------+
void WriteResponse(string response)
{
   int handle = FileOpen(ResponseFile, FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot write response file");
      return;
   }
   FileWriteString(handle, response);
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Get human-readable retcode description                             |
//+------------------------------------------------------------------+
string GetRetcodeDescription(uint retcode)
{
   switch(retcode)
   {
      case TRADE_RETCODE_REQUOTE:       return "Requote";
      case TRADE_RETCODE_REJECT:        return "Rejected";
      case TRADE_RETCODE_CANCEL:        return "Cancelled";
      case TRADE_RETCODE_PLACED:        return "Placed";
      case TRADE_RETCODE_DONE:          return "Done";
      case TRADE_RETCODE_DONE_PARTIAL:  return "Partial fill";
      case TRADE_RETCODE_ERROR:         return "Error";
      case TRADE_RETCODE_TIMEOUT:       return "Timeout";
      case TRADE_RETCODE_INVALID:       return "Invalid request";
      case TRADE_RETCODE_INVALID_VOLUME: return "Invalid volume";
      case TRADE_RETCODE_INVALID_PRICE:  return "Invalid price";
      case TRADE_RETCODE_INVALID_STOPS:  return "Invalid stops";
      case TRADE_RETCODE_TRADE_DISABLED: return "Trading disabled";
      case TRADE_RETCODE_MARKET_CLOSED:  return "Market closed";
      case TRADE_RETCODE_NO_MONEY:       return "Insufficient funds";
      case TRADE_RETCODE_PRICE_CHANGED:  return "Price changed";
      case TRADE_RETCODE_PRICE_OFF:      return "Price off";
      case TRADE_RETCODE_INVALID_EXPIRATION: return "Invalid expiration";
      case TRADE_RETCODE_ORDER_CHANGED:  return "Order changed";
      case TRADE_RETCODE_TOO_MANY_REQUESTS: return "Too many requests";
      default: return "Unknown (" + IntegerToString(retcode) + ")";
   }
}
//+------------------------------------------------------------------+
