//+------------------------------------------------------------------+
//| ForexPredictor.mq5 - File Bridge Expert Advisor                  |
//| Communicates with Python via files in MQL5/Files folder           |
//+------------------------------------------------------------------+
#property copyright "ForexPredictor"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

input string SymbolToTrade = "EURUSD";  // Trading symbol
input int    CheckInterval = 1000;       // Check interval ms

CTrade trade;
CPositionInfo posInfo;

string cmdFile = "fx_command.csv";
string respFile = "fx_response.csv";
string dataFile = "fx_data.csv";
string tickFile = "fx_tick.csv";
string accountFile = "fx_account.csv";
string positionsFile = "fx_positions.csv";

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(123456);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   EventSetMillisecondTimer(CheckInterval);

   // Write initial data
   WriteAccountInfo();
   WriteTickData(SymbolToTrade);
   WriteHistoryData(SymbolToTrade, PERIOD_M5, 20000);
   WritePositions();

   Print("ForexPredictor EA started for ", SymbolToTrade);
   Print("Files location: MQL5/Files/");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("ForexPredictor EA stopped");
}

//+------------------------------------------------------------------+
void OnTimer()
{
   // Always update tick and account
   WriteTickData(SymbolToTrade);
   WriteAccountInfo();
   WritePositions();

   // Check for commands from Python
   if(FileIsExist(cmdFile))
   {
      ProcessCommand();
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   WriteTickData(SymbolToTrade);
}

//+------------------------------------------------------------------+
void WriteHistoryData(string symbol, ENUM_TIMEFRAMES tf, int bars)
{
   MqlRates rates[];
   int copied = CopyRates(symbol, tf, 0, bars, rates);

   if(copied <= 0)
   {
      Print("Failed to copy rates for ", symbol);
      return;
   }

   int handle = FileOpen(dataFile, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("Failed to open data file");
      return;
   }

   FileWrite(handle, "time", "open", "high", "low", "close", "volume");

   for(int i = 0; i < copied; i++)
   {
      FileWrite(handle,
         (int)rates[i].time,
         DoubleToString(rates[i].open, 5),
         DoubleToString(rates[i].high, 5),
         DoubleToString(rates[i].low, 5),
         DoubleToString(rates[i].close, 5),
         (int)rates[i].tick_volume
      );
   }

   FileClose(handle);
   Print("Wrote ", copied, " bars to ", dataFile);
}

//+------------------------------------------------------------------+
void WriteTickData(string symbol)
{
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return;

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double lotMin = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double lotMax = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);

   int handle = FileOpen(tickFile, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE) return;

   FileWrite(handle, "field", "value");
   FileWrite(handle, "bid", DoubleToString(tick.bid, digits));
   FileWrite(handle, "ask", DoubleToString(tick.ask, digits));
   FileWrite(handle, "time", (int)tick.time);
   FileWrite(handle, "volume", (int)tick.volume);
   FileWrite(handle, "symbol", symbol);
   FileWrite(handle, "point", DoubleToString(point, 6));
   FileWrite(handle, "digits", IntegerToString(digits));
   FileWrite(handle, "trade_tick_value", DoubleToString(tickValue, 4));
   FileWrite(handle, "trade_tick_size", DoubleToString(tickSize, 6));
   FileWrite(handle, "volume_min", DoubleToString(lotMin, 2));
   FileWrite(handle, "volume_max", DoubleToString(lotMax, 2));
   FileWrite(handle, "volume_step", DoubleToString(lotStep, 2));
   FileWrite(handle, "spread", IntegerToString(spread));

   FileClose(handle);
}

//+------------------------------------------------------------------+
void WriteAccountInfo()
{
   int handle = FileOpen(accountFile, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE) return;

   FileWrite(handle, "field", "value");
   FileWrite(handle, "balance", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   FileWrite(handle, "equity", DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2));
   FileWrite(handle, "margin", DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN), 2));
   FileWrite(handle, "margin_free", DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2));
   FileWrite(handle, "profit", DoubleToString(AccountInfoDouble(ACCOUNT_PROFIT), 2));
   FileWrite(handle, "currency", AccountInfoString(ACCOUNT_CURRENCY));
   FileWrite(handle, "leverage", IntegerToString(AccountInfoInteger(ACCOUNT_LEVERAGE)));

   FileClose(handle);
}

//+------------------------------------------------------------------+
void WritePositions()
{
   int handle = FileOpen(positionsFile, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE) return;

   FileWrite(handle, "ticket", "symbol", "type", "volume", "price_open", "sl", "tp", "profit", "comment");

   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!posInfo.SelectByIndex(i)) continue;

      FileWrite(handle,
         IntegerToString(posInfo.Ticket()),
         posInfo.Symbol(),
         IntegerToString(posInfo.PositionType()),
         DoubleToString(posInfo.Volume(), 2),
         DoubleToString(posInfo.PriceOpen(), 5),
         DoubleToString(posInfo.StopLoss(), 5),
         DoubleToString(posInfo.TakeProfit(), 5),
         DoubleToString(posInfo.Profit(), 2),
         posInfo.Comment()
      );
   }

   FileClose(handle);
}

//+------------------------------------------------------------------+
void ProcessCommand()
{
   int handle = FileOpen(cmdFile, FILE_READ|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE) return;

   string action = FileReadString(handle);

   string result = "ERROR,Unknown command";

   if(action == "BUY" || action == "SELL")
   {
      string symbol = FileReadString(handle);
      double lotSize = StringToDouble(FileReadString(handle));
      double sl = StringToDouble(FileReadString(handle));
      double tp = StringToDouble(FileReadString(handle));
      string comment = FileReadString(handle);

      result = ExecuteTrade(action, symbol, lotSize, sl, tp, comment);
   }
   else if(action == "CLOSE")
   {
      ulong ticket = (ulong)StringToInteger(FileReadString(handle));
      result = CloseTrade(ticket);
   }
   else if(action == "MODIFY")
   {
      ulong ticket = (ulong)StringToInteger(FileReadString(handle));
      double sl = StringToDouble(FileReadString(handle));
      double tp = StringToDouble(FileReadString(handle));
      result = ModifyTrade(ticket, sl, tp);
   }
   else if(action == "REFRESH_HISTORY")
   {
      string symbol = FileReadString(handle);
      int bars = (int)StringToInteger(FileReadString(handle));
      string tf = FileReadString(handle);
      ENUM_TIMEFRAMES timeframe = StringToTimeframe(tf);
      WriteHistoryData(symbol, timeframe, bars);
      result = "OK,History refreshed";
   }

   FileClose(handle);
   FileDelete(cmdFile);

   // Write response
   int respHandle = FileOpen(respFile, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(respHandle != INVALID_HANDLE)
   {
      FileWrite(respHandle, result);
      FileClose(respHandle);
   }
}

//+------------------------------------------------------------------+
string ExecuteTrade(string type, string symbol, double lotSize, double sl, double tp, string comment)
{
   ENUM_ORDER_TYPE orderType;
   double price;

   if(type == "BUY")
   {
      orderType = ORDER_TYPE_BUY;
      price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   }
   else
   {
      orderType = ORDER_TYPE_SELL;
      price = SymbolInfoDouble(symbol, SYMBOL_BID);
   }

   if(!trade.PositionOpen(symbol, orderType, lotSize, price, sl, tp, comment))
   {
      return "ERROR,Trade failed: " + trade.ResultComment();
   }

   return "OK," + IntegerToString(trade.ResultOrder()) + "," + DoubleToString(trade.ResultPrice(), 5);
}

//+------------------------------------------------------------------+
string CloseTrade(ulong ticket)
{
   if(!trade.PositionClose(ticket))
   {
      return "ERROR,Close failed: " + trade.ResultComment();
   }
   return "OK,Position closed";
}

//+------------------------------------------------------------------+
string ModifyTrade(ulong ticket, double sl, double tp)
{
   if(!trade.PositionModify(ticket, sl, tp))
   {
      return "ERROR,Modify failed: " + trade.ResultComment();
   }
   return "OK,Position modified";
}

//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tf)
{
   if(tf == "M1")  return PERIOD_M1;
   if(tf == "M5")  return PERIOD_M5;
   if(tf == "M15") return PERIOD_M15;
   if(tf == "M30") return PERIOD_M30;
   if(tf == "H1")  return PERIOD_H1;
   if(tf == "H4")  return PERIOD_H4;
   if(tf == "D1")  return PERIOD_D1;
   if(tf == "W1")  return PERIOD_W1;
   return PERIOD_M5;
}
//+------------------------------------------------------------------+
