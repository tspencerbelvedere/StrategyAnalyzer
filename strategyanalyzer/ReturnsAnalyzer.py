import pandas as pd
import numpy as np
import talib as ta
from collections import OrderedDict
import pandas_market_calendars as mcal

class ReturnsHelperFunctions(object):
    def __init__(self):
        pass
        
    @staticmethod
    def GetReturnsLengthInYears(returns):
        years = (returns.index.max() - returns.index.min()).days / 365.
        return years
    
    @staticmethod
    def GetCumulativeReturns(returns, startDate=None, endDate=None):
        rets = returns.copy(deep=True)
        if startDate is not None:
            if endDate is not None:
                rets = rets.loc[startDate:endDate] 
            else:
                rets = rets.loc[startDate:]
        else:
            rets = rets.loc[:endDate]
        
        return (1 + rets).cumprod()

    @staticmethod
    def GetPercentWinners(returns):
        winners = len(returns[returns > 0])
        losers = len(returns[returns < 0])
        return winners * 1.0 / (winners + losers)
    
    @staticmethod
    def GetIrr(returns):
        cumulative = ReturnsHelperFunctions.GetCumulativeReturns(returns)
        return np.log(cumulative.dropna().values[-1]) / ReturnsHelperFunctions.GetReturnsLengthInYears(returns)
    
    @staticmethod
    def GetVol(returns):
        return returns.std() * np.sqrt(252.)
    
    @staticmethod
    def GetMaxDrawDown(returns):
        return ReturnsHelperFunctions.GetDrawDown(returns).min()

    @staticmethod
    def GetDrawDown(returns):
        cumulative = ReturnsHelperFunctions.GetCumulativeReturns(returns)
        drawdown = (cumulative / cumulative.cummax() - 1)
        return drawdown

    @staticmethod
    def GetLongestDrawdown(returns):
        drawdown = ReturnsHelperFunctions.GetDrawDown(returns)
        drawDownCount = 0
        drawDownStreak = [0]
        for ret in range(1, len(drawdown.values)):
            if drawdown.values[ret] < 0:
                drawDownCount += 1
            else:
                drawDownCount = 0
            drawDownStreak.append(drawDownCount)
            
        return max(drawDownStreak)

    @staticmethod
    def GetSharpe(returns):
        return returns.replace(0.0,np.nan).dropna().mean() \
            / returns.replace(0.0,np.nan).dropna().std() * np.sqrt(252.)

    @staticmethod
    def GetSortino(returns):
        rets = returns.replace(0.0,np.nan).dropna()
        mean = rets.mean()
        downsideStdev = rets[rets < 0].std()
        return (mean / downsideStdev) * np.sqrt(252)

    @staticmethod
    def GetCapitalUsage(returns, calendar):
        totalDays = calendar.schedule(returns.index.min(), returns.index.max()).shape[0]
        daysInMarket = len(returns[returns.abs() != 0])
        return float(daysInMarket) / totalDays

    @staticmethod
    def GetCalmarRatio(returns):
        irr = ReturnsHelperFunctions.GetIrr(returns)
        dd = ReturnsHelperFunctions.GetMaxDrawDown(returns)
        return irr / abs(dd)
    
    

class ReturnsAnalyzer(ReturnsHelperFunctions):
    def __init__(self, returns, ticker, calendar):
        self.returns = returns
        self.ticker = ticker
        self.calendar = calendar
        self.report = self.GetReport()
        
    @property
    def returns(self):
        return self._returns
    
    @returns.setter
    def returns(self, returns):
        if not isinstance(returns, pd.Series):
            raise ValueError('Must be pandas series: {}'.format(type(returns)))
        
        returns.index = pd.to_datetime(returns.index)
        returns = returns.dropna()
        self._returns = returns

    def GetReport(self):
        irr = ReturnsHelperFunctions.GetIrr(self.returns)
        vol = ReturnsHelperFunctions.GetVol(self.returns)
        maxdd = ReturnsHelperFunctions.GetMaxDrawDown(self.returns)
        sharpe = ReturnsHelperFunctions.GetSharpe(self.returns)
        sortino = ReturnsHelperFunctions.GetSortino(self.returns)
        pctWinners = ReturnsHelperFunctions.GetPercentWinners(self.returns)
        longestDrawdown = ReturnsHelperFunctions.GetLongestDrawdown(self.returns)
        pctDaysInMarket = ReturnsHelperFunctions.GetCapitalUsage(self.returns, self.calendar)
        calmar = ReturnsHelperFunctions.GetCalmarRatio(self.returns)
        resultsDict = OrderedDict(IRR=irr,
        Maxdd=maxdd, Calmar=calmar, Longestdd=longestDrawdown,
        PctDaysInMarket=pctDaysInMarket,
        Sharpe=sharpe, Sortino=sortino, 
        PctWinners=pctWinners, Vol=vol)
        
        return pd.DataFrame({self.ticker: resultsDict})


class OscillatorBacktester(object):
    def __init__(self, prices):
        self._prices = prices
        
    def EvaluateStrategy(self, rsiLength, willrLength, maxRsi, maxWillr, leverage=1):
        closes = self._prices[['Adj Close']]
        closes['RSI'] = ta.RSI(closes['Adj Close'], rsiLength)
        closes['WillR'] = ta.WILLR(self._prices['High'], self._prices['Low'],
                                   self._prices['Adj Close'], willrLength).values
        closes['NextReturn'] = closes['Adj Close'].pct_change().shift(-1)
        closes['StrategyReturn'] = closes['NextReturn'] * leverage * (closes['WillR'] < maxWillr) * \
        (closes['RSI'] < maxRsi)
        return closes
    