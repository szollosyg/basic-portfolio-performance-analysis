from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd
import matplotlib.pyplot as plt

def perfAnalysis(eq_tickers: list, start: date, end: date, riskfree_rate=0.00, portf_weights=[], init_cap=1000, chart_size=(18,10) ):

    """
    Arguments:
        eq_tickers - the list of tickers to analyze
        start - start date of the analysis period
        end - end date of the analysis period
        riskfree_rate - riskfree rate approximation for the Sharpe ratio calculation
        portf_weights - the weights of the analyzed equities. If remains empty the portfolio won't be calculated
                        The number of tickers must match to the number of portfolio weights. 
                        The weights must add up to 1. 
        init_cap - initial capital when the wealth index is calculated
        chart_size
                        
    Returns: Summary stats of the equities and optionally the portfolio for the given time period. 
    """
    
    
    assert ( len(portf_weights)==0 or (len(portf_weights)==len(eq_tickers))),"The number of tickers must match to the number of portfolio weights"
    if len(portf_weights)==len(eq_tickers):
        assert ( 1.01 > sum(portf_weights) > 0.99), "The portfolio weights must add up to 1. "
        
    
    eq_prices = {}
    eq_rets = {}
    eq_dd = {}
    
    for ticker in eq_tickers:
        # Get price, returns and drawdowns
        eq_prices[ticker] = pdr.get_data_yahoo(ticker, start=start, end=end)
        eq_prices[ticker].index = eq_prices[ticker].index.to_period('D')
        eq_rets[ticker] = eq_prices[ticker]["Adj Close"].pct_change()
        eq_dd[ticker] = drawdown(eq_rets[ticker], init_cap)
        
    # if portfolio weights were provided then calculate portfolio returns and drawdowns
    if portf_weights:
        eq_rets['Portf'] = calcPortfRets(eq_rets, eq_tickers, portf_weights)
        eq_dd['Portf'] = drawdown(eq_rets['Portf'], init_cap)
        eq_tickers.append('Portf')

    # Calc summary stats
    summary_stats = pd.DataFrame(index=eq_tickers, columns=['Cumulative return($)', 'Annualized return', 'Annualized volatility', 'Annualized Sharpe ratio', 'Max drawdown'])    
    for ticker in eq_tickers:
        cumulative_return = eq_dd[ticker]["Wealth"][-1]-1
        n_days = eq_rets[ticker].shape[0]
        annualized_return = (eq_rets[ticker]+1).prod()**(252/n_days) - 1
        annualized_volatility = eq_rets[ticker].std()*252**0.5 # we have daily data and 252 trading days a year
        annualized_sharpe = (annualized_return - riskfree_rate) / annualized_volatility
        max_drawdown = eq_dd[ticker]["Drawdowns"].min()
        
        # Fill up stats dataframe
        summary_stats.loc[ticker] = [cumulative_return, annualized_return, annualized_volatility, annualized_sharpe, max_drawdown]
        
    drawChart(eq_dd, chart_size=chart_size, line_style=':') if portf_weights else drawChart(eq_dd, chart_size=chart_size, line_style='-')
    
    return summary_stats

def calcPortfRets(eq_rets: dict, eq_tickers: list, portf_weights: list):
    
    #create index-only dataframe
    df_rets = pd.DataFrame(index=eq_rets[eq_tickers[0]].index)

    # add ticker data columns
    for ticker in eq_tickers:
        df_rets = pd.concat([df_rets, eq_rets[ticker]], axis=1) 
    df_rets.columns = [ticker for ticker in eq_tickers]
    # create Portfolio Return series as the weighted sum of the ticker returns
    portf_rets = df_rets.dot(pd.Series(portf_weights, index=eq_tickers, name=0))

    return portf_rets

def drawdown(return_series: pd.Series, init_cap: int):
    """
    Takes a pandas time series of asset returns
    Return pandas DF that contains:
    - wealth index
    - previous peaks
    - precent drawdowns
    """
    wealth_index = init_cap*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    
    return pd.DataFrame({
        "Wealth":wealth_index,
        "Peaks": previous_peaks,
        "Drawdowns": drawdowns
    })

def drawChart(eq_dds: dict, chart_size, line_style: str):
    
    colors = ['royalblue', 'orange', 'olive', 'darkgreen', 'pink', 'magenta', 'darkcyan', 'darkorchid']
    
    plt.figure(figsize=chart_size)

    plt.subplot(211)
    plt.title("Wealth index ($)")
    for i,ticker in enumerate(eq_dds):
        line_color = 'firebrick' if ticker == 'Portf' else colors[i%len(colors)]
        style = '-' if ticker == 'Portf' else line_style
        eq_dds[ticker]["Wealth"].plot.line(color=line_color, label=ticker, legend=True, style=style)
    plt.grid(True)

    plt.subplot(212)
    plt.title("Drawdown")
    for i,ticker in enumerate(eq_dds):
        line_color = 'firebrick' if ticker == 'Portf' else colors[i%len(colors)]
        style = '-' if ticker == 'Portf' else line_style
        eq_dds[ticker]["Drawdowns"].plot.line(color=line_color, label=ticker, legend=True, style=style)
    plt.grid(True)

    plt.subplots_adjust(hspace=0.4)
    plt.show()
    
        