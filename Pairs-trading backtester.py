# +1 --> long xom short beta*cvx
# -1 --> short xom long beta*cvx
# row then column
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 1000)
optimising = True
out_of_sample = not optimising


def returns_calc(prices):
    returns = []
    for i in range(len(prices)-1):
        rt=(prices[i+1]-prices[i])/prices[i]
        returns.append(rt)
    return returns

def reversion_checker(signals):
    reversion_rates = []
    for k in [5, 10, 20, 40]:
        reverted = 0
        observations = 0
        for j in signals:
            if abs(residuals[j]) > abs(residuals[j + k]):
                reverted += 1
            observations += 1
        reversion_rates.append(reverted/observations)
    return reversion_rates

class Stock:
    def __init__(self, filename):
        temp = pd.read_csv(filename, nrows=1500)
        self.prices = temp['Close-Last']
        self.dates = (np.array(temp['Date'].tolist(), dtype='datetime64'))

xom = Stock('XOM.csv')
cvx = Stock('CVX.csv')
window = 252


alphas = [np.nan]*window
betas = [np.nan]*window
for i in range(window, len(cvx.prices)):
    rolling_cvx_prices = cvx.prices[i - window:i]
    rolling_xom_prices = xom.prices[i - window:i]
    b, a, r, p, se = sp.stats.linregress(rolling_cvx_prices, rolling_xom_prices)
    alphas.append(a)
    betas.append(b)
residuals = xom.prices - alphas - betas*cvx.prices

def pl_tester(upper, lower):
    df = pd.DataFrame({'Date' : xom.dates,
                       'CVX' : cvx.prices,
                       'XOM' : xom.prices,
                       'beta' : betas,
                       'Residuals': residuals
                       })

    df['rolling mean'] = df['Residuals'].rolling(window).mean()
    df['rolling stds'] = df['Residuals'].rolling(window).std()
    df['rolling z scores'] = (df['Residuals'] - df['rolling mean'])/df['rolling stds']

    positions=[0]
    for i in range(1,len(df)):
        if df['rolling z scores'][i] > upper:
            positions.append(-1)
        elif df['rolling z scores'][i] < lower:
            positions.append(1)
        elif abs(df['rolling z scores'][i]) < 0.5:
            positions.append(0)
        else:
            positions.append(positions[i-1])
    df['positions'] = positions
    df['xom return pct'] = df['XOM'].pct_change()
    df['cvx return pct'] = df['CVX'].pct_change()

    entry_betas = [0]
    entry_index = []
    for i in range(1, len(df)):
        if df['positions'][i] != 0 and df['positions'][i] != df['positions'][i-1]:
            entry_betas.append(df['beta'][i])
            entry_index.append(i)
        elif df['positions'][i] == df['positions'][i-1]:
            entry_betas.append(entry_betas[i-1])
        else:
            entry_betas.append(0)
    df['entry_betas'] = entry_betas

    xom_notional = [0]*(len(df))
    cvx_notional = [0]*(len(df))
    xom_daily_pl = [0]*(len(df))
    cvx_daily_pl = [0]*(len(df))

    base_cost = 1000
    # print(entry_index)
    # print(len(entry_index))
    trade_length = []
    for i in entry_index:
        trading = True
        trade_days=1
        xom_notional[i] = df['positions'][i] * base_cost
        cvx_notional[i] = -1 * df['positions'][i] * base_cost * df['entry_betas'][i]
        while trading:
            xom_daily_pl[i + trade_days] = xom_notional[i + trade_days - 1] * df['xom return pct'][i + trade_days]
            cvx_daily_pl[i + trade_days] = cvx_notional[i + trade_days - 1] * df['cvx return pct'][i + trade_days]

            if df['positions'][i + trade_days] != df['positions'][i + trade_days - 1] and abs(df['positions'][i + trade_days - 1]) == 1:
                trading = False
                trade_length.append(trade_days)
            else:
                xom_notional[i + trade_days] = xom_notional[i + trade_days - 1]
                cvx_notional[i + trade_days] = cvx_notional[i + trade_days - 1]
                trade_days += 1
            if i+trade_days == len(df):
                trade_length.append(trade_days-1)
                break

    df['xom notional'] = xom_notional
    df['cvx notional'] = cvx_notional
    df['xom p/l'] = xom_daily_pl
    df['cvx p/l'] = cvx_daily_pl
    df['daily p/l'] = df['xom p/l'] + df['cvx p/l']
    df['cumulative p/l'] = df['daily p/l'].cumsum()
    opening_pl = df['cumulative p/l'][entry_index]
    closing_pl = df['cumulative p/l'][np.add(entry_index, trade_length)]
    per_trade_p_l = closing_pl.values - opening_pl.values
    return df, len(entry_index), per_trade_p_l

def vary_upper_lower():
    uppers = np.round(np.linspace(1, 3, 11), 1)
    lowers = np.round(np.linspace(-3, -1    , 11), 1)
    pl_results = np.empty((11,11))
    annualised_sharpes = np.empty((11,11))
    winrates = np.empty((11,11))
    max_drawdowns = np.empty((11,11))
    profit_factors = np.empty((11,11))
    n_trades = np.empty((11,11))
    for i in range(len(uppers)):
        for j in range(len(lowers)):
            wincount, totalwin, losscount, totalloss = (0, 0, 0, 0)
            temp, ntrades, trade_pl = pl_tester(uppers[i], lowers[j])
            pl_results[i, j] = temp['cumulative p/l'].iloc[-1]
            mean_daily_return = temp['daily p/l'].mean()
            std_daily_return = temp['daily p/l'].std()
            sharpe_daily = mean_daily_return / std_daily_return
            annualised_sharpes[i, j] = sharpe_daily * np.sqrt(252)
            n_trades[i, j] = ntrades
            for net in trade_pl:
                if net > 0:
                    wincount += 1
                    totalwin += net
                elif net < 0:
                    losscount += 1
                    totalloss += net
            if losscount == 0:
                avg_loss = 0
            else:
                avg_loss = totalloss / losscount
            if wincount == 0:
                avg_win = 0
            else:
                avg_win = totalwin / wincount
            if totalloss == 0:
                profit_factor = np.nan
            else:
                profit_factor = totalwin / abs(totalloss)
            winrate = wincount / ntrades
            running_max = temp['cumulative p/l'].cummax()
            drawdown = temp['cumulative p/l'] - running_max
            max_drawdown = drawdown.min()
            max_drawdowns[i, j] = max_drawdown
            winrates[i, j] = winrate
            profit_factors[i, j] = profit_factor

    return uppers, lowers, pl_results, annualised_sharpes, winrates, max_drawdowns, profit_factors, n_trades

if optimising:
    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    uppers, lowers, pl_res, ann_sharpes, winrates, max_drawdowns, profitfactors, ntrades = vary_upper_lower()

    axs[0,0].set_xticks(range(len(lowers)), labels=lowers)
    axs[0,0].set_yticks(range(len(uppers)), labels=uppers)
    pl_map = axs[0,0].imshow(pl_res)
    axs[0,0].set_xlabel('Lower signal bound')
    axs[0,0].set_ylabel('Upper signal bound')
    axs[0,0].set_title('Profit / Loss')
    fig.colorbar(pl_map, ax=axs[0,0])

    axs[0,1].set_xticks(range(len(lowers)), labels=lowers)
    axs[0,1].set_yticks(range(len(uppers)), labels=uppers)
    ann_sharpes_map = axs[0,1].imshow(ann_sharpes)
    axs[0,1].set_xlabel('Lower signal bound')
    axs[0,1].set_ylabel('Upper signal bound')
    axs[0,1].set_title('Annualised Sharpe Ratio')
    fig.colorbar(ann_sharpes_map, ax=axs[0,1])

    axs[0,2].set_xticks(range(len(lowers)), labels=lowers)
    axs[0,2].set_yticks(range(len(uppers)), labels=uppers)
    winrates_map = axs[0,2].imshow(winrates)
    axs[0,2].set_xlabel('Lower signal bound')
    axs[0,2].set_ylabel('Upper signal bound')
    axs[0,2].set_title('Winrate')
    fig.colorbar(winrates_map, ax=axs[0,2])

    axs[1,0].set_xticks(range(len(lowers)), labels=lowers)
    axs[1,0].set_yticks(range(len(uppers)), labels=uppers)
    max_drawdowns_map = axs[1,0].imshow(max_drawdowns)
    axs[1,0].set_xlabel('Lower signal bound')
    axs[1,0].set_ylabel('Upper signal bound')
    axs[1,0].set_title('Max Drawdown')
    fig.colorbar(max_drawdowns_map, ax=axs[1,0])

    axs[1,1].set_xticks(range(len(lowers)), labels=lowers)
    axs[1,1].set_yticks(range(len(uppers)), labels=uppers)
    profitfactors_map = axs[1,1].imshow(profitfactors)
    axs[1,1].set_xlabel('Lower signal bound')
    axs[1,1].set_ylabel('Upper signal bound')
    axs[1,1].set_title('Profit Factor')
    fig.colorbar(profitfactors_map, ax=axs[1,1])

    axs[1,2].set_xticks(range(len(lowers)), labels=lowers)
    axs[1,2].set_yticks(range(len(uppers)), labels=uppers)
    ntrades_map = axs[1,2].imshow(ntrades)
    axs[1,2].set_xlabel('Lower signal bound')
    axs[1,2].set_ylabel('Upper signal bound')
    axs[1,2].set_title('No of trades')
    fig.colorbar(ntrades_map, ax=axs[1,2])
    fig.suptitle('Trading Backtest: XOM vs CVX')


if out_of_sample:
    #fig, axs = plt.subplots(2, 2, constrained_layout=True)
    u_bound, l_bound = (1.8, -1.0)
    df , n_trades, trade_pl = pl_tester(u_bound, l_bound)
    mean_daily_return = df['daily p/l'].mean()
    std_daily_return = df['daily p/l'].std()
    sharpe_daily = mean_daily_return / std_daily_return
    sharpe_annual = sharpe_daily * np.sqrt(252)
    wincount, totalwin, losscount, totalloss = (0, 0, 0, 0)
    for net in trade_pl:
        if net > 0:
            wincount += 1
            totalwin += net
        elif net < 0:
            losscount += 1
            totalloss += net
    if losscount == 0:
        avg_loss = 0
    else:
        avg_loss = totalloss / losscount
    if wincount == 0:
        avg_win = 0
    else:
        avg_win = totalwin / wincount
    if totalloss == 0:
        profit_factor = np.nan
    else:
        profit_factor = totalwin / abs(totalloss)
    winrate = wincount / n_trades
    running_max = df['cumulative p/l'].cummax()
    drawdown = df['cumulative p/l'] - running_max
    max_drawdown = drawdown.min()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    print('upper', u_bound, 'and lower', l_bound)
    print('no of trades is ', n_trades, ',generating $', df['cumulative p/l'].iloc[-1])
    print('avg win is', avg_win, ',avg loss is', avg_loss)
    print('profit factor is', profit_factor)
    print('max drawdown is', max_drawdown)
    print('annualised sharpe is', sharpe_annual)
    print(winrate, ' is the winrate')

plt.show()