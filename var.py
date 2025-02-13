import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import yfinance as yf

def main():
    nifty50 = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS','HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS']
    data = yf.download(nifty50, start="2023-08-29", end="2024-08-29")['Adj Close']

    ret = data.pct_change().dropna()
    

    geranw = len(nifty50)
    weights = np.full(geranw, 1/geranw)

    c = float(input(f"Enter Confindence level in decimal value: "))
    z_scr = stats.norm.ppf(c)

    method1_var = abs(variance_meth(ret,weights,z_scr))
    method2_var = abs(historic_meth(ret,c,weights))
    method3_var = abs(montecarlo(ret,c,weights))
    #print(f"The weight distribution for this portfolio is: {weights}")
    print(f"VaR using Variance Method: {method1_var*100:.3f}%")
    print(f"VaR using Historical Method: {method2_var*100:.3f}%")
    print(f"VaR using Monte Carlo Method: {method3_var*100:.3f}%")


#df is just industry practice??
def montecarlo(retns,conf,weigh):
    meanret = retns.mean()
    depon = retns.cov()
    simp = np.random.multivariate_normal(meanret, depon, 1000000)
    frets = simp.dot(weigh)
    return np.percentile(frets, (1 - conf) * 100)

def historic_meth(retns,conf,weigh):
    portret = retns.dot(weigh)
    portret = np.sort(portret)
    wpf = int((1-conf)*len(portret))

    return portret[wpf]    


def variance_meth(retns,weigh,zc):
    meanret = retns.mean()
    #print(meanret)
    fmean = np.dot(meanret,weigh)
    depon = retns.cov()
    portvar = np.dot(weigh.T,np.dot(depon,weigh))
    stddev = np.sqrt(portvar)

    return fmean + stddev*zc


if __name__ == "__main__":
    main()