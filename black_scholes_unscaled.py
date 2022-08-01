from scipy.stats import norm
import numpy as np

"""
Black and Scholes formula for the value of a call option
S = underlying asset price
K = strike price
r = risk-free interest rate
Tmt = time to maturity = T - t where T = expiry
sigma = volatility parameter
"""


def blackscholes_unscaled(S, K, r, sigma, Tmt):
    s = sigma * np.sqrt(Tmt)
    d1 = (np.log(S/K) + (r + sigma**2/2)*(Tmt)) / s
    d2 = d1 - s
    optionValue = S * norm.cdf(d1) - K * np.exp(-r*Tmt) * norm.cdf(d2)
    return optionValue
