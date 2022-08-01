import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the code is based on a book by Paolo Brandimarte (Numerical Methods in finance and economics)
# the original code was in MATLAB and we coverted the code to Python
# some assistance has also been taken from Professor Josef Dick's notes in Mathematical Computing for finance
# again the code was in MATLAB and we had to convert eveything to Python

"""
Prices European call option using explicit finite difference method
-- Input parameters --
K  = strike price
T  = expiry time
r  = risk-free interest rate
sigma = volatility
-- Output arguments --
V  = values of option at asset values in S for varying time values in t
"""


def bspdeexp(K, t, r, sigma, S):
    dt = t[1] - t[0]
    dS = S[1] - S[0]
    V = torch.ones((S.size()[0], t.size()[0]), dtype=torch.float64)
    V[0, :] = 0
    V[-1, :] = S[-1] - K * torch.exp(-r*(t[-1] - t))
    V[1:-1, -1] = torch.maximum(S[1:-1] - K,
                                torch.zeros((S.size()[0]-2), dtype=torch.float64))
    c1 = sigma**2 * S**2 * dt / (2 * dS ** 2)
    c2 = r * S * dt / (2 * dS)
    alpha = c1 - c2
    beta = 1 - r*dt - 2*c1
    gamma = c1 + c2
    for n in range(t.size()[0] - 1, 0, -1):
        V[1:S.size()[0] - 1, n - 1] = alpha[1:S.size()[0] - 1] * V[0:S.size()[0] - 2, n] + \
            beta[1:S.size()[0] - 1] * V[1:S.size()[0] - 1, n] + \
            gamma[1:S.size()[0] - 1] * V[2:S.size()[0], n]
    return V
