import torch

# the code is based on a book by Paolo Brandimarte (Numerical Methods in finance and economics)
# the original code was in MATLAB and we coverted the code to Python

"""
Prices European call option using Crank-Nicolson finite difference method
-- Input parameters --
K  = strike price
T  = expiry time
r  = risk-free interest rate
sigma = volatility
-- Output arguments --
V  = values of option at asset values in S for varying time values in t
"""


def bspdeCN(K, t, r, sigma, S):
    dt = t[1] - t[0]
    dS = S[1] - S[0]
    V = torch.zeros((S.size()[0], t.size()[0]), dtype=torch.float64)
    V[0, :] = 0
    V[-1, :] = S[-1] - K * torch.exp(-r*(t[-1] - t))
    V[1:-1, -1] = torch.maximum(S[1:-1] - K,
                                torch.zeros((S.size()[0]-2), dtype=torch.float64))
    S_sub = S/dS

    alpha = 0.25 * dt * (sigma**2 * (S_sub ** 2) - r * S_sub)
    beta = -dt * 0.5 * (sigma ** 2 * (S_sub ** 2) + r)
    gamma = 0.25 * dt * (sigma ** 2 * (S_sub ** 2) + r * S_sub)

    M1 = -torch.diag(alpha[2:S.size()[0]-1], -1) + torch.diag(1 -
                                                              beta[1:S.size()[0]-1]) - torch.diag(gamma[1:S.size()[0]-2], 1)
    M2 = torch.diag(alpha[2:S.size()[0]-1], -1) + torch.diag(1 +
                                                             beta[1:S.size()[0]-1]) + torch.diag(gamma[1:S.size()[0]-2], 1)
    aux = torch.zeros(S.size()[0] - 2)

    for n in range(t.size()[0]-1, 0, -1):
        aux[0] = alpha[1] * V[0, n] + alpha[1]*V[0, n-1]
        aux[S.size()[0]-3] = gamma[S.size()[0]-2] * V[S.size()[0]-1, n] + \
            gamma[S.size()[0]-2] * V[S.size()[0]-1, n-1]
        V[1:S.size()[0]-1, n-1] = torch.linalg.solve(M1,
                                                     M2@V[1:S.size()[0] - 1, n] + aux)
    return V
