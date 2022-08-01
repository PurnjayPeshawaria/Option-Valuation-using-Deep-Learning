from scipy.sparse import spdiags
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the code is based on a book by Paolo Brandimarte (Numerical Methods in finance and economics)
# the original code was in MATLAB and we coverted the code to Python
# some assistance has also been taken from Professor Josef Dick's notes in Mathematical Computing for finance
# again the code was in MATLAB and we had to convert eveything to Python


def bspdeimp(K, t, r, sigma, S):
    dt = t[1] - t[0]
    dS = S[1] - S[0]
    V = torch.zeros((S.size()[0], t.size()[0]), dtype=torch.float64)
    V[0, :] = 0
    V[-1, :] = S[-1] - K * torch.exp(-r*(t[-1] - t))
    V[1:-1, -1] = torch.maximum(S[1:-1] - K,
                                torch.zeros((S.size()[0]-2), dtype=torch.float64))
    c1 = sigma**2 * S[1:S.size()[0]-1]**2 * dt / (2 * dS**2)
    c2 = r * S[1:S.size()[0]-1] * dt / (2 * dS)
    alpha = -c1 + c2
    beta = 1 + r*dt + 2*c1
    gamma = -c1 - c2
    k = torch.cat(
        (alpha[1:S.size()[0]-2], torch.tensor(([0]), dtype=torch.float64)))
    l = torch.cat(
        (torch.tensor(([0]), dtype=torch.float64), gamma[0:S.size()[0]-3]))
    data = torch.vstack((k, beta, l))
    diags = torch.tensor([-1, 0, 1], dtype=torch.float64)
    A = spdiags(data, diags, S.size()[0]-2, S.size()[0]-2).toarray()
    A_torch = torch.from_numpy(A)
    for n in range(t.size()[0]-2, -1, -1):
        b = V[1:S.size()[0]-1, n+1] - torch.cat((torch.tensor([alpha[0] * V[0, n]], dtype=torch.float64),
                                                 torch.zeros(
                                                     (S.size()[0]-4), dtype=torch.float64),
                                                 torch.tensor([gamma[S.size()[0]-3]*V[S.size()[0] - 1, n]], dtype=torch.float64)))
        V[1:S.size()[0]-1, n] = torch.linalg.solve(A_torch, b)
    return V
