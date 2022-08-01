"""
This file contains the code for generating option values and training an ANN using
the Heston model.

Implementation based on:

https://github.com/larsphilipp/AdvNum19_COS-FFT

References:

F. Fang and C. W. Oosterlee. “A Novel Pricing Method for European Options Based
on Fourier-Cosine Series Expansions (2009)
https://mpra.ub.uni-muenchen.de/9319/1/MPRA_paper_9319.pdf

A Fitt et al. “Progress in industrial mathematics at ECMI 2008” (2010)
https://link.springer.com/book/10.1007/978-3-642-12110-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Generating option prices using COS method with Heston model 

"""
Characteristic function of the Heston PDE
"""
def heston_char_fn(a, b, r, tau, v_0, v_bar, kappa, rho, gamma, k):
    u = k * np.pi / (b-a)
    D = np.sqrt((kappa - 1j * rho * gamma * u)**2 + (u**2 + 1j * u) * gamma**2)
    G = (kappa - 1j * rho * gamma * u - D) / (kappa - 1j * rho * gamma * u + D)
    p1 = 1j * u * r * tau + (v_0 / gamma**2) * (1 - np.exp(-D * tau)) / (2 - G * np.exp(-D * tau)) * (kappa - 1j * rho * gamma * u - D)
    p2 = (kappa * v_bar) / gamma**2 * (tau * (kappa - 1j * rho * gamma * u - D) - 2 * np.log((1 - G * np.exp(-D * tau)) / (1 - G)))
    return np.exp(p1) * np.exp(p2)

"""
Cosine series of g(y) = e^y
"""
def cos_series_exp(a, b, c, d, k):
    u = k * np.pi / (b-a)
    p1 = 1 / (1 + u**2)
    p2 = np.cos(u * (d-a)) * np.exp(d) - np.cos(u * (c-a)) * np.exp(c)
    p3 = u * np.sin(u * (d-a)) * np.exp(d) - u * np.sin(u * (c-a)) * np.exp(c)
    return p1 * (p2 + p3)

"""
Cosine series of g(y) = 1
"""
def cos_series_1(a, b, c, d, k):
    u = k[1:] * np.pi / (b-a)
    return np.concatenate([[d-c],  1/u * (np.sin(u * (d-a)) - np.sin(u * (c-a)))])

"""
Payoff series coefficients U used in getting call/put options using the COS method
"""
def U_call(a, b, k):
    return  2 / (b-a) * (cos_series_exp(a, b, 0, b, k) - cos_series_1(a, b, 0, b, k))

def U_put(a, b, k):
    return  2 / (b-a) * (cos_series_1(a, b, a, 0, k) - cos_series_exp(a, b, a, 0, k))

"""
Truncation range formula from Fitt et al. (2010)
"""
def truncation_range(L, r, tau, v_0, v_bar, kappa, rho, gamma):
    c1 = r * tau + (1 - np.exp(-kappa * tau)) * (v_bar - v_0) / (2 * kappa) - v_bar * tau / 2
    
    c2 = 1/(8 * kappa**3) * (gamma * tau * kappa * np.exp(-kappa * tau) \
            * (v_0 - v_bar) * (8 * kappa * rho - 4 * gamma) \
            + kappa * rho * gamma * (1 - np.exp(-kappa * tau)) * (16 * v_bar - 8 * v_0) \
            + 2 * v_bar * kappa * tau * (-4 * kappa * rho * gamma + gamma**2 + 4 * kappa**2) \
            + gamma**2 * ((v_bar - 2 * v_0) * np.exp(-2 * kappa * tau) \
            + v_bar * (6 * np.exp(-kappa * tau) - 7) + 2 * v_0) \
            + 8 * kappa**2 * (v_0 - v_bar) * (1 - np.exp(-kappa * tau)))

    a = c1 - L * np.sqrt(np.abs(c2))
    b = c1 + L * np.sqrt(np.abs(c2))

    return a, b

"""
Generate call/put option prices using the Heston model.
Call options can be generated directly or using the put-call parity.
"""
def heston_COS(a, b, N, S, K, r, tau, v_0, v_bar, kappa, rho, gamma, q=0):
    k = np.arange(N)
    x = np.log(S/K)
    characteristic_function = heston_char_fn(a, b, r, tau, v_0, v_bar, kappa, rho, gamma, k)
    add_integrated_term = np.exp(1j * k * np.pi * (x-a) / (b-a))
    U_k_put = U_put(a, b, k)
    U_k_call = U_call(a, b, k)
    F_k = np.real(characteristic_function * add_integrated_term)
    F_k[0] *= 0.5
    
    V_call =  K * np.sum(np.multiply(F_k, U_k_call)) * np.exp(-r * tau)
    V_put = K * np.sum(np.multiply(F_k, U_k_put)) * np.exp(-r * tau)
    V_pcp = V_put + S * np.exp(-q * tau) - K * np.exp(-r * tau)
    
    return V_call, V_put, V_pcp

"""
ANN
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Hidden4Net(nn.Module):
    def __init__(self, num_input, num_hid, num_out, batch_size=1024):
        super(Hidden4Net, self).__init__()
        self.batch_size = batch_size
        self.in_to_hid1 = nn.Linear(num_input, num_hid)
        self.hid1_to_hid2 = nn.Linear(num_hid,num_hid)
        self.hid2_to_hid3 = nn.Linear(num_hid,num_hid)
        self.hid3_to_hid4 = nn.Linear(num_hid,num_hid)
        self.hid4_to_out = nn.Linear(num_hid,num_out)

    def forward(self, inp):
        # Take into account batch size here
        hid1_sum = self.in_to_hid1(inp)
        self.hid1 = F.relu(hid1_sum)
        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = F.relu(hid2_sum)
        hid3_sum = self.hid2_to_hid3(self.hid2)
        self.hid3 = F.relu(hid3_sum)
        hid4_sum = self.hid3_to_hid4(self.hid3)
        self.hid4 = F.relu(hid4_sum)
        preOutput = self.hid4_to_out(self.hid4)
        output = F.relu(preOutput)
        return output

def train(net, train_loader, optimizer):
    net.train()
    train_running_loss = 0.0
    counter = 0
    for batch_id, (data,target) in enumerate(train_loader):
        counter += 1
        #print(data)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()    # zero the gradients
        output = net(data)       # apply network
        loss = F.mse_loss(output, target)
        #print(loss)
        train_running_loss += loss.item()
        loss.backward()          # compute gradients
        optimizer.step()         # update weights
    
    epoch_loss = train_running_loss / counter
    return epoch_loss

# validation
def validate(net, test_loader):
    net.eval()
    
    # we need two lists to keep track of class-wise accuracy
    #print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            counter += 1

            data = data.to(device)
            target = target.to(device)
            # forward pass
            outputs = net(data)
            # calculate the loss
            loss = F.mse_loss(outputs, target).item()
            valid_running_loss += loss
        
    epoch_loss = valid_running_loss / counter
    return epoch_loss

"""
Generate multiple option prices using Heston COS method
"""

n = 1000000

N = 1500
K = 1
L = 50

S = np.linspace(0.6, 1.4, n+1)[1:] * K
tau = np.linspace(0.1, 1.4, n+1)[1:]
r = np.linspace(0, 0.1, n+1)[1:]
rho = np.linspace(-0.95, 0, n+1)[1:]
kappa = np.linspace(0, 2, n+1)[1:]
v_bar = np.linspace(0, 0.5, n+1)[1:]
gamma = np.linspace(0, 0.5, n+1)[1:]
v_0 = np.linspace(0.05, 0.5, n+1)[1:]

a, b = truncation_range(L, r, tau, v_0, v_bar, kappa, rho, gamma)

V_np = np.zeros(n)

for i in tqdm(range(n)):
    V_call, _, V_pcp = heston_COS(a[i], b[i], N, S[i], K, r[i], tau[i], v_0[i], v_bar[i], kappa[i], rho[i], gamma[i])
    V_np[i] = V_pcp if S[i] < K else V_call
    V_np[i] = max(V_np[i], 0)
    #V_np[i] = min(V_np[i], 0.67)

# Reshape and concatenate input parameters
S = np.reshape(S, (-1, 1))
tau = np.reshape(tau, (-1, 1))
r = np.reshape(r, (-1, 1))
rho = np.reshape(rho, (-1, 1))
kappa = np.reshape(kappa, (-1, 1))
v_bar = np.reshape(v_bar, (-1, 1))
gamma = np.reshape(gamma, (-1, 1))
v_0 = np.reshape(v_0, (-1, 1))

inp_np = np.concatenate((S, tau, r, rho, kappa, v_bar, gamma, v_0), axis=1)

# input for ANN
inp = torch.from_numpy(inp_np).float()
V = torch.from_numpy(V_np).float()

"""
Train/validate ANN
"""

full_dataset = torch.utils.data.TensorDataset(inp, V.reshape(-1,1))
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size)

# Initialize Static variable
num_input = 8
num_hid = 400
num_out = 1
net = Hidden4Net(num_input, num_hid, num_out, batch_size=20).to(device)

for m in list(net.parameters()):
    if m.dim() == 2:
        # Initialize Weight Matrix
        torch.nn.init.xavier_uniform_(m, gain=nn.init.calculate_gain('relu'))
    else:
        # Initialize Bias
        torch.nn.init.zeros_(m)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# training loop
epoch = 1000
train_loss = []
valid_loss = []
print('Start training loop')
for i in tqdm(range(epoch)):
    train_epoch_loss = train(net,train_loader,optimizer)
    if i < 5:
        print(train_epoch_loss)
    valid_epoch_loss = validate(net, test_loader)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(
    train_loss, color='orange', linestyle='-', 
    label='train loss'
)
plt.plot(
    valid_loss, color='red', linestyle='-', 
    label='validation loss'
)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
 
print('TRAINING COMPLETE')