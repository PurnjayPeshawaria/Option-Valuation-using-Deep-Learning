# Training an ANN to predict implied volatility using the Black-Scholes model

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

def blackscholes(scaled_price, r, sigma, Tmt):
    s = sigma * np.sqrt(Tmt)
    d1 = (np.log(scaled_price) + (r + sigma**2/2)*(Tmt)) / s
    d2 = d1 - s
    optionValue = scaled_price * norm.cdf(d1) - np.exp(-r*Tmt) * norm.cdf(d2)
    return optionValue

class Hidden4NetTuned(nn.Module):
    def __init__(self, num_input, num_hid, num_out, batch_size=388):
        super(Hidden4NetTuned, self).__init__()
        self.batch_size = batch_size
        self.in_to_hid1 = nn.Linear(num_input, num_hid)
        self.hid1_to_hid2 = nn.Linear(num_hid,num_hid)
        self.hid2_to_hid3 = nn.Linear(num_hid,num_hid)
        self.hid3_to_hid4 = nn.Linear(num_hid,num_hid)
        self.hid4_to_out = nn.Linear(num_hid,num_out)

    def forward(self, input):
        # Take into account batch size here
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = F.elu(hid1_sum, 1)

        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = F.elu(hid2_sum, 1)

        hid3_sum = self.hid2_to_hid3(self.hid2)
        self.hid3 = F.elu(hid3_sum, 1)

        hid4_sum = self.hid3_to_hid4(self.hid3)
        self.hid4 = F.elu(hid4_sum, 1)

        preOutput = self.hid4_to_out(self.hid4)
        output = F.elu(preOutput, 1)
        return output

def train(net, train_loader, optimizer):
    net.train()
    train_running_loss = 0.0
    counter = 0
    for batch_id, (data,target) in enumerate(train_loader):
        counter += 1
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()    # zero the gradients
        output = net(data)       # apply network
        loss = F.mse_loss(output, target)
        train_running_loss += loss.item()
        loss.backward()          # compute gradients
        optimizer.step()         # update weights
    
    epoch_loss = train_running_loss / counter
    return epoch_loss

# validation
def validate(net, test_loader):
    net.eval()
    # we need two lists to keep track of class-wise accuracy
    print('Validation')
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

# Initialize Static variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input = 4
num_hid = 395
num_out = 1
net = Hidden4NetTuned(num_input, num_hid, num_out, batch_size=388).to(device)

for m in list(net.parameters()):
    if m.dim() == 2:
        # Initialize Weight Matrix
        torch.nn.init.xavier_uniform_(m, gain=nn.init.calculate_gain('relu'))
    else:
        # Initialize Bias
        torch.nn.init.zeros_(m)

optimizer = torch.optim.Adam(net.parameters(),lr=1.60E-04)

# testing the wide range here
n = 1000000

# risk free rate 
r = torch.FloatTensor(n).uniform_(0.0, 0.1)

# volatility
sigma = torch.FloatTensor(n).uniform_(0.05, 1.0)

# Time till expiration
tau = torch.FloatTensor(n).uniform_(0.05, 1.0)

# scaled stock price
scaled_price = torch.FloatTensor(n).uniform_(0.5, 1.4)

# calculate Black-Scholes option value
blackScholesTarget = blackscholes(scaled_price, r, sigma, tau).float()

# apply gradient squash to Black-Scholes option value
intrinsic_value = scaled_price - np.exp(-r*tau)
intrinsic_value[intrinsic_value < 0] = 0
squashedBlackScholesTarget = blackScholesTarget - intrinsic_value

# remove all data points for which the squashedBlackScholesTarget is less than 10^-7
#lower_bound = torch.full((n,1), 0.0000001)
reducedBlackScholesTarget = squashedBlackScholesTarget[squashedBlackScholesTarget > 0.0000001]
reducedR = r[squashedBlackScholesTarget > 0.0000001]
reducedSigma = sigma[squashedBlackScholesTarget > 0.0000001]
reducedTau = tau[squashedBlackScholesTarget > 0.0000001]
reducedScaled_price = scaled_price[squashedBlackScholesTarget > 0.0000001]

# log transform the option values
scaled_time_value = np.log(reducedBlackScholesTarget)

# input for ANN
input = torch.stack((reducedScaled_price, reducedTau, reducedR, scaled_time_value), -1)

full_dataset = torch.utils.data.TensorDataset(input, reducedSigma.reshape(-1,1))
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size)

# # # training loop
epoch = 200
train_loss = []
valid_loss = []
print('Start training loop')
for i in tqdm(range(epoch)):
    train_epoch_loss = train(net,train_loader,optimizer)
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



# root finding problem for implied volatility using Black-Scholes model

def blackscholesNotScaled(S, K, r, sigma, Tmt):
    s = sigma * np.sqrt(Tmt)
    d1 = (np.log(S / K) + (r + sigma**2/2)*(Tmt)) / s
    d2 = d1 - s
    optionValue = S * norm.cdf(d1) - K * np.exp(-r*Tmt) * norm.cdf(d2)
    return optionValue

def g(sigma, Vmkt):
  r = 0
  Tmt = 0.5
  K = 1.0
  S = 1.0
  return blackscholesNotScaled(S, K, r, sigma, Tmt) - Vmkt



# Brent's method
# Based on code from https://nickcdryan.com/2017/09/13/root-finding-algorithms-in-python-line-search-bisection-secant-newton-raphson-boydens-inverse-quadratic-interpolation-brents/

def brents_method(Vmkt):
    tolerance = 1e-4

  # not sure how to choose our initial guesses
    sigma0 = 0.01
    gsigma0 = g(sigma0, Vmkt)

    sigma1 = 0.99
    
    gsigma1 = g(sigma1, Vmkt)
    assert (gsigma0 * gsigma1) <= 0, "Root not bracketed" 
 
    if abs(gsigma0) < abs(gsigma1):
        sigma0, sigma1 = sigma1, sigma0
        gsigma0, gsigma1 = gsigma1, gsigma0
    
    sigma2, gsigma2 = sigma0, gsigma0
    d = 0.7
    
    mflag = True
    steps_taken = 0
    
    while steps_taken < 50 and abs(sigma1-sigma0) > tolerance:
        gsigma0 = g(sigma0, Vmkt)
        gsigma1 = g(sigma1, Vmkt)
        gsigma2 = g(sigma2, Vmkt)
    
        if gsigma0 != gsigma2 and gsigma1 != gsigma2:
            L0 = (sigma0 * gsigma1 * gsigma2) / ((gsigma0 - gsigma1) * (gsigma0 - gsigma2))
            L1 = (sigma1 * gsigma0 * gsigma2) / ((gsigma1 - gsigma0) * (gsigma1 - gsigma2))
            L2 = (sigma2 * gsigma1 * gsigma0) / ((gsigma2 - gsigma0) * (gsigma2 - gsigma1))
            new = L0 + L1 + L2
        else:
            new = sigma1 - ( (gsigma1 * (sigma1 - sigma0)) / (gsigma1 - gsigma0) )
    
        if ((new < ((3 * sigma0 + sigma1) / 4) or new > sigma1) or
            (mflag == True and (abs(new - sigma1)) >= (abs(sigma1 - sigma2) / 2)) or
            (mflag == False and (abs(new - sigma1)) >= (abs(sigma2 - d) / 2)) or
            (mflag == True and (abs(sigma1 - sigma2)) < tolerance) or
            (mflag == False and (abs(sigma2 - d)) < tolerance)):
            new = (sigma0 + sigma1) / 2
            mflag = True
    
        else:
            mflag = False
    
        gnew = g(new, Vmkt)
        d, sigma2 = sigma2, sigma1
 
        if (gsigma0 * gnew) < 0:
            sigma1 = new
        else:
            sigma0 = new
    
        if abs(gsigma0) < abs(gsigma1):
            sigma0, sigma1 = sigma1, sigma0
    
        steps_taken += 1

        return sigma1
    
    
# generating data for numerical methods for implied volatility

n = 20000
sigma = (torch.linspace(0.011, 0.99, n)).float()
r = 0
Tmt = 0.5
K = 1.0
S = 1.0
Vmkt = blackscholesNotScaled(S, K, r, sigma, Tmt)


# predicting implied volatility using Brent's method

import time

brents_diff = []

start_time = time.time()

for i in range(n):
 actual_sigma = sigma[i]
 predicted_sigma = brents_method(Vmkt[i])
 brents_diff.append(actual_sigma - predicted_sigma)

end_time = time.time()

brents_mse = sum(list(map(lambda x:pow(x,2), brents_diff))) / n
print("MSE = " + str(brents_mse))
print("CPU time: " + str(end_time - start_time))




# rework the data into the form required for the ANN 

import torch
import numpy as np
from scipy.stats import norm

def blackscholes(scaled_price, r, sigma, Tmt):
    s = sigma * np.sqrt(Tmt)
    d1 = (np.log(scaled_price) + (r + sigma**2/2)*(Tmt)) / s
    d2 = d1 - s
    optionValue = scaled_price * norm.cdf(d1) - np.exp(-r*Tmt) * norm.cdf(d2)
    return optionValue

n = 20000

# risk free rate 
r = (torch.linspace(0, 0, n)).float()

# Time till expiration
tau = (torch.linspace(0.5, 0.5, n)).float()

# scaled stock price
scaled_price = (torch.linspace(1, 1, n)).float()

# calculate Black-Scholes option value
blackScholesTarget = torch.from_numpy(Vmkt).float()

# apply gradient squash to Black-Scholes option value
intrinsic_value = scaled_price - np.exp(-r*tau)
intrinsic_value[intrinsic_value < 0] = 0
squashedBlackScholesTarget = blackScholesTarget - intrinsic_value

# remove all data points for which the squashedBlackScholesTarget is less than 10^-7
reducedBlackScholesTarget = squashedBlackScholesTarget[squashedBlackScholesTarget > 0.0000001]
reducedR = r[squashedBlackScholesTarget > 0.0000001]
reducedSigma = sigma[squashedBlackScholesTarget > 0.0000001]
reducedTau = tau[squashedBlackScholesTarget > 0.0000001]
reducedScaled_price = scaled_price[squashedBlackScholesTarget > 0.0000001]

# log transform the option values
scaled_time_value = np.log(reducedBlackScholesTarget)

# input for ANN
input = torch.stack((reducedScaled_price, reducedTau, reducedR, scaled_time_value), -1)

dataset = torch.utils.data.TensorDataset(input, reducedSigma.reshape(-1,1))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=n)



# implied volatility using the ANN

start_time = time.process_time()

loss = validate(net, data_loader)

end_time = time.process_time()

print("MSE = " + str(loss))
print("CPU time: " + str(end_time - start_time))