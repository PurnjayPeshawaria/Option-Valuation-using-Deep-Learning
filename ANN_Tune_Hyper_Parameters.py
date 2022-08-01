import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.utils.data import DataLoader
import warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')

# Tune Hyperparameters of our network
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Hidden4Net(nn.Module):
    def __init__(self, num_input, num_output, neurons, dropout_rate,activation,batch_normalization):
        super(Hidden4Net, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(num_input, neurons),
            nn.BatchNorm1d(neurons) if batch_normalization else nn.Identity(),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(neurons, neurons),
            nn.BatchNorm1d(neurons) if batch_normalization else nn.Identity(),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(neurons, neurons),
            nn.BatchNorm1d(neurons) if batch_normalization else nn.Identity(),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(neurons, neurons),
            nn.BatchNorm1d(neurons) if batch_normalization else nn.Identity(),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(neurons, num_output),
            activation,
        )

    def forward(self, input):
        output = self.module(input)
        return output

def blackscholes(scaled_price, r, sigma, Tmt):
    s = sigma * np.sqrt(Tmt)
    d1 = (np.log(scaled_price) + (r + sigma**2/2)*(Tmt)) / s
    d2 = d1 - s
    optionValue = scaled_price * norm.cdf(d1) - np.exp(-r*Tmt) * norm.cdf(d2)
    return optionValue

# Train loss function
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

# # Validation loss function
def validate(net, test_loader):
    net.eval()
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            counter += 1
            data = data.to(device)
            target = target.to(device)
            outputs = net(data)
            loss = F.mse_loss(outputs, target).item()
            valid_running_loss += loss
    epoch_loss = valid_running_loss / counter
    return epoch_loss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
MAX_EVALS = 1000

# testing the wide range here
n = 1000
r = (torch.linspace(0.02, 0.1, n)).float() 
sigma = (torch.linspace(0.01, 1.0, n)).float()
tau = (torch.linspace(0.2, 1.1, n)).float()
scaled_price = (torch.linspace(0.4, 1.6, n)).float()
input = torch.stack((scaled_price, tau, sigma, r), -1)
blackScholesTarget = blackscholes(scaled_price, r, sigma, tau).float()
full_dataset = torch.utils.data.TensorDataset(input, blackScholesTarget.reshape(-1, 1))
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

param_grid = {
    'Activation': [nn.ReLU(True),nn.Tanh(),nn.Sigmoid(),nn.ELU()],
    'Dropout_rate': [0.0,0.2],
    'Neurons': list(range(200, 400)),
    'Initialization':['uniform', 'glorot_uniform', 'he_uniform'],
    'Batch_normalization':[True,False],
    'Optimizer': ['SGD', 'RMSprop', 'Adam'],
    'batch_size': list(range(256, 3001))
}

# record the best hyperparams
best_eval_loss = -1
best_hyperparams = {}
epoch=20

for i in range(MAX_EVALS):
    print(f'Hyperparameters {i+1}')
    random.seed(i)  # Set a random seed, and set a different seed for each search. If the seed is fixed, the hyperparameters selected each time are the same 
    hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items() }

    activation = hyperparameters['Activation']
    Dropout_rate = hyperparameters['Dropout_rate']
    Neurons = hyperparameters['Neurons']
    Initialization = hyperparameters['Initialization']
    Batch_normalization = hyperparameters['Batch_normalization']
    Optimizer = hyperparameters['Optimizer']
    batch_size = hyperparameters['batch_size']


    model = Hidden4Net(num_input=4, num_output=400, neurons=Neurons, activation=activation,dropout_rate=Dropout_rate,batch_normalization=Batch_normalization).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if Initialization == 'uniform':
                torch.nn.init.uniform_(m.weight, a=0, b=1)
            elif Initialization == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            else:
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    if Optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=1.60E-04)
    elif Optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(),lr=1.60E-04)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.60E-04)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for i in range(epoch):
        train_epoch_loss = train(model, train_loader, optimizer)
        valid_epoch_loss = validate(model,test_loader=test_loader)

    
    print(f"Last Training loss: {train_epoch_loss:.3f}")
    print(f"Last Validation loss: {valid_epoch_loss:.3f}")
    print()
    if best_eval_loss == -1:
        best_eval_loss = valid_epoch_loss
        best_hyperparams = hyperparameters
    if valid_epoch_loss < best_eval_loss:
        best_hyperparams = hyperparameters
        best_eval_loss = valid_epoch_loss

print('Best evaluation loss is ', best_eval_loss)
print('Best Hyperparameters is' , best_hyperparams)