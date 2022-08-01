import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Find the best learning rate
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


def blackscholes(scaled_price, r, sigma, Tmt):
    s = sigma * np.sqrt(Tmt)
    d1 = (np.log(scaled_price) + (r + sigma**2/2)*(Tmt)) / s
    d2 = d1 - s
    optionValue = scaled_price * norm.cdf(d1) - np.exp(-r*Tmt) * norm.cdf(d2)
    return optionValue

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
num_input = 4
num_hid = 400
num_out = 1

class network_process():
    def __init__(self, network, optimizer, dataset, epoch=10):
        self.network = network
        self.epoch = epoch
        self.optimizer = optimizer
        self.dataset = dataset
        self.train_loss = []
        self.valid_loss = []
        self.train_loader = None
        self.test_loader = None
        self.initalise_weights()
        self.initialise_dataset()

    def initalise_weights(self):
        for m in list(self.network.parameters()):
            # Initialize Weight Matrix
            if m.dim() == 2:
                torch.nn.init.xavier_uniform_(m, gain=nn.init.calculate_gain('relu'))
            else:
            # Initialize Bias
                torch.nn.init.zeros_(m)

    def initialise_dataset(self):
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size)

    def train_network(self):
        self.network.train()
        train_running_loss = 0.0
        counter = 0
        for batch_id, (data,target) in enumerate(self.train_loader):
            counter += 1
            data = data.to(device)
            target = target.to(device)
            self.optimizer.zero_grad()    # zero the gradients
            output = self.network(data)       # apply network
            loss = F.mse_loss(output, target)
            train_running_loss += loss.item()
            loss.backward()          # compute gradients
            self.optimizer.step()         # update weights
        
        epoch_loss = train_running_loss / counter
        return epoch_loss

    # validation
    def validate_network(self):
        self.network.eval()
        # we need two lists to keep track of class-wise accuracy
        valid_running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                counter += 1
                data = data.to(device)
                target = target.to(device)
                # forward pass
                outputs = self.network(data)
                # calculate the loss
                loss = F.mse_loss(outputs, target).item()
                valid_running_loss += loss
            
        epoch_loss = valid_running_loss / counter
        return epoch_loss
    
    def start_train(self):
        print('Start training loop')
        for i in range(self.epoch):
            print(f"[INFO]: Epoch {i+1} of {self.epoch}")
            train_epoch_loss = self.train_network()
            valid_epoch_loss = self.validate_network()
            self.train_loss.append(train_epoch_loss)
            self.valid_loss.append(valid_epoch_loss)
            print(f"Training loss: {train_epoch_loss:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}")
        print('Training Complete')

    def plot_loss(self):
        plt.figure(figsize=(10, 7))
        plt.plot(
            self.train_loss, color='orange', linestyle='-', 
            label='train loss'
        )
        plt.plot(
            self.valid_loss, color='red', linestyle='-', 
            label='validation loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

# Create Network, Optimizer and Dataset
n = 1000000
# risk free rate
num_input = 4
num_hid = 400
num_out = 1
scaled_price = torch.FloatTensor(n).uniform_(0.4, 1.6)
tau = torch.FloatTensor(n).uniform_(0.2, 1.1)
r = torch.FloatTensor(n).uniform_(0.02, 0.1)
sigma = torch.FloatTensor(n).uniform_(0.01, 1.0)
input = torch.stack((scaled_price, tau, sigma, r), -1)
blackScholesTarget = blackscholes(scaled_price, r, sigma, tau).float()
full_dataset = torch.utils.data.TensorDataset(input, blackScholesTarget.reshape(-1, 1))
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size)

# Test random values for the parameters
model = Hidden4NetTuned(num_input, num_hid, num_out)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
lr_finder.range_test(train_loader, end_lr=1e-3, num_iter=100, step_mode="linear")
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state