import torch
import time
from ANN import Hidden4NetTuned, network_process, blackscholes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create Network, Optimizer and Dataset
n = 1000
scaled_price = torch.FloatTensor(n).uniform_(0.4, 1.6)
tau = torch.FloatTensor(n).uniform_(0.2, 1.1)
r = torch.FloatTensor(n).uniform_(0.02, 0.1)
sigma = torch.FloatTensor(n).uniform_(0.01, 1.0)
input = torch.stack((scaled_price, tau, sigma, r), -1)
blackScholesTarget = blackscholes(scaled_price, r, sigma, tau).float()
full_dataset = torch.utils.data.TensorDataset(
    input, blackScholesTarget.reshape(-1, 1))

# Test random values for the parameters
start_time = time.process_time()
net = Hidden4NetTuned(num_input=4, num_hid=321, num_out=1).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1.60E-04)
full_network = network_process(net, optimizer, full_dataset, epoch=200)
full_network.start_train()
full_network.plot_loss()
end_time = time.process_time()
print("CPU time for ANN method is : " + str(end_time - start_time))

print(f"MSE {full_network.valid_loss[-1]}")
