import matplotlib.pyplot as plt
import random
import time
from black_scholes_unscaled import blackscholes_unscaled
from bspdeimp import bspdeimp
from bspdeexp import bspdeexp
from bspdeCN import bspdeCN
import numpy as np
import torch

random.seed(23)

K = 1
r = random.uniform(0.02, 0.1)
sigma = random.uniform(0.01, 1.0)
S = np.linspace(0, 1.6*K, 1000)
Tmt = 1.1

V = blackscholes_unscaled(S[1:-1], K, r, sigma, Tmt)
V = np.concatenate((np.array([0]), V, np.array([S[-1] - K*np.exp(-r*(Tmt))])))
V_tor = torch.from_numpy(V)

S = torch.linspace(0, 1.6*K, 1000, dtype=torch.float64)
t = torch.linspace(0, 1.1, 1000, dtype=torch.float64)


start_time = time.process_time()
V_imp = bspdeimp(K, t, r, sigma, S)
Err = V_tor[0:-1] - V_imp[0:-1, 0]
MSE = (1/S.size()[0])*torch.linalg.vector_norm(Err, ord=2)**2
end_time = time.process_time()

print("CPU time for implicit method is : " + str(end_time - start_time))
print(f"The MSE for implicit method is {MSE}")

X, Y = torch.meshgrid(S, t, indexing="ij")

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')
ax.plot_surface(X.detach().numpy(), Y.detach().numpy(),
                (V_imp.T).detach().numpy(), cmap='viridis')
ax.set_title('Valuation of European call using Implicit Euler')
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')
plt.savefig("implicit_FDM_option_value.jpg", dpi=300)
plt.show()

plt.plot(S[0:-1], Err)
plt.xlabel("S")
plt.title("Error for implict method at t = 0")
plt.savefig("implicit_error.png", dpi=300)
plt.show()

V_exp = bspdeexp(K, t, r, sigma, S)
Err = V_tor - V_exp[:, 0]
MSE = (1/S.size()[0])*torch.linalg.vector_norm(Err, ord=2)**2
print(f"The MSE for explicit method is {MSE}")

start_time = time.process_time()
V_CN = bspdeCN(K, t, r, sigma, S)
Err = V_tor[0:-1] - V_CN[0:-1, 0]
MSE = (1/S.size()[0])*torch.linalg.vector_norm(Err, ord=2)**2
end_time = time.process_time()
print("CPU time for CN method is : " + str(end_time - start_time))
print(f"The MSE for Crank-Nicolson method is {MSE}")
X, Y = torch.meshgrid(S, t, indexing="ij")
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')
ax.plot_surface(X.detach().numpy(), Y.detach().numpy(),
                (V_CN.T).detach().numpy(), cmap='viridis')
ax.set_title('Valuation of European call using Crank-Nicolson')
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')
plt.savefig("CN_FDM_option_value.jpg", dpi=300)
plt.show()
plt.plot(S[0:-1], Err)
plt.xlabel("S")
plt.title("Error for CN method at t = 0")
plt.savefig("CN_error.png", dpi=300)
plt.show()
