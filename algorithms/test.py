import torch
from torch import nn
import torch.optim as optim
a = 2.4785694
b = 7.3256989
error = 0.1
n = 100 

# Data
x = torch.randn(n, 1)

t = a * x + b + (torch.randn(n, 1) * error)

model = nn.Linear(1, 1)
optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

# Run training
niter = 10
for _ in range(0, niter):
	optimizer.zero_grad()
	predictions = model(x)
	loss = loss_fn(predictions, t)
	loss.backward()
	optimizer.step()

	print("-" * 10)
	print("learned a = {}".format(list(model.parameters())[0].data[0, 0]))
	print("learned b = {}".format(list(model.parameters())[1].data[0]))