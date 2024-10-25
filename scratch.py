import torch
import numpy as np
from fancy_einsum import einsum
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def generate_synthetic_data(num_samples, n, sparsity, device='cpu'):
    """
    Generate synthetic data with specified sparsity and importance.
    
    Args:
    num_samples (int): Number of samples to generate
    n (int): Number of features for each sample
    sparsity (float): Probability of a feature being zero (0 <= sparsity < 1)
    device (str): Device to place the tensor on ('cpu' or 'cuda')
    
    Returns:
    torch.Tensor: Synthetic data tensor of shape (num_samples, n)
    """
    # Generate uniform random values between 0 and 1
    x = torch.rand(num_samples, n, device=device)
    
    # Generate mask for sparse features
    mask = torch.rand(num_samples, n, device=device) < sparsity
    x[mask] = 0
    
    return x

#Note: bias is by default in torch linear
class SimpleLinear(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()
        # self.W = torch.nn.Linear(n, m)
        self.W = torch.nn.Parameter(torch.rand(n, m))
        self.b = torch.nn.Parameter(torch.rand(n))

    def forward(self, x):
        # h=Wx
        hidden = einsum('num_samples features, features d_hidden -> num_samples d_hidden', x, self.W)
        # x' = W^T@h+b
        invert = einsum('num_samples d_hidden, features d_hidden -> num_samples features', hidden, self.W) + self.b
        return invert

class SimpleNonLinear(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()
        # self.W = torch.nn.Linear(n, m)
        self.W = torch.nn.Parameter(torch.rand(n, m))
        self.b = torch.nn.Parameter(torch.rand(n))
        self.hidden = None
        self.invert = None

    def forward(self, x):
        # h = Wx
        self.hidden = einsum('num_samples features, features d_hidden -> num_samples d_hidden', x, self.W)
        # x' = ReLU(W^T@h+b)
        self.invert = torch.relu(einsum('num_samples d_hidden, features d_hidden -> num_samples features', self.hidden, self.W) + self.b)
        return self.invert

def weighted_mean_loss(output, target, importance):
    """
    Args:
    output (num_samples, num_features): 
    target (num_samples, num_features):  
    importance (num_features):   
    """
    # breakpoint()

    squared_diff = (output - target) ** 2
    weighted_mean_loss = einsum('num_features, num_samples num_features -> num_samples', importance, squared_diff).mean()
    return weighted_mean_loss


num_samples = 10
#sparsity = 0.999
sparsity = 0
# sparsity = 0.99
n = 20
m = 5
importances = torch.tensor([0.7**i for i in range(n)])

synthetic_data = generate_synthetic_data(num_samples, n, sparsity)

linear = SimpleLinear(n, m)
linear(synthetic_data)

non_linear = SimpleNonLinear(n, m)
non_linear(synthetic_data)

model = non_linear #TODO: specify model here


# TODO: these details arent specified in the paper

num_epochs = 1000
lr = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    outputs = model(synthetic_data)
    # breakpoint()

    loss = weighted_mean_loss(outputs, synthetic_data, importances)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

feature_importance = torch.norm(model.W, dim=1)
# breakpoint()
# assert feature_importance.shape == n

sorted_indices = torch.argsort(feature_importance, descending=True)
W_sorted = model.W[sorted_indices, :]

wwt = (W_sorted @ W_sorted.T).detach()
print(f'Top 5: {torch.diag(wwt)}')
wwt = wwt.numpy()

# Creating a normalized color scheme between -1 and 1
norm = Normalize(vmin=-1, vmax=1)
plt.figure(figsize=(6, 6))
plt.imshow(wwt, cmap='bwr', interpolation='nearest')
plt.colorbar(label='Weight/Bias Element Values')

# Adding title and labels
plt.title(r'$W^TW$')
plt.xlabel(r'$W^T$')
plt.ylabel(r'$W$')

plt.savefig(f"WWT_S={sparsity}.png")
plt.close()