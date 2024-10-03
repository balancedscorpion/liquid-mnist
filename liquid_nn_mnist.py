import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchdiffeq

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 5

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Liquid Neural Network Components
class LiquidLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidLayer, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, hidden_size)
        
        # Parameters for the liquid dynamics
        self.alpha = nn.Parameter(torch.randn(hidden_size))
        self.beta = nn.Parameter(torch.randn(hidden_size))
        self.gamma = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, t, h, input_t):
        # Liquid dynamics: dh/dt = -alpha * h + beta * activation(fc(input) + gamma * h)
        activation = torch.tanh(self.fc(input_t) + self.gamma * h)
        dhdt = -self.alpha * h + self.beta * activation
        return dhdt

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps=1.0):
        super(LiquidNeuralNetwork, self).__init__()
        self.liquid = LiquidLayer(input_size, hidden_size)
        self.time_steps = time_steps
        self.readout = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        h0 = torch.zeros(batch_size, self.liquid.hidden_size).to(x.device)
        
        # Define the ODE function
        def ode_func(t, h):
            # Different approaches to interpolate t
            interval = self.time_steps / seq_len
            current_step = min(int(t // interval), seq_len -1)
            input_t = x[:, current_step, :]
            return self.liquid(t, h, input_t)
        
        # Solve ODE
        t = torch.linspace(0, self.time_steps, steps=seq_len)
        h_final = torchdiffeq.odeint(ode_func, h0, t, method='dopri5')[-1]
        
        # Readout
        out = self.readout(h_final)
        return out

# Model, Loss, Optimizer
input_size = 28  # Each row of the MNIST image (28 pixels)
hidden_size = 128
output_size = 10  # Number of classes
time_steps = 1.0  # Total time for ODE integration

model = LiquidNeuralNetwork(input_size, hidden_size, output_size, time_steps).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Preprocess data
        data = data.to(device)  # Shape: (batch, 1, 28, 28)
        targets = targets.to(device)
        data = data.squeeze(1)  # Shape: (batch, 28, 28)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        data = data.squeeze(1)  # Shape: (batch, 28, 28)
        
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


