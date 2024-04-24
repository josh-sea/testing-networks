import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import copy

# Load and transform the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Custom Activation Layer with correct input dimension
class CustomActivationLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomActivationLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation_net = ActivationNetwork(output_dim)

    def forward(self, x):
        x = self.linear(x)
        return self.activation_net(x)

# Define a custom activation network to be used as an activation function
class ActivationNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ActivationNetwork, self).__init__()
        self.layer1 = nn.Linear(in_features=input_dim, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return x

# Adjust the main network architecture
class SimpleMainNetwork(nn.Module):
    def __init__(self):
        super(SimpleMainNetwork, self).__init__()
        self.layer1 = CustomActivationLayer(28 * 28, 10)  # 28 * 28 = 784 (input image size)
        self.layer2 = CustomActivationLayer(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x
    
    # Add this method to the SimpleMainNetwork class
    def update_activation_network(self):
        self.layer1.activation_net = copy.deepcopy(self)
        self.layer2.activation_net = copy.deepcopy(self)

# Training function that continues until the network has converged
def train_until_convergence(main_net, data_loader, threshold=0.01, patience=5):
    optimizer = torch.optim.Adam(main_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    previous_losses = []
    epoch = 0

    while True:
        total_loss = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(main_net.device), labels.float().to(main_net.device)
            optimizer.zero_grad()
            outputs = main_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}, Average Loss: {average_loss}')

        if has_converged(average_loss, previous_losses, threshold, patience):
            # Update the ActivationNetwork with the trained SimpleMainNetwork
            main_net.update_activation_network()
            previous_losses = []
            epoch = 0

        previous_losses.append(average_loss)
        epoch += 1

        if len(previous_losses) >= 2 * patience:
            break

    return main_net


# Function to check if the network has converged
def has_converged(current_loss, previous_losses, threshold, patience):
    if current_loss < threshold:
        return True
    if len(previous_losses) < patience:
        return False
    return all(previous_loss - current_loss < 1e-4 for previous_loss in previous_losses[-patience:])

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model initialization and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMainNetwork()
model.to(device)
model.device = device  # Store the device in the model for easy access

num_epochs = 10  # Example: Define the number of epochs for non-convergent criteria training
trained_model = train_until_convergence(model, train_loader, threshold=0.01, patience=5)

# Save the trained model state
torch.save(trained_model.state_dict(), 'trained_main_network.pth')