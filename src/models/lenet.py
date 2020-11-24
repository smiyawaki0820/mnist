############################################
### --- Define a LeNet in this block --- ###
############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define LeNet
class LeNet(nn.Module):
  def __init__(self, input_dim=1, num_class=10):
    super(LeNet, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(input_dim, 20,  kernel_size=5, stride=1, padding=0)
    self.conv2 = nn.Conv2d(    20,    50,  kernel_size=5, stride=1, padding=0)

    # Fully connected layers
    self.fc1 = nn.Linear(800, 500)
    self.fc2 = nn.Linear(500, num_class)
    
    # Activation func.
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))        # Conv.-> ReLU
    x = F.max_pool2d(x, kernel_size=2, stride=2)  # Pooling with 2x2 window
    x = self.relu(self.conv2(x))        # Conv.-> ReLU
    x = F.max_pool2d(x, kernel_size=2, stride=2)  # Pooling with 2x2 window

    b,c,h,w = x.size()                  # batch, channels, height, width
    x = x.view(b, -1)                   # flatten the tensor x

    x = self.relu(self.fc1(x))          # fc-> ReLU
    x = self.fc2(x)                     # fc
    return x