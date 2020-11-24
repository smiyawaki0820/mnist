#####################################################
### --- Define a simple network in this block --- ###
#####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

# Build a simple network
class SimpleNetwork(nn.Module):
  def __init__(self, input_dim=1, num_class=10):
    super(SimpleNetwork, self).__init__()

    # Fully connected layers
    self.fc1 = nn.Linear(784, 512)
    #nn.init.kaiming_uniform_(self.fc1.weight)
    self.fc2 = nn.Linear(512, num_class)
    #nn.init.kaiming_uniform_(self.fc2.weight)
    
    # Activation func.
    #self.relu = nn.ReLU()

    #self.dropout1 = nn.Dropout(0.2)
    
  def forward(self, x):
    #x = self.relu(self.fc1(x))                   print("Trainig Data is limited, only the first "+str(self.data.size(0))+" samples are used.")
    #x = self.relu(self.fc2(x))     
    x = F.relu(self.fc1(x))  
    #x = self.dropout1(x)
   # x = F.relu(self.fc2(x))     

    return x