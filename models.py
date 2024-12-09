import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_length=2000, output_dim=2):
        """
        Initialize the CNN.

        Parameters:
            input_length (int): Length of the input sequences (default: 2000).
            output_dim (int): Number of output dimensions (default: 2 for start and end indices).
        """
        super(CNN, self).__init__()
        self.input_length = input_length

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)  # Input: [batch_size, 1, input_length]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)

        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Halves the sequence length each time

        # Fully connected layers
        fc_input_dim = input_length // (2 ** 4) * 256  # Reduced length after 4 pooling layers * number of filters in conv4
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, input_length].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        # Add channel dimension to match Conv1D input requirement
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, input_length]

        # Convolutional and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x



class DeepCpGModel(nn.Module):
    """
    A deeper neural network for predicting CpG island start and end indices with high-dimensional input.
    """
    def __init__(self, input_dim=8000, output_dim=2):
        """
        Initialize the network.

        Parameters:
            input_dim (int): Dimension of the input features (default: 8000).
            output_dim (int): Dimension of the output predictions (default: 2 for start and end indices).
        """
        super(DeepCpGModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define layers
        self.fc1 = nn.Linear(input_dim, 1024)  # Reduce dimensionality
        self.fc2 = nn.Linear(1024, 512)       # Further reduce dimensionality
        self.fc3 = nn.Linear(512, 256)        # Additional fully connected layer
        self.fc4 = nn.Linear(256, 128)        # Additional fully connected layer
        self.fc5 = nn.Linear(128, output_dim) # Output layer

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# Example usage
