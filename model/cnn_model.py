import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
We will be using a Convolutional Neural Network to extract spatial patterns from the board to predict an evaluation score.
We will use a CNN as they are designed for spatial pattern recognition. It will take an 8x8x12 chess board tensor as an input.

Our network has two main sections: 
    1. Convolutional layers: used for pattern detection. 
        - Layer 1 will detect simple local patterns (for ex two pawns side by side)
        - Layer 2 will combine simple patterns into tactics (for ex pawn chains)
        - Layer 3 combines tactics into strategy (for ex strong control of center of board or strong king safety and ect)
    2. Fully Connected layers: used for decisino making 
        - This section will take all the decteded patterns and run them through a neural network and it will produce and evaluation score.
"""


# Define the neural network archietecture 
class ChessEvalCNN(nn.Module):
    def __init__(self):
        """ Constructor used to define all the layres of our network. This is where we specify 
            the architecture of our network and in forward() we will specify how the data flows through it.
        """

        # First call the parent class constructor which initializes all the nn.MOdule machinery
        super().__init__()

        # Convolutional Layer 1: Basic Pattern Detector 
        self.conv1 = nn.Conv2d(in_channels=12, # Input: 12 peice type layers
                               out_channels=32, # Output: 32 different pattern detectors (filters/kernels)
                               kernel_size=3, # Each filter looks at 3x3 square patches (the ideal size) 
                               padding=1 # This basically adds zeros around the edge so the 8x8 output doesn't shrink
                               )
        
        # conv1 will produce 32 feauture maps and their values might be all over the place which could make the training stage slow
        # so we will fix this by using BatchNOrm which normalizes each of the 32 channels.
        self.bn1 = nn.BatchNorm2d(32)

        # Second conv layer: Tactical pattern. The input will be the 32 feauture maps from conv1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third conv layer: Strategic Patterns
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        
        # At this point we've extracted all spatial patterns through the first section of our CNN. 
        # Now we need to combine them into a single evaluation score which can be done through the fully connected layers.
        # Note that Linear (also called dense or fully connected) performs output = (input x weights) + bias. This is basically the first layer of the neural network 
        self.fc1 = nn.Linear(in_features=128*8*8, # Input is the flattened conv3 output
                             out_features=256 # Output of the first layer will be 256 compressed feautures
                             )

        # Second layer of the nn note that input is now the output of first layer and we will compress it to 64 features 
        self.fc2 = nn.Linear(in_features=256, out_features=64)

        # At this point the network has distiled all the chess knowledge into just 64 numbers. And these numbers represent
        # high level strategic assessments (which the network will learn on its own) 
        # Apply the final layer which produces a single scalar value for our position evaluation
        self.fc3 = nn.Linear(in_features=64, out_features=1)


    # Define the Forward Pass (how the data flows through the network). Note this function is called automatically when we do: output = model(input)
    def forward(self, x):
        """ Parameters
                        x: torch.Tensor"""
        # Conv block 1 
        x = self.conv1(x)
        x = self.bn1(x) 

        # Apply ReLU (REctified Linear Unit) which basically does the following f(x) = max(0, x). In short what this does is it 
        # adds non-linearity so the network can learn complex patterns else it would just be a linear regression (cuz of the negative weights)
        # here it focueses on the positive patterns and ignores negative activations
        x = F.relu(x)

        # Conv block 2 
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Note we keep only positive activations (by applying relu) across all conv layers. 
        # At this point x has shape (batch_size, 128, 8, 8) which means for each posiiton in the batch, we have 128 feature maps, each 8x8 showing where 
        # strategic patterns appear on the board.

        # Convert the 3d tensor to a 1d vector so we can feed it into the fully connected layers. 
        x = x.view(x.size(0), -1)

        # Fully connected layer 1 
        x = self.fc1(x)

        # apply non linearity again 
        x = F.relu(x)

        # Fully connected layer 2 
        x = F.relu(self.fc2(x))

        # Final output 
        x = self.fc3(x)
        return x 
