import torch

from torch import nn

class PrunedConvNetIDS(nn.Module):
    def __init__(self, number_of_outputs=1, window_size=44, number_of_columns=116):
        super(PrunedConvNetIDS, self).__init__()
        # Input image: window_size rows (differences between messages) x number_of_columns nibbles.
        # Kept so the timing code can build a dummy input of the right shape.
        self.input_shape = (window_size, number_of_columns)
        # Each of the two 2x2 max poolings halves both sides (rounding down), and the last convolution has
        # 26 channels, so the flattened size is 26 x (rows // 4) x (columns // 4). For the paper's
        # 44 x 116 image this is 26 x 11 x 29 = 8294, the value the authors wrote by hand.
        flattened_size = 26 * ((window_size // 2) // 2) * ((number_of_columns // 2) // 2)

        self.feature_extraction_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=27, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(27, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=27, out_channels=26, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(26, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.binary_classification_layer_fc1 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=flattened_size, out_features=64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
        )

        self.binary_classification_layer_fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=number_of_outputs)
        )

    def forward(self, x):
        x = self.feature_extraction_layer(x)
        x = torch.flatten(x, 1)
        x = self.binary_classification_layer_fc1(x)
        x = self.binary_classification_layer_fc2(x)
        x = torch.sigmoid(x)
        return x

    def cnn_forward(self, x):
        x = self.feature_extraction_layer(x)
        x = torch.flatten(x, 1)

        return x

    def fc1_forward(self, x):
        x = self.feature_extraction_layer(x)
        x = torch.flatten(x, 1)
        x = self.binary_classification_layer_fc1(x)

        return x
