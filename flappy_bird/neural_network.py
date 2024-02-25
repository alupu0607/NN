import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.initial_epsilon = 0.1
        self.number_of_actions = 2
        self.gamma = 0.99
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.final_epsilon = 0.0001
        self.minibatch_size = 32

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.number_of_actions)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)

        return out
