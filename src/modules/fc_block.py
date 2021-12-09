from torch import nn


class FCBlock(nn.Module):
    def __init__(self, fc_sizes, dropout=0.15, end_activation=False):
        super(FCBlock, self).__init__()
        blocks = nn.ModuleList()

        for i in range(1, len(fc_sizes) - 1):
            blocks.append(nn.Linear(fc_sizes[i - 1], fc_sizes[i]))
            blocks.append(nn.LeakyReLU())

        # the last layer
        blocks.append(nn.Linear(fc_sizes[-2], fc_sizes[-1]))
        if end_activation:
            blocks.append(nn.LeakyReLU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
