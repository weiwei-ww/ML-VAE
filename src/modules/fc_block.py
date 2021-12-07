from torch import nn


class FCBlock(nn.Module):
    def __init__(self, fc_sizes, dropout=0.15):
        super(FCBlock, self).__init__()
        blocks = nn.ModuleList()

        for i in range(1, len(fc_sizes)):
            blocks.append(nn.Linear(fc_sizes[i - 1], fc_sizes[i]))
            blocks.append(nn.LeakyReLU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
