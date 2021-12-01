from torch import nn

from speechbrain.lobes.models.CRDNN import DNN_Block


class FCBlock(nn.Module):
    def __init__(self, fc_sizes, dropout=0.15):
        super(FCBlock, self).__init__()
        blocks = nn.ModuleList()

        # dnn_block = DNN_Block(
        #     input_shape=(None, input_size),
        #     neurons=fc_size,
        #     dropout=dropout
        # )
        # blocks.append(dnn_block)

        # if type(fc_sizes) is not list:
        #     fc_sizes = [fc_sizes]
        #
        # blocks.append(nn.Linear(input_size, fc_sizes[0]))
        # blocks.append(nn.LeakyReLU())

        for i in range(1, len(fc_sizes)):
        # for fc_size in fc_sizes[1:]:
            blocks.append(nn.Linear(fc_sizes[i - 1], fc_sizes[i]))
            blocks.append(nn.LeakyReLU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
