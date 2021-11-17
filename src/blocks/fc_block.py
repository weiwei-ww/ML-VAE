from torch import nn

from speechbrain.lobes.models.CRDNN import DNN_Block


class FCBlock(nn.Module):
    def __init__(self, input_size, fc_size, dropout=0.15, num_layers=0):
        super(FCBlock, self).__init__()
        blocks = nn.ModuleList()

        dnn_block = DNN_Block(
            input_shape=(None, input_size),
            neurons=fc_size,
            dropout=dropout
        )
        blocks.append(dnn_block)

        for _ in range(num_layers):
            dnn_block = DNN_Block(
                input_shape=(None, fc_size),
                neurons=fc_size,
                dropout=dropout
            )
            blocks.append(dnn_block)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
