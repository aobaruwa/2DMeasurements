import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_dim=3, h_dims=[16,16], n_classes=3*75):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, h_dims[0], activation=F.relu)
        self.conv2 = dglnn.GraphConv(h_dims[0], h_dims[1], activation=None)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(h_dims[1], n_classes)

    def forward(self, blocks, x):
        x = self.conv1(blocks, x)
        x = self.dropout(x)
        x = self.conv2(blocks, x)
        x = self.dropout(x)
        with blocks.local_scope():
            blocks.ndata['h'] = x
            # Calculate graph representation by average readout.
            hg = dgl.max_nodes(blocks, 'h')
            return self.linear(hg)
      

class GAT(nn.Module):
    def __init__(self, in_dim=3, h_dims=[16,16], n_classes=3*75):
        super().__init__()
        self.conv1 = dglnn.GATConv(in_dim, h_dims[0], num_heads=1, activation=F.relu)
        self.conv2 = dglnn.GATConv(h_dims[0], h_dims[1], num_heads=1, activation=F.relu)
        self.maxPool = dglnn.MaxPooling()
        self.linear = nn.Linear(h_dims[1], n_classes)

    def forward(self, blocks, x):
        x = self.conv1(blocks, x)
        x = self.conv2(blocks, x)
        with blocks.local_scope():
            blocks.ndata['h'] = x
            # Calculate graph representation by average readout.
            hg = dgl.max_nodes(blocks, 'h')
            return self.linear(hg)
