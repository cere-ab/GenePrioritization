import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.data.batch import Batch


class GNN7L_SAGEConv(nn.Module):
    """
    <Purpose>

    Attributes
    ----------
    atr1 : dtype
        <purpose>

    Methods
    -------
    __init__():
        Constructor
    method1(param):
        <purpose>
    
    """

    def __init__(self, in_features, out_classes):
        """
        <Purpose>

        Parameter
        ---------
        arg1 : dtype
            <purpose>

        Raises
        ------
        err1
            <purpose>

        Returns
        -------
        dtype
            <purpose>

        """

        super(GNN7L_SAGEConv, self).__init__()
        self.conv1 = SAGEConv(in_features, 16, aggr='max')
        self.conv2 = SAGEConv(16, 16, aggr='max')
        self.conv3 = SAGEConv(16, 16, aggr='max')
        self.conv4 = SAGEConv(16, 16, aggr='max')
        self.conv5 = SAGEConv(16, 16, aggr='max')
        self.conv6 = SAGEConv(16, 16, aggr='max')
        self.conv7 = SAGEConv(16, out_classes, aggr='max')

    def arguments_read(self, *args, **kwargs):
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        return x, edge_index, batch

    def forward(self, data=None, edge_index=None, *args, **kwargs):
        """
        <Purpose>

        Parameter
        ---------
        arg1 : dtype
            <purpose>

        Raises
        ------
        err1
            <purpose>

        Returns
        -------
        dtype
            <purpose>

        """
        
        x = None
        if data is None and edge_index is None:
            x, edge_index, batch = self.arguments_read(*args, **kwargs)
        elif data is not None and edge_index is None:
            x, edge_index = data.x, data.edge_index
        elif data is not None and edge_index is not None:
            x = data

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv7(x, edge_index))

        return F.log_softmax(x, dim=1)
