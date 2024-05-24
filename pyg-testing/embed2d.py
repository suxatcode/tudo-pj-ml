# data
import torch
import pandas as pd

nodes = pd.read_csv("../lg-nodes.csv.gz", compression='gzip')
edges = pd.read_csv("../lg-edges.csv.gz", compression='gzip')

unique_nodes = pd.concat([edges["from_id"], edges["to_id"]]).unique()
node_to_index = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
edges["from_index"] = edges["from_id"].map(node_to_index)
edges["to_index"] = edges["to_id"].map(node_to_index)
edge_index = torch.tensor(
    [edges["from_index"].to_numpy(), edges["to_index"].to_numpy()], dtype=torch.long
)

# model
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

x = torch.tensor([[i] for i in range(len(edge_index[0]))], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)


class GCN(torch.nn.Module):
    """Simple GCN model"""

    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return x


model = GCN()
out = model(data)
print(out)
