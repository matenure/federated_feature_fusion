## This is the code for the predictor of global model. We can use either a GNN or an MLP depending on the parameter "withGraph".
## For GNN model, we implement it as a two-layer GCN, with skip connections (you can also use the original GCN by setting skip_connection=False).
## Note that the input "A" is not the original adjacency matrix but (adj+I).

from utilities.util import *
from models.transformer import *
import math

class GNN(nn.Module):  # a GNN layer
    def __init__(self, n_in, n_out, bias=True, skip_connection=True):
        super(GNN, self).__init__()
        self.n_in, self.n_out = n_in, n_out
        self.W = nn.Parameter(torch.Tensor(n_in, n_out), requires_grad = True)
        self.skip_connection = skip_connection
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, X, A, mask = None):  # X is (batch_size, N, n_in), A is (batch_size, N, N) -- output is (batch_size, N, n_out)
        # A already added the self-loop (diagnonal is 1).
        # B = torch.mm(A, torch.diag(torch.sum(A, dim = 1) ** (-1.0)))  # normalization
        # return F.relu(B @ (X @ self.W[None, :, :]))
        out = torch.matmul(X, self.W)

        deg_inv_sqrt = A.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        if self.skip_connection:
            if len(adj.shape)==3:
                adj = adj + torch.diag(torch.ones(X.shape[1])).unsqueeze(0).to(device)
            else:
                adj = adj + torch.diag(torch.ones(X.shape[1])).to(device)
        out = torch.matmul(adj, out)
        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = torch.mul(out, mask.unsqueeze(-1))

        return out



class Predictor(nn.Module):  # 2-layer GNN model
    def __init__(self, input_dim, no_event, hidden_dim = 8, no_sensor=10, withGraph = True):
        super(Predictor, self).__init__()
        self.input_dim, self.hidden_dim, self.no_event = input_dim, hidden_dim, no_event
        self.withGraph = withGraph
        self.no_sensor = no_sensor
        self.transformer = TransformerModel(self.no_sensor, input_dim, d_model=input_dim, nhead=4, d_hid=hidden_dim, nlayers=2)
        self.mlp2 = nn.Linear(self.hidden_dim*2, self.no_event).to(device)
        self.models = nn.ModuleList([self.transformer, self.mlp2]).to(device)

    def forward(self, X, normalize = True):
        output = torch.relu(self.transformer(X)) #batch, hidden_dim

        output = torch.mean(output, dim=1)
        output = self.mlp2(output)  # batch, no_event
        if normalize:
            output = F.softmax(output, dim=1)  # output is (batch, no_event)
        return output

