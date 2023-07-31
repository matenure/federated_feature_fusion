
from utilities.util import *

def normalize(M, dim = 0):
    norm = torch.sum(M, dim = dim, keepdim = True) ** (-1.0)
    return M * norm

class Identifier(nn.Module):
    def __init__(self, input_dim, no_sensor, C = 10, gamma = 0.1):
        super(Identifier, self).__init__()
        self.input_dim, self.C, self.gamma, self.no_sensor = input_dim, C, gamma, no_sensor
        self.M = nn.Parameter(torch.rand(self.no_sensor, self.input_dim, self.input_dim), requires_grad = True)
        self.params = nn.ParameterList([self.M]).to(device)

    def forward(self, X):  # X is (batch, no_sensor, input dim)
        K = torch.exp(-self.M / self.gamma)
        for _ in range(self.C):
            K = normalize(normalize(K, dim = 1), dim = 2)
        output = X[:, :, None, :] @ K[None, :, :, :]  # output is (batch, no_sensor, 1, input dim)
        return output.view(X.shape[0], X.shape[1], -1)  # output is (batch, no_sensor, input_dim)




