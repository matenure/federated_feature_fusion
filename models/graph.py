## This file contains the codes for generating latent graphs.
## The GraphICDF class is based on the ICDF method, and the Graph class is based on Gumbel-softmax.
## Since our node numer is not large, we can directly use an n*n prob parameter.
# If we have too many nodes, we can parameter prob as a low-rank matrix prob = embed^T embed


from utilities.util import *

class GraphICDF(nn.Module):  # ICDF
    def __init__(self, n_node, n_dim=8, scale=1, batch_size=100, gamma=0.14):
        super(GraphICDF, self).__init__()
        self.n_node, self.n_dim = n_node, n_dim
        # self.embed = nn.Parameter(torch.randn(self.n_node, self.n_dim), requires_grad = True).to(device)
        # self.embed = nn.Parameter(torch.Tensor(self.n_node, self.n_dim), requires_grad=True).to(device)
        # torch.nn.init.xavier_uniform_(self.embed)
        self.prob = nn.Parameter(torch.Tensor(self.n_node, self.n_node))
        nn.init.uniform_(self.prob, 0, 1)
        self.scale = scale
        self.gamma = gamma
        self.batch_size = batch_size
        self.distribution = Normal(loc=torch.zeros(self.n_node, self.n_node, batch_size),
                                   scale=torch.ones(self.n_node, self.n_node, batch_size) * scale)

    def normalize_l2(self, X):
        return X/torch.max(torch.norm(X), torch.tensor([1e-6]))

    def forward(self, X):
        pass

    def sample_A(self, prob_param, tmp_batch_size):
        # prob = prob_param.clamp(min=0.0001,max= 0.9999).unsqueeze(-1)
        gamma = self.gamma
        # prob = torch.sigmoid(prob_param).unsqueeze(-1)
        prob = prob_param.unsqueeze(-1)

        if not tmp_batch_size == self.batch_size:
            distribution = Normal(loc=torch.zeros(self.n_node, self.n_node, tmp_batch_size),
                                  scale=torch.ones(self.n_node, self.n_node, tmp_batch_size) * self.scale)
            q = distribution.icdf(prob.cpu())
            s = distribution.sample()
        else:
            q = self.distribution.icdf(prob.cpu())
            s = self.distribution.sample()
        # A = (torch.max(q,s) - s) / (q-s) # hard sampling does not work well
        A = torch.sigmoid((q - s) / gamma)  # soft sampling works well
        return A

    def sample_A_sym(self, tmp_batch_size): #prob_param = self.prob
        prob = self.prob.clamp(min=0.0001, max=0.9999)
        A = self.sample_A(prob, tmp_batch_size)
        # Symmetrize each slice of A
        A = symmetrize(A).transpose(0,2).to(device)
        I = torch.diag(torch.ones(A.shape[1])).unsqueeze(0).to(device)

        return A+I #output: batch*N*N

    def create(self, tmp_batch_size, gradient = True):
        def _create(tmp_batch_size):
            #embed: n_node * n_dim
            # nor_embed = self.normalize_l2(torch.exp(embed))
            # nor_embed = embed
            # prod = torch.sigmoid(torch.mm(nor_embed, nor_embed.t()))

            A = self.sample_A_sym(tmp_batch_size)
            return A
        if gradient is False:
            with torch.no_grad():
                return _create(tmp_batch_size)
        return _create(tmp_batch_size)

    def generate(self, tmp_batch_size, n_sample):
        res = []
        for _ in range(n_sample):  # sample n_sample graphs one by one
            res.append(self.create(tmp_batch_size))
        return res

class Graph(nn.Module): # Gumbel-softmax
    def __init__(self, n_node, batch=None, tau = 0.1):
        super(Graph, self).__init__()
        self.n_node = n_node
        self.tau = tau
        self.prob = nn.Parameter(torch.Tensor(self.n_node, self.n_node)).to(device)
        nn.init.uniform_(self.prob, 0, 1)

    def forward(self, X):
        pass

    def sample_A(self, prob_param, tmp_batch_size=None, hard=False):
        logits_pos = torch.log(prob_param)
        logits_neg = torch.log(1 - prob_param)
        logits = torch.stack([logits_pos, logits_neg], -1).unsqueeze(-1)

        if tmp_batch_size is None:
            gumbel = Gumbel(loc=torch.zeros(self.n_node, self.n_node, 2, 1),
                            scale=torch.ones(self.n_node, self.n_node, 2, 1))
        else:
            gumbel = Gumbel(loc=torch.zeros(self.n_node, self.n_node, 2, tmp_batch_size),
                            scale=torch.ones(self.n_node, self.n_node, 2, tmp_batch_size))

        g = self.gumbel_softmax(logits, gumbel, hard=hard, dim=2)
        # A = (torch.max(g, 0)[0] - g[0]) / (g[1] - g[0])
        A = g[:, :, 0, :]
        return A

    def create(self, tmp_batch_size=None, hard=False):
        prob = self.prob.clamp(min=0.0001, max=0.9999)
        A = self.sample_A(prob, tmp_batch_size, hard)
        A = symmetrize(A).transpose(0,2).to(device)
        I = torch.diag(torch.ones(A.shape[1])).unsqueeze(0).to(device)

        return A + I  # output: batch*N*N

    def gumbel_softmax(self, logits, gumbel, hard=False, eps=1e-10, dim=-1):
        # type: (Tensor, int, float, bool, float, int) -> Tensor
        def _gen_gumbels():
            gumbels = gumbel.sample().to(device)
            if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
                gumbels = gumbel.sample()
            return gumbels

        gumbels = _gen_gumbels()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self.tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim=2)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def generate(self, tmp_batch_size=None, n_sample=1):
        res = []
        for _ in range(n_sample):  # sample n_sample graphs one by one
            res.append(self.create(tmp_batch_size=tmp_batch_size))
        return res

class GraphkNN(nn.Module):  # kNN Graph
    def __init__(self, n_node, n_dim=16, batch_size=100):
        super(GraphkNN, self).__init__()
        self.n_node, self.n_dim = n_node, n_dim
        # self.embed = nn.Parameter(torch.randn(self.n_node, self.n_dim), requires_grad = True).to(device)
        # self.embed = nn.Parameter(torch.Tensor(self.n_node, self.n_dim), requires_grad=True).to(device)
        # torch.nn.init.xavier_uniform_(self.embed)
        self.mlp = nn.Linear(n_dim, 8)
        self.batch_size = batch_size


    def forward(self, X):
        pass

    def sample_A(self, features):
        embeddings = self.mlp(features).transpose(0,1).reshape(self.n_node,-1)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = torch.mm(embeddings, embeddings.t())
        similarities = self.top_k(similarities, 5+1) ###k=10
        similarities = F.relu(similarities)
        return similarities.unsqueeze(0).repeat(features.shape[0],1,1)

    def top_k(self, raw_graph, K):
        values, indices = raw_graph.topk(k=int(K), dim=-1)
        assert torch.max(indices) < raw_graph.shape[1]
        mask = torch.zeros(raw_graph.shape).to(device)
        mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

        mask.requires_grad = False
        sparse_graph = raw_graph * mask
        return sparse_graph


    def sample_A_sym(self, features): #prob_param = self.prob
        A = self.sample_A(features)
        # Symmetrize each slice of A
        A = symmetrize(A.transpose(0,2)).transpose(0,2).to(device)
        I = torch.diag(torch.ones(A.shape[1])).unsqueeze(0).to(device)

        return A+I #output: batch*N*N

    def create(self, features, gradient = True):
        def _create(features):
            #embed: n_node * n_dim
            # nor_embed = self.normalize_l2(torch.exp(embed))
            # nor_embed = embed
            # prod = torch.sigmoid(torch.mm(nor_embed, nor_embed.t()))

            A = self.sample_A_sym(features)
            return A
        if gradient is False:
            with torch.no_grad():
                return _create(features)
        return _create(features)

    def generate(self, features):
        res = []
        res.append(self.create(features))
        return res
