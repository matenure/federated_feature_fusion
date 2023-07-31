from models.gnn import *
from models.identifier import *
from models.graph import *
from torch.distributions import Categorical
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

class InferenceNet(nn.Module):  # accept the stacked embedding collected from N sensors (N = no_sensor)
    def __init__(self, input_dim, no_event, no_sensor, hidden_dim = 8, with_graph = True, istrain=True):
        super(InferenceNet, self).__init__()
        self.input_dim, self.no_event, self.no_sensor, self.hidden_dim = input_dim, no_event, no_sensor, hidden_dim
        self.withGraph = with_graph
        self.identifier = Identifier(self.input_dim, self.no_sensor).to(device)
        self.predictor = Predictor(self.input_dim, self.no_event, self.hidden_dim, withGraph=self.withGraph).to(device)
        self.models = nn.ModuleList([self.identifier, self.predictor]).to(device)
        self.istrain = istrain

    def forward(self, X, A, mask=None, normalize = True):  # X is (batch, no_sensor, input dim) -- A is an adjacent matrix (no_sensor, no_sensor)
        # output = self.identifier(X)  # output is (batch, no_sensor, input dim)
        output = X  # this is the same as turning off the matching component
        output = self.predictor(output, A, mask=mask, normalize = normalize)  # output is (batch, no_event)
        return output

class Permutation(nn.Module):
    def __init__(self, num_node, hidden_dim, exp=True, gamma=0.01, iter=5):
        super(Permutation, self).__init__()
        self.num_node = num_node
        self.hidden_dim = hidden_dim
        self.exp = exp
        self.gamma = gamma
        self.iter = iter
        self.permutation_M = nn.Parameter(torch.FloatTensor(self.num_node, self.hidden_dim, self.hidden_dim), requires_grad = True).to(device)
        nn.init.xavier_uniform_(self.permutation_M)

        # #identity+noise initialization
        # self.permutation_M = nn.Parameter(torch.randn(self.num_node, self.hidden_dim, self.hidden_dim)*0.1 + torch.eye(self.hidden_dim).unsqueeze(0),
        #                                   requires_grad=True).to(device)

    def sinkhorn(self, M):
        if self.exp:
            M1 = M.clamp(min=-10.0, max=10.0)
            M1 = torch.sigmoid(M1) # it will make the convergence much faster but easy to get nan values later. Removing this step does not solve the nan problem, so we stop updating M after 20epochs.
            C = torch.exp(M1/self.gamma)
        else:
            C = torch.pow(M, 2) # it is better to add entropy regularization when using this form.
            # C = M.clamp_min(min=0.0)
        for i in range(self.iter):
            C = norm1(C.transpose(0, 1)).transpose(0, 1)
            C = norm1(C)
        return C

class Trainer(nn.Module):
    def __init__(self, input_dim, no_event, no_sensor, nodes, hidden_dim = 16, lr=0.01, batch_size=100, permute=False,with_graph = True, isGumbel=False, permute_reg=False, tau=0.1, model_name="global.model"):
        super(Trainer, self).__init__()
        self.input_dim, self.no_event, self.no_sensor, self.hidden_dim = input_dim, no_event, no_sensor, hidden_dim
        self.withGraph = with_graph
        self.reg = permute_reg #whether add regularization for permutation
        self.predictor = InferenceNet(self.hidden_dim, self.no_event, self.no_sensor, hidden_dim = 8, with_graph=self.withGraph).to(device)
        if isGumbel:
            self.generator = Graph(self.no_sensor, tau=tau).to(device)
        else:
            self.generator = GraphICDF(self.no_sensor, batch_size=batch_size, gamma=tau).to(device)
        self.nodes = nodes  # list of local models which have been pre-trained on local sensor data
        self.permutation_module = Permutation(self.no_sensor, self.hidden_dim, exp=False, gamma=0.01, iter=10)
        self.model = nn.ModuleList([self.predictor, self.generator, self.permutation_module]).to(device)
        self.model.extend(self.nodes)

        self.optimizer = opt.Adam(self.model.parameters(),lr=lr)
        self.regret = nn.CrossEntropyLoss().to(device)
        self.permute = permute
        self.model_name = model_name
        self.ce_loss, self.acc, self.f1 = None, None, None


    def permute_summarize(self, Z):
        output = torch.matmul(Z, self.permutate_M_bin)
        return output

    def ask(self, X, start=True):  # X is (sensor, window size, batch, input dim) -- probably cache all these in advance
        if self.permute and start: # adding the parameter start to make training faster in the beginning
            H = torch.stack([torch.matmul(self.nodes[i].embedding(X[i])[0], self.permutation_module.permutation_M[i]) for i in range(len(self.nodes))]).to(
                device)  # H is (no_sensor, batch, hidden dim)
        else:
            H = torch.stack([self.nodes[i].embedding(X[i])[0] for i in range(len(self.nodes))]).to(device)  # H is (no_sensor, batch, hidden dim)

        H = H.permute(1, 0, 2)  # H is (batch, no_sensor, hidden dim)
        return H

    def loss(self, X, Y, A, mask=None):  # X is (batch, no_sensor, hidden_dim)
        pred = self.predictor(X, A, mask=mask, normalize = False)  # pred is (batch, no_event)
        return self.regret(pred, torch.argmax(Y, dim = 1))

    def permute_entropy(self):
        ent = 0
        # for i in range(self.no_sensor):
        #     ent += Categorical(self.permutation_M_bin[i]).entropy().mean()
        #     ent += Categorical(self.permutation_M_bin[i].t()).entropy().mean()
        return ent/self.no_sensor


    def predict(self, X, graph = None, mask=None, n_sample = 10, normalize = False):  # X is (batch, no_sensor, hidden_dim)
        with torch.no_grad():  # prediction phase -- no need to propagate gradient
            if graph is None:
                graphs = self.generator.generate(X.shape[0], n_sample)  # generate n_sample graphs
            else:
                graphs = [graph]  # ignore the graph generator if the ground-truth is given
                n_sample = 1
            pred = self.predictor(X, graphs[0], mask=mask, normalize = normalize)
            for i in range(1, n_sample):
                pred += self.predictor(X, graphs[i], mask=mask, normalize = normalize)
            pred = (pred * 1.0) / n_sample  # pred is (batch, no_event)
            return pred

    def accuracy(self, pred, Y):  # X is (sensor, window size, batch, input dim) -- Y is (batch, no_event)
        with torch.no_grad():  # no need to propagate gradient during test phase
            pred = torch.argmax(pred, dim = 1)
            truth = torch.argmax(Y, dim = 1)
            return (pred == truth).sum().item() * 1.0 / pred.shape[0]

    def F1(self, pred, Y):
        with torch.no_grad():  # no need to propagate gradient during test phase
            pred = torch.argmax(pred, dim=1)
            truth = torch.argmax(Y, dim=1)
            r = pred > 0
            tp = (pred[r] == truth[r]).sum().item() * 1.0
            fp = (pred[r] != truth[r]).sum().item() * 1.0
            fn = (pred != truth).sum().item() * 1.0 - fp
            return tp / (tp + 0.5 * (fp + fn))



    def pred_scores(self, X, Y, graph = None, mask=None, n_sample = 10):
        self.model.eval()
        pred = self.predict(X, graph = graph, mask=mask, n_sample = n_sample, normalize=False)
        loss = self.regret(pred, torch.argmax(Y, dim = 1))
        pred = F.softmax(pred,1)
        # pred = torch.argmax(pred, dim=1)
        num_class = Y.shape[-1]
        target_y = torch.argmax(Y, dim=1).cpu().data.numpy()
        if num_class ==2:
            pred_y = pred[:,1].cpu().data.numpy() #prob of target label.
            roc_auc = roc_auc_score(target_y, pred_y)
            acc = accuracy_score(target_y,pred_y>0.5)
            pr_auc = average_precision_score(target_y, pred_y)
            f1 = f1_score(target_y, pred_y>0.5)
        else:
            pred_y = torch.argmax(pred, dim=1).cpu().data.numpy()
            pred = pred.cpu().data.numpy()
            roc_auc = roc_auc_score(target_y, pred, multi_class='ovo', average='macro')
            acc = accuracy_score(target_y, pred_y)
            pr_auc = None  # there is no average precision score for the multi-class case
            f1 = f1_score(target_y, pred_y, average='macro')
        return acc, f1, roc_auc, pr_auc, loss


    def fit(self, dataset, graph = None, n_iter = 100, n_train_sample = 1, n_test_sample = 10, verbal = True, period = 10, X_val = None, Y_val = None, mask_val = None, Xt = None, Yt = None, mask_test=None):
        self.ce_loss, self.acc, self.f1 = [], [], []

        start_time = time.time()
        start_permute_epoch=0

        self.best_loss = 100000.0
        self.step_counter = 0
        for i in range(1, n_iter + 1):
            if graph is not None:
                graphs = [graph]
                n_train_sample = 1
            ave, n_batch = 0.0, 0.0
            for (X, Y, mask_train) in dataset:  # dataset is a data loader object -- Y is (batch, no_event) -- X is (batch, window size, sensor, input dim)
                X = X.permute(2, 1, 0, 3)  # re-format X to (sensor, window size, batch, input dim)
                X = self.ask(X)  # X is (batch, no_sensor, hidden_dim)
                if mask_val is not None: #For implementation efficiency, we never make mask_train None. Instead we can judge the existing of masks by mask_val
                    mask = mask_train  #(batch, no_sensor)
                else:
                    mask = None
                self.model.train()
                self.optimizer.zero_grad()
                if graph is None:
                    graphs = self.generator.generate(X.shape[0], n_train_sample)  # generate n_train_sample graphs for training
                func = self.loss(X, Y, graphs[0], mask=mask)
                for u in range(1, n_train_sample):  # for each sampled graph
                    func += self.loss(X, Y, graphs[u], mask=mask)

                func *= 1.0 / n_train_sample  # average batch loss over no. of sampled graph
                # if self.permute and self.reg:
                #     func += 0.1*self.permute_entropy()
                ave += func.item()  # accumulate batch loss
                n_batch += 1.0  # accumulate no. batch
                func.backward(retain_graph = False)
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 10000)
                self.optimizer.step()
            self.ce_loss.append(ave / n_batch)  # store average loss (over batches) for this epoch

            if X_val is not None:
                X_val_sum = self.ask(X_val.permute(2, 1, 0, 3), i>=start_permute_epoch)
                val_acc, val_f1, val_roc, val_pr, val_loss = self.pred_scores(X_val_sum, Y_val, graph= graph, mask=mask_val, n_sample=n_test_sample)
                print("Training Epoch {}: CE (train) = {} | CE (val) = {} | ACC (val) = {} | F1 (val) = {} | RocAUC (val) = {} | PRAUC (val) = {}".format(
                        i, self.ce_loss[-1], val_loss, val_acc, val_f1, val_roc, val_pr))

                if val_loss<self.best_loss:
                    self.best_loss = val_loss
                    self.step_counter = 0
                    torch.save(self.state_dict(), './results/savedmodels/'+self.model_name)
                    Xt_sum = self.ask(Xt.permute(2, 1, 0, 3), i>=start_permute_epoch)
                    test_acc, test_f1, test_roc, test_pr, _ = self.pred_scores(Xt_sum, Yt, graph=graph, mask=mask_test, n_sample=n_test_sample)
                else:
                    self.step_counter +=1
                    if self.step_counter>50:
                        break


            if verbal is True and i % period == 0:
                if Xt is None or Yt is None:
                    print("No test data.")
                else:
                    print("Update Test Res | ACC (test) = {} | F1 (test) = {} | RocAUC_macro = {} | PRAUC = {}".format(
                        test_acc, test_f1, test_roc, test_pr))
                    # print("Alternative Test ACC and F1,", val_test_acc, val_test_f1)
                    print("Time,", time.time() - start_time)
                    start_time = time.time()
            #         if graph is None:
            #             self.acc.append(self.accuracy(Xt, Yt))
            #             self.f1.append(self.F1(Xt, Yt))
            #         else:
            #             self.acc.append(self.accuracy(Xt, Yt, graph = graph))
            #             self.f1.append(self.F1(Xt, Yt, graph = graph))
            #         print("Training Epoch {}: CE (train) = {} | ACC (test) = {} | F1 (test) = {}".format(i, self.ce_loss[-1], self.acc[-1], self.f1[-1]))
            #         print("Time,",time.time()-start_time)
            #         start_time = time.time()

    def forward(self):
        pass

