from utilities.util import *
import os
import uuid
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score


class Embedding(nn.Module):  # embedding a windowed time series into a flat feature vector
    def __init__(self, input_dim, hidden_dim = 100, stack_size = 1, window_size = 10):
        super(Embedding, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim  # input dimension, no. of the LSTM's hidden units
        self.stack_size, self.window_size = stack_size, window_size  # no. of the LSTM's hidden layers, window length
        self.summarizer = nn.LSTM(self.input_dim, self.hidden_dim, self.stack_size).to(device)  # LSTM unit
        self.aggregator = nn.Linear(self.window_size, 1).to(device)  # linear layer that aggregates the LSTM's output
        self.models = nn.ModuleList([self.summarizer, self.aggregator]).to(device)

    def forward(self, X):  # X is a tensor (window size, batch size, input dim)
        output, (hn, cn) = self.summarizer(X)  # output is a tensor (window size, batch size, hidden dim)
        output = output.permute(1, 2, 0)  # output is a tensor (batch size, hidden dim, window size)
        output = self.aggregator(output)  # output is a tensor (batch size, hidden dim, 1)
        return output.view(X.shape[1], -1), (hn, cn) # output is a matrix (batch size, hidden dim)


# class Node(nn.Module):  # a local model that predicts using data from one sensor
#     def __init__(self, input_dim, no_event, hidden_dim = 32, stack_size = 2, window_size = 10):
#         super(Node, self).__init__()
#         self.input_dim, self.hidden_dim, self.stack_size = input_dim, hidden_dim, stack_size
#         self.window_size, self.no_event = window_size, no_event
#         self.linear_embedding = nn.Linear(self.window_size*self.input_dim, self.hidden_dim,bias=False).to(device)
#         torch.nn.init.normal(self.linear_embedding.weight)
#         self.permute_M = nn.Parameter(torch.rand(self.hidden_dim, self.hidden_dim), requires_grad = True).to(device) # this parameter is actually not used. It is not easy to pass gradient in Trainer. It will be useful if one also wants to fine-tune local models.
#
#     def forward(self, X, normalize = True):  # X is (window size, batch size, input dim)
#         output, _ = self.embedding(X)  # output is (batch size, hidden dim)
#         output = self.predictor(output)  # output is (batch size, no_event)
#         if normalize:
#             output = F.softmax(output, dim = 1)  # output is (batch size, no_event)
#         return output
#
#     def summarize(self, X): # X is (window size, batch size, input dim)
#         with torch.no_grad():
#             output = X.permute(1, 2, 0)
#             output = output.view(X.shape[1], -1)
#             output = self.linear_embedding(output)
#             return output



class Node(nn.Module):  # a local model that predicts using data from one sensor
    def __init__(self, input_dim, no_event, hidden_dim = 32, stack_size = 2, window_size = 10, predictor=None):
        super(Node, self).__init__()
        self.input_dim, self.hidden_dim, self.stack_size = input_dim, hidden_dim, stack_size
        self.window_size, self.no_event = window_size, no_event
        self.embedding = Embedding(self.input_dim, hidden_dim = self.hidden_dim, stack_size = self.stack_size, window_size = self.window_size).to(device)
        if predictor is None:
            self.predictor = nn.Linear(self.hidden_dim, self.no_event).to(device)
        else:
            self.predictor = predictor
            for p in predictor.parameters():
                p.requires_grad = False
        self.linear_embedding = nn.Linear(self.window_size*self.input_dim, self.hidden_dim,bias=False).to(device)
        torch.nn.init.normal(self.linear_embedding.weight)
        self.models = nn.ModuleList([self.embedding, self.predictor, self.linear_embedding]).to(device)
        self.permute_M = nn.Parameter(torch.rand(self.hidden_dim, self.hidden_dim), requires_grad = True).to(device) # this parameter is actually not used. It is not easy to pass gradient in Trainer. It will be useful if one also wants to fine-tune local models.

    def forward(self, X, normalize = True, predictor=None):  # X is (window size, batch size, input dim)
        output, _ = self.embedding(X)  # output is (batch size, hidden dim)
        output = self.predictor(output)  # output is (batch size, no_event)
        if normalize:
            output = F.softmax(output, dim = 1)  # output is (batch size, no_event)
        return output

    def summarize(self, X):  # X is (window size, batch size, input dim) -- summarize each batch -- use after local training
        with torch.no_grad():
            output, _ = self.embedding(X)
            # output = F.dropout(output, p=0.5)
            return output  # output is (batch size, hidden dim) -- no need to maintain gradient


#
#
#     # def summarize_permute(self, X):
#     #     # with torch.no_grad():
#     #     #     output, _ = self.embedding(X) # output is (batch size, hidden dim)
#     #     self.permutate_M_bin = sinkhorn_transform(self.permute_M, gamma=0.01, iter=10)
#     #     output = torch.matmul(X, self.permutate_M_bin)
#     #     return output  # output is (batch size, hidden dim)

class NodeLearner(nn.Module):  # local trainer -- only use local data (from a single sensor)
    def __init__(self, input_dim, no_event, hidden_dim = 32, stack_size = 2, window_size = 10,prediction_layer=None):
        super(NodeLearner, self).__init__()
        self.input_dim, self.no_event, self.hidden_dim = input_dim, no_event, hidden_dim
        self.stack_size, self.window_size = stack_size, window_size
        self.predictor = Node(self.input_dim, self.no_event, hidden_dim = self.hidden_dim, stack_size = self.stack_size, window_size = self.window_size,predictor=prediction_layer).to(device)
        self.optimizer = opt.Adam(filter(lambda p:p.requires_grad, self.predictor.parameters()))
        self.regret = torch.nn.CrossEntropyLoss().to(device)
        self.ce_loss, self.acc, self.f1 = [], [], []
        self.test_acc, self.test_f1, self.test_auc, self.test_ap = 0, 0, 0, 0

    def loss(self, X, Y):  # cross-entropy batch loss -- averaged over loss items in batch
        pred = self.predictor(X, normalize = False)  # pred is (batch, no_event)
        return self.regret(pred, torch.argmax(Y, dim = 1))

    def accuracy(self, X, Y):
        with torch.no_grad():
            pred = torch.argmax(self.predictor(X, normalize = False), dim = 1)
            truth = torch.argmax(Y, dim = 1)
            return (pred == truth).sum().item() * 1.0 / pred.shape[0]

    def F1(self, X, Y):  #This only fits the binary classification case
        with torch.no_grad():
            pred = torch.argmax(self.predictor(X, normalize = False), dim = 1)
            truth = torch.argmax(Y, dim = 1)
            r = pred > 0
            tp = (pred[r] == truth[r]).sum().item() * 1.0
            fp = (pred[r] != truth[r]).sum().item() * 1.0
            fn = (pred != truth).sum().item() * 1.0 - fp
            return tp / (tp + 0.5 * (fp + fn))

    def cal_scores(self, pred, Y):
        num_class = Y.shape[-1]
        target_y = torch.argmax(Y, dim=1).cpu().data.numpy()
        if num_class == 2:
            pred_y = pred[:, 1].cpu().data.numpy()  # prob of target label.
            roc_auc = roc_auc_score(target_y, pred_y)
            acc = accuracy_score(target_y, pred_y > 0.5)
            pr_auc = average_precision_score(target_y, pred_y)
            f1 = f1_score(target_y, pred_y > 0.5)
        else:
            pred_y = torch.argmax(pred, dim=1).cpu().data.numpy()
            pred = pred.cpu().data.numpy()
            roc_auc = roc_auc_score(target_y, pred, multi_class='ovo', average='macro')
            acc = accuracy_score(target_y, pred_y)
            pr_auc = None  # there is no average precision score for the multi-class case
            f1 = f1_score(target_y, pred_y, average='macro')
        return acc, f1, roc_auc, pr_auc

    def pred_scores(self, X, Y):
        with torch.no_grad():
            self.predictor.eval()
            pred = self.predictor(X, normalize=True)
            return self.cal_scores(pred, Y)




    def fit(self, dataset, n_iter = 100, verbal = True, period = 10, X_val=None, Y_val=None, X_test = None, Y_test = None):
        self.ce_loss, self.acc, self.f1 = [], [], []
        self.best_acc = 0.0
        self.step_counter = 0
        tmp_model_path = str(uuid.uuid4())+".pth"
        for i in range(1, n_iter + 1):
            ave, n_batch = 0.0, 0.0
            for (X, Y) in dataset:  # dataset is a data loader object -- Y is (batch, 1) -- X is (batch, window size, input dim)
                self.predictor.train()
                self.optimizer.zero_grad()
                X = X.permute(1, 0, 2)  # re-format X to (window size, batch, input dim)
                func = self.loss(X, Y)
                ave += func.item()  # accumulate batch loss
                n_batch += 1.0  # accumulate no. batch
                func.backward()
                self.optimizer.step()
            self.ce_loss.append(ave / n_batch)  # store average loss (over batches) for this epoch

            if X_val is not None:
                # self.cur_acc = self.accuracy(X_val.permute(1, 0, 2),Y_val)  # re-format X_val to (window size, batch, input dim)
                self.cur_acc, _, _, _ = self.pred_scores(X_val.permute(1, 0, 2),
                                                      Y_val)
                if self.cur_acc >= self.best_acc:
                    self.best_acc = self.cur_acc
                    self.step_counter = 0
                    if X_test is not None:
                        # self.test_acc = self.accuracy(X_test.permute(1, 0, 2),
                        #                               Y_test)
                        # self.test_f1 = self.F1(X_test.permute(1, 0, 2), Y_test)
                        self.test_acc, self.test_f1, self.test_auc, self.test_ap = self.pred_scores(X_test.permute(1, 0, 2),
                                                      Y_test)
                    torch.save(self.predictor.state_dict(), tmp_model_path)
                else:
                    self.step_counter +=1
                    if self.step_counter > 20: #ealy stopping
                        break

            if verbal is True and i % period == 0:
                if X_test is None:
                    print("Training Epoch {}: CE (train) = {}".format(i, self.ce_loss[-1]))
                else:
                    print("Training Epoch {}: CE (train) = {} | ACC (test) = {} | F1 (test) = {} | RocAUC (test) = {} | PRAUC (test) = {}".format(i, self.ce_loss[-1], self.test_acc, self.test_f1, self.test_auc, self.test_ap))
        self.predictor.load_state_dict(torch.load(tmp_model_path))
        os.remove(tmp_model_path)


    def forward(self):
        pass

