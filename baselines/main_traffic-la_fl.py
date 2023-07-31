from models.parser import parameter_parser
from models.embedding import NodeLearner
from models.fedgnn import *

if __name__ == "__main__":
    # for reproducibility
    parser = parameter_parser()
    np_seed = 1000
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_seed = 20000
    lr = parser.lr
    n_iter = parser.epochs
    torch.manual_seed(parser.seed)
    n_train_sample = parser.train_sample_num
    n_test_sample = parser.test_sample_num
    # n_test_sample = 1

    # set up parameters
    input_dim, window_size, no_event, no_series = 1, 12, 2, None

    batch_size, test_ratio = 100, 0.2
    hidden_dim = parser.hidden

    # load data
    data = np.load('./event_classification_data/traffic/metr-la_indexed.npz')

    f = open('./event_classification_data/traffic/adj_mx_la.pkl', 'rb')
    graph_data = pickle.load(f, encoding = 'latin1')
    graph = graph_data[2] > 0
    graph = graph.astype(float)
    graph = torch.FloatTensor(graph)
    if no_series is not None:
        graph = graph[:no_series, :no_series]
    else:
        no_series = graph.shape[0]

    X_train = data['x'][data['train_index']]
    X_val = data['x'][data['val_index']]
    X_test = data['x'][data['test_index']]
    dY_train = data['y'][data['train_index']]
    dY_val = data['y'][data['val_index']]
    dY_test = data['y'][data['test_index']]

    mask_X = np.ones((data['x'].shape[0], no_series))
    mask_train = mask_X[data['train_index']]
    mask_train = torch.FloatTensor(mask_train).to(device)
    mask_val = None
    mask_test = None


    X_train = torch.FloatTensor(X_train).permute(0, 2, 1, 3).to(device)
    X_val = torch.FloatTensor(X_val).permute(0, 2, 1, 3).to(device)
    X_test = torch.FloatTensor(X_test).permute(0, 2, 1, 3).to(device)

    Y_train = torch.zeros(X_train.shape[0], 2).to(device)
    Y_val = torch.zeros(X_val.shape[0], 2).to(device)
    Y_test = torch.zeros(X_test.shape[0], 2).to(device)

    for i in range(X_train.shape[0]):
        Y_train[i, dY_train[i]] = 1.0

    for i in range(X_val.shape[0]):
        Y_val[i, dY_val[i]] = 1.0

    for i in range(X_test.shape[0]):
        Y_test[i, dY_test[i]] = 1.0

    # set up global data generator
    dataset = TensorDataset(X_train[:, :, :no_series, :], Y_train, mask_train)
    global_data = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # separate into local data sources
    local_train, local_val, local_test = [], [], []
    for i in range(no_series):
        source_train = TensorDataset(X_train[:, :, i, :].view(X_train.shape[0], X_train.shape[1], input_dim), Y_train)
        local_train.append(DataLoader(source_train, batch_size = batch_size, shuffle = True))
        local_val.append(X_val[:, :, i, :].view(X_val.shape[0], X_val.shape[1], input_dim))
        local_test.append(X_test[:, :, i, :].view(X_test.shape[0], X_test.shape[1], input_dim))

    # initialize workspace
    workspace = Workspace(no_series, exp_name = "traffic-la_common_local_model")


    # # train a common local model
    node = NodeLearner(input_dim, no_event, hidden_dim=hidden_dim, stack_size=1, window_size=window_size)
    best_val_acc = 0.0
    for iter in range(200):
        ave, n_batch = 0.0, 0.0
        for i in range(no_series):
            for (X,Y) in local_train[i]:  # dataset is a data loader object -- Y is (batch, 1) -- X is (batch, window size, input dim)
                node.predictor.train()
                node.optimizer.zero_grad()
                X = X.permute(1, 0, 2)  # re-format X to (window size, batch, input dim)
                func = node.loss(X, Y)
                ave += func.item()  # accumulate batch loss
                n_batch += 1.0  # accumulate no. batch
                func.backward()
                node.optimizer.step()
        node.ce_loss.append(ave / n_batch)
        ave_val = 0.0
        for i in range(no_series):
            val_acc, _, _, _ =node.pred_scores(local_val[i].permute(1, 0, 2),
                                                      Y_val)
            ave_val += val_acc
        ave_val = ave_val/no_series
        print("{}th epoch val_acc: {}".format(iter, ave_val))
        if ave_val > best_val_acc:
            best_val_acc = ave_val
            with torch.no_grad():
                node.predictor.eval()
                test_acc = 0.0
                test_f1 = 0.0
                test_auc =0.0
                for i in range(no_series):
                    tmp_acc, tmp_f1, auc, _ = node.pred_scores(local_test[i].permute(1, 0, 2), Y_test)
                    test_acc += tmp_acc
                    test_f1 += tmp_f1
                    test_auc += auc
                print("update test_acc, test_f1, test_auc", test_acc/no_series, test_f1/no_series, test_auc/no_series)


            workspace.nodes = node
            torch.save(workspace, './results/saves/' + workspace.exp_name + '.pth')
