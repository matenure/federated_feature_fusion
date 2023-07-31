from models.parser import parameter_parser
from models.embedding import NodeLearner
from models.fedgnn import *



def cal_scores(pred, Y):
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
    # input_dim, window_size, no_event, no_series = 1, 12, 2, 6
    # input_dim, window_size, no_event, no_series = 1, 12, 2, 10
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
    workspace = Workspace(no_series, exp_name = "traffic_"+str(no_series) + "_node_local_model")


    nodes = []
    for i in range(no_series):
        node = NodeLearner(input_dim, no_event, hidden_dim=hidden_dim, stack_size=1, window_size=window_size)
        nodes.append(node.predictor)

    workspace.nodes = nodes
    # load local model
    workspace = torch.load('./results/saves/' + workspace.exp_name + '.pth', map_location=device)

    # combine local model
    print("Combining local models ...")

    voting = True
    preds_forsave = []

    with torch.no_grad():
        if voting:
            preds = []
            for i in range(no_series):
                workspace.nodes[i].eval()
                pred = workspace.nodes[i](local_test[i].permute(1, 0, 2), normalize=True) # (N, no_event)
                preds_forsave.append(pred)
                preds.append(F.softmax(pred*100.0,-1)) # make it close to binary

            np.savez("./results/middle_results/traffic-la_local_res.npz", preds = torch.stack(preds_forsave,0).detach().numpy(), truth=Y_test)
            final_pred = torch.stack(preds, 1)
            final_pred = torch.argmax(final_pred.sum(1) , -1)
            target_y = torch.argmax(Y_test, dim=1).cpu().data.numpy()
            test_acc = accuracy_score(target_y, final_pred)
            test_f1 = f1_score(target_y, final_pred)
            #     preds.append(torch.argmax(pred, dim = 1))
            # final_pred = torch.stack(preds, 1).to(float)
            # final_pred = (final_pred.mean(1)>0.5).long()
            # truth = torch.argmax(Y_test, dim=1)
            # test_acc = (final_pred == truth).sum().item() * 1.0 / final_pred.shape[0]
            # r = final_pred > 0
            # tp = (final_pred[r] == truth[r]).sum().item() * 1.0
            # fp = (final_pred[r] != truth[r]).sum().item() * 1.0
            # fn = (final_pred != truth).sum().item() * 1.0 - fp
            # test_f1 = tp / (tp + 0.5 * (fp + fn))

            print("voting test_acc, test_f1", test_acc, test_f1)
        else:
            best_acc_val = 0.0
            best_res = []
            for i in range(no_series):
                workspace.nodes[i].eval()
                pred_val = workspace.nodes[i](local_val[i].permute(1, 0, 2), normalize=True) # (N, no_event)
                acc_val, f1_val, _, _ = cal_scores(pred_val, Y_val)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    pred_test = workspace.nodes[i](local_test[i].permute(1, 0, 2), normalize=True) # (N, no_event)
                    best_res = cal_scores(pred_test, Y_test)
            print("Best local model: test_acc, test_f1, roc, pr", best_res)
