from models.parser import parameter_parser
from models.embedding import NodeLearner
from models.fedgnn_transformer import *
from torch.distributions import Laplace

if __name__ == "__main__":
    # for reproducibility
    parser = parameter_parser()
    np_seed = 1000
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    lr = parser.lr
    n_iter = parser.epochs
    torch.manual_seed(parser.seed)
    n_train_sample = parser.train_sample_num
    n_test_sample = parser.test_sample_num
    tau = parser.tau
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


    # X_train, X_test, dY_train, dY_test = train_test_split(data['x'], data['y'], test_size = test_ratio)

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

    if parser.local:

        # # train local model
        nodes = []
        for i in range(no_series):
            print("Pre-training local model " + str(i) + " ...")
            node = NodeLearner(input_dim, no_event, hidden_dim = hidden_dim, stack_size = 1, window_size = window_size)
            node.fit(local_train[i], n_iter = 200, X_test = local_test[i], Y_test = Y_test, X_val=local_val[i], Y_val = Y_val)
            nodes.append(node.predictor)

        workspace.nodes = nodes
        torch.save(workspace, './results/saves/' + workspace.exp_name + '.pth')
    else:
        # # train local model
        nodes = []
        for i in range(no_series):
            node = NodeLearner(input_dim, no_event, hidden_dim=hidden_dim, stack_size=1, window_size=window_size)
            nodes.append(node.predictor)

        workspace.nodes = nodes
        # load local model
        workspace = torch.load('./results/saves/' + workspace.exp_name + '.pth', map_location=device)

        # combine local model
        print("Combining local models ...")
        print("Whether permute?", parser.permute)
        print("Whether Gumbel?", parser.isGumbel)

        model_name = "traffic-la_global_transformer"


        model_name = model_name +"lr_" + str(lr) + "_dim"+str(hidden_dim) + ".model"


        synthesizer = Trainer(input_dim, no_event, no_series, workspace.nodes, hidden_dim=hidden_dim, lr=lr,
                              permute=parser.permute, with_graph=False, model_name=model_name, permute_reg=parser.reg)  # must use the same hidden dim as each node
        if parser.isload:
            synthesizer.load_state_dict(torch.load('./results/savedmodels/' + model_name, map_location=device),
                                        strict=False)
            permutation_param = synthesizer.permutation_module.permutation_M
            permutation_matrix = torch.stack(
                [synthesizer.permutation_module.sinkhorn(permutation_param[i]) for i in range(no_series)], 0)
            np.savez('./results/middle_results/' + model_name + '.npz',
                     permutation_param=permutation_param.detach().numpy(),
                     permutation_matrix=permutation_matrix.detach().numpy(),
                     graph=torch.sigmoid(synthesizer.generator.prob).detach().numpy())
        else:
            synthesizer.fit(global_data, n_iter=n_iter, period=5, X_val=X_val[:, :, :no_series, :], Y_val=Y_val,
                        Xt=X_test[:, :, :no_series, :], Yt=Y_test)
