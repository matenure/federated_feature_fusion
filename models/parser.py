import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description = "Run .")



    # parser.add_argument("--model_path",
    #                     nargs = "?",
    #                     default = "./saved_model.pth",
	#                 help = "Saved Model Path.")


    parser.add_argument("--epochs",
                        type = int,
                        default = 500,
	                help = "Number of training epochs. Default is 500.")

    parser.add_argument("--hidden",
                        type=int,
                        default=16,
                        help="Hidden dimension. Default is 16")

    parser.add_argument("--train_sample_num",
                        type=int,
                        default=1,
                        help="Number of Sampled Graphs in Training")

    parser.add_argument("--test_sample_num",
                        type=int,
                        default=10,
                        help="Number of Sampled Graphs in Testing")


    parser.add_argument("--seed",
                        type = int,
                        default = 2021,
	                help = "Random seed. Default is 2021 (old res is using 42).")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                help = "Dropout parameter. Default is 0.5.")

    parser.add_argument("--lr",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--tau",
                        type=float,
                        default=0.1,
                        help="temperature. Default is 0.1.")


    parser.add_argument("--KnownGraph",
                        dest='KnownGraph',
                        action='store_true',
                        help = "Whether use existing graph or not")


    parser.add_argument('--MLP',
                        dest='MLP',
                        action='store_true',
                        help = "Whether use MLP instead of GNN")

    parser.add_argument('--permute',
                        dest='permute',
                        action='store_true',
                        help="Whether permute")

    parser.add_argument("--reg",
                        dest='reg',
                        action='store_true',
                        help="Entropy regularization for permutation")

    parser.add_argument('--isGumbel',
                        dest='isGumbel',
                        action='store_true',
                        help="Whether Gumbel or ICDF")

    parser.add_argument('--local',
                        dest='local',
                        action='store_true',
                        help="Whether Training Local Models.")

    parser.add_argument("--isload",
                        dest='isload',
                        action='store_true',
                        help="loading, otherwise training")

    # parser.set_defaults(layers = [16, 16, 16])
    
    return parser.parse_args()
