import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--device', type=str, default='cuda', help='device to use [cuda, cpu]')
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--n_cls', type=int, default=10, help="number of classes")

    # other arguments
    parser.add_argument('--l_tol', type=int, default=8, help="tolerance distance for sync group")
    parser.add_argument('--alpha_max', type=int, default=0.8, help='max weight param used in FedAsync aggregation')
    parser.add_argument('--alpha_min', type=int, default=0.2, help='min weight param used in FedAsync aggregation')
    parser.add_argument('--n_users', type=int, default=500, help="number of users")
    parser.add_argument('--n_sample', type=int, default=3000, help="sample dataset for each user")
    

    parser.add_argument('--use_predict', action='store_true', help='whether to use predict index')
    parser.add_argument('--predict_model', type=str, default='Linear', help="which model used for trajectory prediction")

    parser.add_argument('--use_softlabel', action='store_true', help='whether to use soft label')
    parser.add_argument('--topk', type=int, default=5, help='saved top k logits for knowledge distillation')

    parser.add_argument('--use_mp', action='store_true', help='whether to use multi-process')
    parser.add_argument('--fixed_seed', action='store_true', help='whether to set random seed')
    args = parser.parse_args()
    return args
