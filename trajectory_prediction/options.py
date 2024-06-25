import argparse


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default='cuda', metavar='N',
                        help='device for training (default: cuda)')
    parser.add_argument('--stride', type=int, default=10, metavar='N',
                        help='stride (default: 12)')
    parser.add_argument('--Lag', type=int, default=40, metavar='N',
                        help='Lag (default: 12)')
    parser.add_argument('--Horizon', type=int, default=20, metavar='N',
                        help='Horizon (default: 12)')
    parser.add_argument('--kernel_size', type=int, default=3, metavar='N',
                        help='kernel size (default: 10)')
    parser.add_argument('--individual', type=int, default=1, metavar='N',
                        help='individual (default: 1)')
    parser.add_argument('--enc_in', type=int, default=2, metavar='N',
                        help='number of encoder input (default: 1)')
    parser.add_argument('--model_name', type=str, default='Linear', metavar='N',
                        help='which model to use for prediction')
    args = parser.parse_args()

    return args