import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")

    parser.add_argument('--data_path', nargs='?', default='./data',
                        help='Input data path.')

    parser.add_argument('--num_user', type=int, default=3773,
                        help='number of users.')
    parser.add_argument('--num_item', type=int, default=4544,
                        help='number of items.')
    parser.add_argument('--dataset', nargs='?', default='Beidian',
                        help='Choose a dataset from {Beidian, Beibei}')

    parser.add_argument('--model', nargs='?', default='SHGCN',
                        help='Name of the model')
    parser.add_argument('--loss', nargs='?', default='BPRLoss',
                        help='choose from {BPRLoss, LogLoss}')
    parser.add_argument('--topK', nargs='?', default='[10]',
                        help='TopK')
    parser.add_argument('--gpu', nargs='?', default='0',
                        help='GPU id')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Flag of using weights.')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='Flag of saving weights.')
    parser.add_argument('--weight_name', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='negative sampling.')
    parser.add_argument('--batch_size_eval', type=int, default=4096,
                        help='number of negative items when evaluating.')

    parser.add_argument('--emb_dim', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of layers.')
    parser.add_argument('--num_negative', type=int, default=8,
                        help='number of negative items when training.')
    parser.add_argument('--num_negative_eval', type=int, default=100,
                        help='number of negative items when evaluating.')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--fail', type=float, default=1,
                        help='attenuation.')
    parser.add_argument('--lam', type=float, default=1e-5,
                        help='lambda.')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip.')

    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Threshold for early stop.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Message dropout for GNN-based model.')
    parser.add_argument('--maxdown', type=int, default=20,
                        help='Max epoch allowed where metrics drop.')

    parser.add_argument('--print', type=int, default=1,
                        help='Interval of print.')
    parser.add_argument('--p', type=int, default=0,
                        help='Flag of print.')

    return parser.parse_args()
