import platform
import os
import torch
import torch.optim as optim
import numpy as np
import random
from numpy.random import randint
from utility.parser import parse_args
from model.SHGCN import SHGCN
from utility.utils import train_all, eva
from utility.data_utils import prepare_data
from utility.loss import BPRLoss, LogLoss
from utility.evaluate_one import Evaluate


def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def main(args, data_dict, p=False, test=False, seed=321):
    if seed is not None:
        setup_seed(seed)
    if p:
        print('Creating Model...')
    model = eval(args.model)
    model = model(args, data_dict)
    model.to(args.device)

    dirs = './weight/'
    if args.pretrain:
        weight_path = dirs + args.weight_name
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print('Weight loaded!')
    K = eval(args.topK)
    loss = eval(args.loss)
    criterion = loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.epoch > 0:
        if p:
            print('Training...')
        metrics, best_epoch, test_metrics = train_all(model, criterion, optimizer, args, data_dict, p, test=test)
        metrics['best_epoch'] = best_epoch
        if p:
            printResult('Best result in val:', metrics, K)
            printResult('Result in test:', test_metrics, K)
        if args.save_flag:

            if not os.path.exists(dirs):
                os.makedirs(dirs)
            save_path = dirs + args.model + '_' + args.dataset + '_ndcg_' + str(
                round(metrics['ndcg@%d' % K[-1]], 6)) + '_layer%d' % args.num_layer + '_lr%.5f_lam%.7f' % (
                            args.lr, args.lam) + '.pth'
            torch.save(model.state_dict(), save_path)
    else:
        e = Evaluate()
        val_loss, metrics = eva(model, args, data_dict['val_loader'], criterion, e)
        test_loss, test_metrics = eva(model, args, data_dict['test_loader'], criterion, e)
        if p:
            printResult('Best result in val:', metrics, K)
            printResult('Result in test:', test_metrics, K)

    return metrics, test_metrics


def printResult(perf_str, quota, K):
    metrics = []
    for k in K:
        metrics.append('recall@%d' % k)
        metrics.append('ndcg@%d' % k)

    for metric in metrics:
        perf_str = perf_str + metric + '=%.4f, ' % quota[metric]
    print(perf_str)


if __name__ == '__main__':
    args = parse_args()
    print("load data...")
    if platform.system() == 'Linux':
        import setproctitle

        setproctitle.setproctitle(args.model + "_%d" % randint(1000))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from utility.data_arr import arr

    args = arr(args)

    data_dict = prepare_data(args)
    main(args, data_dict, True, True)
