import torch
import torch.optim as optim
# from torch.optim import lr_scheduler
from time import time
import copy
from utility.evaluate_one import Evaluate
from collections import defaultdict


class TrainLogger:
    def __init__(self, threshold, maxdown, benchmark):
        """
        :param threshold: metrics / best_metrics > 1 + threshold, set the early stop flag true
        :param maxdown: Maximum number of epochs allowed where the metrics is going down
        :param benchmark: `p` | `map` | `ndcg` | `mrr` | `hit` | `r` | `f`
        """
        self.shock = maxdown * 5
        self.threshold = threshold
        self.maxdown = maxdown
        self.benchmark = benchmark
        self.best_metrics = defaultdict(lambda: 0)
        self.test_metrics = None
        self.best_weights = None
        self.best_epoch = -1
        self.down = 0
        self.last_metric = defaultdict(lambda: 0)

    def log(self, metrics, test_metrics, epoch, state_dict):
        if self.best_metrics[self.benchmark] > 0 and \
                metrics[self.benchmark] / self.best_metrics[self.benchmark] < self.threshold:
            return True
        if epoch - self.best_epoch > self.shock:
            return True
        if metrics[self.benchmark] > self.last_metric[self.benchmark]:
            self.down = 0
        else:
            self.down += 1
        self.last_metric = metrics
        if metrics[self.benchmark] > self.best_metrics[self.benchmark]:
            self.best_metrics = copy.deepcopy(metrics)
            self.test_metrics = copy.deepcopy(test_metrics)
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(state_dict)
        return self.down >= self.maxdown


def train_all(model, loss_func, optimizer, args, data_dict, p, test=False):
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    K = eval(args.topK)
    e = Evaluate()
    bestlogger = TrainLogger(args.alpha, args.maxdown, 'ndcg@%d' % K[-1])
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(args.epoch):
        t1 = time()
        train_loss = train(model, args, train_loader, loss_func, optimizer)
        # scheduler.step()
        t2 = time()
        val_loss, val_quota = eva(model, args, val_loader, loss_func, e)
        if test:
            test_loss, test_quota = eva(model, args, test_loader, loss_func, e)
        else:
            test_loss = 0
            test_quota = defaultdict(lambda: 0)
        t3 = time()
        early_stop = bestlogger.log(val_quota, test_quota, epoch, model.state_dict())
        printMetrics(epoch, train_loss, val_loss, test_loss, t1, t2, t3, K, p, args.print, early_stop, val_quota,
                     test_quota)
        if early_stop:
            break

    if bestlogger.best_weights is not None:
        model.load_state_dict(bestlogger.best_weights)

    return bestlogger.best_metrics, bestlogger.best_epoch, bestlogger.test_metrics


def train(model, arg, train_loader, loss_func, optimizer):
    tmp_loss = 0.
    model.train()
    num_batch = len(train_loader)
    device = arg.device
    for users, POIs in train_loader:
        users, POIs = users.to(device), POIs.to(device)
        optimizer.zero_grad()
        pred, reg_loss = model(users, POIs)
        loss = loss_func(pred, reg_loss, arg.batch_size)
        loss.backward()
        tmp_loss += loss.item()
        if arg.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), arg.clip)
        optimizer.step()
    return tmp_loss / num_batch


def eva(model, arg, test_loader, loss_func, e):
    test_loss = 0
    num_batch = len(test_loader)
    device = arg.device
    all_pred = []
    with torch.no_grad():
        model.eval()
        for users, POIs in test_loader:
            users, POIs = users.to(device), POIs.to(device)
            pred, reg_loss = model(users, POIs)
            if arg.p:
                print(reg_loss)
            loss = loss_func(pred, reg_loss, arg.batch_size_eval)
            test_loss += loss
            all_pred += [pred]
    all_pred = torch.cat(all_pred, dim=0)
    if arg.p:
        print(all_pred[0:5, 0:20])
    metrics = e.evaluate(all_pred, eval(arg.topK))
    return test_loss / num_batch, metrics


def printMetrics(epoch, train_loss, val_loss, test_loss, t1, t2, t3, K, p, interval, early_stop, val_quota, test_quota):
    metrics = []
    for k in K:
        metrics.append('recall@%d' % k)
        metrics.append('ndcg@%d' % k)

    perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f], ' % (
        epoch, t2 - t1, t3 - t2, train_loss)
    if len(K) > 1:
        perf_str = perf_str + '\n'
    perf_str = perf_str + 'val==[%.5f], ' % val_loss

    for metric in metrics:
        perf_str = perf_str + metric + '=%.4f, ' % val_quota[metric]
    if len(K) > 1:
        perf_str = perf_str + '\n'

    perf_str = perf_str + 'test==[%.5f], ' % test_loss

    for metric in metrics:
        perf_str = perf_str + metric + '=%.4f, ' % test_quota[metric]

    if early_stop and p:
        print(perf_str)
        print('Overfitting! Early Stop at epoch %d' % epoch)
    elif p and epoch % interval == 0:
        print(perf_str)
