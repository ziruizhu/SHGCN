import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class TrainSet(data.Dataset):
    def __init__(self, num_item, num_negative, train_data, train_mat):
        super().__init__()
        self.num_item = num_item
        self.num_ng = num_negative
        self.train_data = train_data
        self.train_mat = train_mat

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        user, pos_item = self.train_data[idx]
        all_item = [pos_item]
        while len(all_item) <= self.num_ng:
            j = np.random.randint(self.num_item)
            while (user, j) in self.train_mat or j in all_item:
                j = np.random.randint(self.num_item)
            all_item.append(j)
        return torch.LongTensor([user]), torch.LongTensor(all_item)


class TestSet(data.Dataset):
    def __init__(self, num_item, num_negative, test_data, train_mat, seed=123):
        super().__init__()
        self.num_item = num_item
        self.num_ng = num_negative
        self.test_data = test_data
        self.train_mat = train_mat
        np.random.seed(seed)
        self._ng_sample()

    def _ng_sample(self):
        self.users = []
        self.items = []
        for x in self.test_data:
            u, i = x[0], x[1]
            all_item = [i]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat or j == i:
                    j = np.random.randint(self.num_item)
                all_item.append(j)
            self.users.append([u])
            self.items.append(all_item)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        return torch.LongTensor(user), torch.LongTensor(item)


def load_all(arg):
    train_data = pd.read_csv(
        arg.data_path + '/' + arg.dataset + '/data.train',
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    train_data = train_data.values.tolist()

    val_data = pd.read_csv(
        arg.data_path + '/' + arg.dataset + '/data.val',
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    val_data = val_data.values.tolist()

    test_data = pd.read_csv(
        arg.data_path + '/' + arg.dataset + '/data.test',
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    test_data = test_data.values.tolist()

    share_data = pd.read_csv(
        arg.data_path + '/' + arg.dataset + '/social.share',
        sep=',', header=None, names=['user', 'friend', 'item'],
        usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

    share_data = share_data.values.tolist()
    return train_data, val_data, test_data, share_data


def manage(arg, train_data, val_data, test_data, share_data):
    train_mat = sp.dok_matrix((arg.num_user, arg.num_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    for x in val_data:
        train_mat[x[0], x[1]] = 1.0

    share_info = defaultdict(list)
    for each in share_data:
        share_info[(each[0], each[1])].append(each[2])

    CE_adj = sp.dok_matrix((len(share_data), arg.num_user + arg.num_item), dtype=np.float32)
    RC_adj = sp.dok_matrix((len(share_info), len(share_data)), dtype=np.float32)
    UU_idx = np.zeros((2, len(share_info)), dtype=np.int64)
    c_idx = 0
    for i, pair in enumerate(share_info.items()):
        users, items = pair
        UU_idx[0, i] = users[0]
        UU_idx[1, i] = users[1]
        for entry in items:
            CE_adj[c_idx, users[0]] = 1.0
            CE_adj[c_idx, users[1]] = 1.0
            CE_adj[c_idx, entry + arg.num_user] = 1.0
            RC_adj[i, c_idx] = 1.0
            c_idx += 1

    EC_adj = sparse_mx_to_torch_sparse_tensor(normalize(CE_adj.T))
    CE_adj = sparse_mx_to_torch_sparse_tensor(normalize(CE_adj))
    RC_adj = sparse_mx_to_torch_sparse_tensor(normalize(RC_adj))
    UU_idx = torch.from_numpy(UU_idx)

    data_dict = dict()
    data_dict['train_data'] = train_data
    data_dict['val_data'] = val_data
    data_dict['test_data'] = test_data
    data_dict['train_mat'] = train_mat
    data_dict['EC_adj'] = EC_adj
    data_dict['CE_adj'] = CE_adj
    data_dict['RC_adj'] = RC_adj
    data_dict['UU_idx'] = UU_idx
    return data_dict


def prepare_data(args):
    train_data, val_data, test_data, share_data = load_all(args)
    data_dict = manage(args, train_data, val_data, test_data, share_data)
    data_dict = getloader(args, data_dict)
    return data_dict


def getloader(args, data_dict):
    train_data = data_dict['train_data']
    val_data = data_dict['val_data']
    test_data = data_dict['test_data']
    train_mat = data_dict['train_mat']

    train_set = TrainSet(args.num_item, args.num_negative, train_data, train_mat)
    val_set = TestSet(args.num_item, args.num_negative_eval, val_data, train_mat, seed=213)
    test_set = TestSet(args.num_item, args.num_negative_eval, test_data, train_mat)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_eval, shuffle=False, num_workers=4)

    data_dict['train_loader'] = train_loader
    data_dict['val_loader'] = val_loader
    data_dict['test_loader'] = test_loader
    return data_dict


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    np.seterr(divide='ignore')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    np.seterr(divide='ignore')
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
