import torch


class Evaluate:
    '''
        designed for leave-one-out test-set
    '''

    def recall(self, rlist):
        hit = torch.sum(rlist)
        return hit / rlist.shape[0]

    def dcg(self, rlist):
        dcg_list = torch.arange(2, rlist.shape[1] + 2, dtype=torch.float32, device=rlist.device)
        dcg_list = torch.log2(dcg_list).pow(-1)
        self.rdcg = rlist * dcg_list

    def ndcg(self, k):
        rdcg = self.rdcg[:, :k]
        ndcg = torch.sum(rdcg, dim=-1)
        return torch.mean(ndcg)

    def evaluate(self, pred, topK):
        '''
        topK : [1, 3, 5, 10]
        pred : shape = batch_size * ( 1 + num_neg)
        '''
        metrics = dict()
        pred[:, 0] = pred[:, 0] - torch.finfo(pred.dtype).tiny

        topK = sorted(topK, reverse=True)
        _, rlist = torch.topk(pred, topK[0])
        rlist = (rlist == 0).double()

        self.dcg(rlist)
        for k in topK:
            rlist = rlist[:, :k]
            recall = self.recall(rlist)
            ndcg = self.ndcg(k)
            metrics['ndcg@%d' % k] = ndcg.item()
            metrics['recall@%d' % k] = recall.item()
        return metrics
