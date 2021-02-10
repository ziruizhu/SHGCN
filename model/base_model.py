import torch
import torch.nn as nn


class Model(nn.Module):
    '''
    base class for all MF-based model
    packing embedding initialization, embedding choosing in forward

    NEED IMPLEMENT:
    - `propagate`: all raw embeddings -> processed embeddings(user/POI)
    - `predict`: processed embeddings of targets(users/POIs inputs) -> scores

    OPTIONAL:
    - `regularize`: processed embeddings of targets(users/POIs inputs) -> extra loss(default: L2)
    '''

    def __init__(self, args):
        super().__init__()
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.lam = args.lam
        self.emb_dim = args.emb_dim

        self.u_embeds = nn.Parameter(
            torch.FloatTensor(self.num_user, self.emb_dim), requires_grad=True)
        self.i_embeds = nn.Parameter(
            torch.FloatTensor(self.num_item, self.emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.u_embeds)
        nn.init.xavier_normal_(self.i_embeds)

    def propagate(self, *args, **kwargs):
        '''
        raw embeddings -> embeddings for predicting
        return (user's,POI's)
        '''
        raise NotImplementedError

    def predict(self, users_feature, POIs_feature, *args, **kwargs):
        return torch.sum(users_feature * POIs_feature, dim=2)

    def regularize(self, users_feature, POIs_feature, *args, **kwargs):
        '''
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        '''
        loss = self.lam * \
               ((users_feature ** 2).sum() + (POIs_feature ** 2).sum())
        return loss

    def forward(self, users, POIs):
        users_feature, POIs_feature = self.propagate()
        POIs_embedding = POIs_feature[POIs]
        users_embedding = users_feature[users].expand(
            - 1, POIs.shape[1], -1)
        pred = self.predict(users_embedding, POIs_embedding)
        loss = self.regularize(users_embedding, POIs_embedding)
        return pred, loss
