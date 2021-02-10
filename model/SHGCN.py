import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import Model
from .Layers import SpecialSpmm


class ATT_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb_dim = args.emb_dim
        self.num_user = args.num_user
        self.C_transformer = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.R_transformer = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.rating_func = nn.Linear(self.emb_dim, 1, bias=False)

    def forward(self, E, CE_adj, EC_adj, RC_adj, UU_idx):
        c_emb = self.C_transformer(torch.sparse.mm(CE_adj, E))
        r_emb = self.R_transformer(torch.sparse.mm(RC_adj, c_emb))

        R_rating = self.rating_func(r_emb).squeeze(dim=1)
        uu_message = SpecialSpmm(UU_idx, R_rating, torch.Size([self.num_user, self.num_user]), E[:self.num_user])
        uu_message = F.leaky_relu(uu_message)
        ec_message = F.leaky_relu(torch.sparse.mm(EC_adj, c_emb))
        E = ec_message
        E[:self.num_user] = E[:self.num_user] + uu_message
        return E


class SHGCN(Model):
    def __init__(self, args, data_dict):
        super().__init__(args)
        self.num_layer = args.num_layer
        self.fail = args.fail
        self.setMat(args, data_dict)
        self.graph_conv = nn.ModuleList([])
        self.dropout = nn.Dropout(args.dropout, True)
        for i in range(self.num_layer):
            self.graph_conv.append(ATT_layer(args))

    def setMat(self, args, data_dict):
        self.CE_adj = data_dict['CE_adj'].to(args.device)
        self.EC_adj = data_dict['EC_adj'].to(args.device)
        self.RC_adj = data_dict['RC_adj'].to(args.device)
        self.UU_idx = data_dict['UU_idx'].to(args.device)

    def propagate(self):
        all_emb = torch.cat((self.u_embeds, self.i_embeds), dim=0)
        cat_emb = [all_emb]
        fail = 1.0
        for layer in range(self.num_layer):
            message = self.graph_conv[layer](all_emb, self.CE_adj, self.EC_adj,
                                             self.RC_adj, self.UU_idx)
            all_emb = all_emb + fail * message
            all_emb = self.dropout(all_emb)

            norm_emb = F.normalize(all_emb, p=2, dim=1)
            cat_emb += [norm_emb]
            fail *= self.fail
        cat_emb = torch.cat(cat_emb, dim=1)
        return cat_emb[:self.num_user], cat_emb[self.num_user:]
