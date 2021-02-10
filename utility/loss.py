import torch
import torch.nn as nn


class _Loss(nn.Module):
    def __init__(self, reduction='mean'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        super().__init__()
        assert (reduction == 'mean' or reduction == 'sum' or reduction == 'none')
        self.reduction = reduction


class BPRLoss(_Loss):
    def __init__(self, reduction='mean'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        super().__init__(reduction)

    def forward(self, pred, reg_loss, batch_size=None):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive pois, column 1 must be the negative.
        '''
        neg_pred = pred[:, 1:]
        pos_pred = pred[:, 0].unsqueeze(-1).expand(-1, neg_pred.shape[1])
        loss = - torch.log(torch.sigmoid(pos_pred - neg_pred))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise ValueError("reduction must be 'mean' | 'sum'")
        if batch_size is not None and self.reduction == 'mean':
            loss += reg_loss / batch_size
        else:
            loss += reg_loss
        return loss


class LogLoss(_Loss):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        super().__init__(reduction)
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, reg_loss, batch_size=None):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive pois, column 1 must be the negative.
        '''
        tag = torch.zeros_like(pred, device=pred.device)
        tag[:, 0] = 1
        loss = self.criterion(pred.view(-1), tag.view(-1))
        if batch_size is not None and self.reduction == 'mean':
            loss += reg_loss / batch_size
        else:
            loss += reg_loss
        return loss
