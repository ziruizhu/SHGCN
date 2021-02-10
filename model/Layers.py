import torch


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse.FloatTensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.sparse.mm(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_idx = a._indices()[0, :]
            b_idx = a._indices()[1, :]
            grad_select = grad_output[grad_idx]
            b_select = b[b_idx]
            grad_values = torch.sum(grad_select.mul(b_select), dim=1)

            # grad_a_dense = grad_output.matmul(b.t())
            # edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            # grad_values = grad_a_dense.view(-1)[edge_idx]

        if ctx.needs_input_grad[3]:
            grad_b = torch.sparse.mm(a.t(), grad_output)
        return None, grad_values, None, grad_b


def SpecialSpmm(indices, values, shape, b):
    return SpecialSpmmFunction.apply(indices, values, shape, b)
