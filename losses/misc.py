import torch
import torch.nn.functional as F

def __shape_assertion(y_masks_batch, pred_logits_batch):
    assert len(y_masks_batch.shape) == 4
    assert len(pred_logits_batch.shape) == 5

    assert y_masks_batch.shape[0] == pred_logits_batch.shape[0]
    assert y_masks_batch.shape[1] == pred_logits_batch.shape[2]
    assert y_masks_batch.shape[2] == pred_logits_batch.shape[3]
    assert y_masks_batch.shape[3] == pred_logits_batch.shape[4]

def __reduction_metric(metric_batch, reduction):
    if reduction == 'mean':
        loss = torch.mean(1 - metric_batch, axis=0)
    elif reduction == 'sum':
        loss = torch.sum(1 - metric_batch, axis=0)
    else:
        raise RuntimeError()

    return loss

def __reduction_loss(loss_batch, reduction):
    if reduction == 'mean':
        loss = torch.mean(loss_batch, axis=0)
    elif reduction == 'sum':
        loss = torch.sum(loss_batch, axis=0)
    else:
        raise RuntimeError()

    return loss

def __logits_to_probs(logits, nclasses, activation):
    if nclasses == 1:
        probs = torch.sigmoid(logits)
    else:
        if activation == 'softmax':
            probs = F.softmax(logits, dim=1)
        elif activation == 'sigmoid':
            probs = torch.sigmoid(logits)
        else:
            raise ValueError()

    return probs
