import torch
import torch.nn.functional as F

from functools import partial

from losses.misc import __reduction_loss
from losses.checkers import check_arguments, _average_value_checker, _reduction_value_checker, _3d_shape_asserter

@partial(check_arguments, checkers=[_average_value_checker, _reduction_value_checker, _3d_shape_asserter])
def cross_entropy_with_logits_loss( y_masks_batch,
                                    pred_logits_batch,
                                    *,
                                    average='binary',
                                    activation='softmax',
                                    reduction='mean',
                                    parameters=dict(),
                                    hyperparameters=dict() ):
    """ y_masks_batch size [batch_size, depth, height, width]
        pred_logits_batch [batch_size, nclasses, depth, height, width]
    """

    if parameters.get('nclasses', None) is not None:
        assert pred_logits_batch.shape[1] == parameters.get('nclasses', None)
    else:
        nclasses = pred_logits_batch.shape[1]

    if nclasses == 1:
        pred_logits_batch = pred_logits_batch.squeeze(1)
        ce_loss_batch = F.binary_cross_entropy_with_logits(pred_logits_batch, y_masks_batch, reduction='none')
    else:
        if activation == 'softmax':
            ce_loss_batch = F.cross_entropy(pred_logits_batch, y_masks_batch, reduction='none')
        elif activation == 'sigmoid':
            raise NotImplemented()
        else:
            raise ValueError()

    ce_loss_batch = torch.mean(ce_loss_batch, dim=(1, 2, 3))
    ce_loss = __reduction_loss(ce_loss_batch, reduction)

    return ce_loss

@partial(check_arguments, checkers=[_average_value_checker, _reduction_value_checker, _3d_shape_asserter])
def focal_with_logits_loss( y_masks_batch,
                            pred_logits_batch,
                            *,
                            average='binary',
                            activation='softmax',
                            reduction='mean',
                            parameters=dict(),
                            hyperparameters=dict(gamma=1) ):
    """ y_masks_batch size [batch_size, depth, height, width]
        pred_logits_batch [batch_size, nclasses, depth, height, width]
    """

    if parameters.get('nclasses', None) is not None:
        assert pred_logits_batch.shape[1] == nclasses
    else:
        nclasses = pred_logits_batch.shape[1]

    if nclasses == 1:
        raise NotImplemented()
    else:
        if activation == 'softmax':
            ce_loss_batch = F.cross_entropy(pred_logits_batch, y_masks_batch, reduction='none')
            pred_probs_batch = torch.exp(-ce_loss_batch)

            fl_loss_batch = ce_loss_batch * (1 - pred_probs_batch)**hyperparameters['gamma']
        elif activation == 'sigmoid':
            raise NotImplemented()
        else:
            raise ValueError()

    fl_loss_batch = torch.mean(fl_loss_batch, dim=(1, 2, 3))
    fl_loss = __reduction_loss(fl_loss_batch, reduction)

    return fl_loss
