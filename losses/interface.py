import torch, torch.nn.functional as F

class LossTemplate(object):
    def __init__(self, loss_type):
        self._loss_type = loss_type

    def _assess_nclasses(self, pred_logits_batch, nclasses):
        if nclasses is not None:
            if pred_logits_batch.shape[1] != nclasses:
                raise RuntimeError( f'Expected nclasses - {nclasses},'
                                    f'logits nclass - {pred_logits_batch.shape[1]}' )
        else:
            nclasses = pred_logits_batch.shape[1]

        return nclasses

    @staticmethod
    def __reduction_metric(metric_batch, reduction):
        if reduction == 'mean':
            loss = torch.mean(1 - metric_batch, axis=0)
        elif reduction == 'sum':
            loss = torch.sum(1 - metric_batch, axis=0)
        elif reduction == 'none':
            loss = 1 - metric_batch
        else:
            raise ValueError()

        return loss

    @staticmethod
    def __reduction_loss(loss_batch, reduction):
        if reduction == 'mean':
            loss = torch.mean(loss_batch, axis=0)
        elif reduction == 'sum':
            loss = torch.sum(loss_batch, axis=0)
        elif reduction == 'none':
            loss = loss_batch
        else:
            raise ValueError()

        return loss

    def _reduce(self, type_batch, reduction):
        if self._loss_type == 'metric':
            return self.__reduction_metric(type_batch, reduction)
        elif self._loss_type == 'loss':
            return self.__reduction_loss(type_batch, reduction)
        else:
            raise ValueError()

class LossTemplateND(LossTemplate):
    def _activate_logits(self, pred_logits_batch, nclasses, activation):
        if activation == 'sigmoid':
            pred_activated_logits_batch = torch.sigmoid(pred_logits_batch)
        elif activation == 'softmax':
            if nclasses < 2: raise RuntimeError()
            pred_activated_logits_batch = F.softmax(pred_logits_batch, dim=1)
        elif activation == 'none':
            pred_activated_logits_batch = pred_logits_batch
        else:
            raise ValueError()

        return pred_activated_logits_batch

class SegmentationLossTemplate(LossTemplateND):
    def __init__(self, loss_type):
        super().__init__(loss_type)
