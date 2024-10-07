import torch

from functools import partial

from losses.interface import SegmentationLossTemplate
from losses.checkers import check_arguments, _average_value_checker, _reduction_value_checker, _3d_shape_asserter

def _confusion_parameters_asserter(args):
    if args['average'] == 'binary':
        if args['activation'] not in {'sigmoid', 'softmax'}:
            raise RuntimeError(f'Incorrect combination of average {args["average"]} and activation {args["activation"]}')

        ALLOWED_PARAMETER_NAMES = { 'nclasses' }
        passed_parameter_names = set(args['parameters'].keys())

        if len(passed_parameter_names - ALLOWED_PARAMETER_NAMES) > 0:
            raise RuntimeError(f'Unexpected parameters are passed: {" ".join(passed_parameter_names - ALLOWED_PARAMETER_NAMES)}')
    elif args['average'] == 'macro':
        if args['activation'] not in {'softmax'}:
            raise RuntimeError(f'Incorrect combination of average {args["average"]} and activation {args["activation"]}')

        ALLOWED_PARAMETER_NAMES = { 'nclasses', 'averaged_classes' }

        passed_parameter_names = set(args['parameters'].keys())

        if len(passed_parameter_names - ALLOWED_PARAMETER_NAMES) > 0:
            raise RuntimeError(f'Unexpected parameters are passed: {" ".join(passed_parameter_names - ALLOWED_PARAMETER_NAMES)}')

class ConfusionLossTemplate(SegmentationLossTemplate):
    def __init__(self, metric):
        super().__init__('metric')

        self.__metric = metric

    def _get_confusions(self, y_masks_batch, pred_probs_batch):
        tmp = y_masks_batch * pred_probs_batch
        
        tp = torch.sum(tmp, dim=(1, 2, 3))
        fp = torch.sum(pred_probs_batch-tmp, dim=(1, 2, 3))
        fn = torch.sum(y_masks_batch-tmp, dim=(1, 2, 3))
        tn = torch.sum(1-y_masks_batch-pred_probs_batch+tmp, dim=(1, 2, 3))

        return tp, fp, fn, tn

    @partial(check_arguments, checkers=[ _average_value_checker,
                                         _reduction_value_checker,
                                         _confusion_parameters_asserter,
                                         _3d_shape_asserter ])
    def __call__( self,
                  y_masks_batch,
                  pred_logits_batch,
                  *,
                  average='binary',
                  activation='sigmoid',
                  reduction='mean',
                  parameters=dict(),
                  hyperparameters=dict() ):
        """ y_masks_batch size [batch_size, height, width]
            pred_logits_batch [batch_size, nclasses, height, width]
        """

        nclasses = self._assess_nclasses( pred_logits_batch, parameters.get('nclasses', None) )
        pred_activated_logits_batch = self._activate_logits( pred_logits_batch, nclasses, activation )

        if average == 'binary':
            if activation == 'sigmoid':
                if nclasses != 1: raise RuntimeError()
                pred_activated_logits_batch = pred_activated_logits_batch.squeeze()
            elif activation == 'softmax':
                if nclasses != 2: raise RuntimeError()
                pred_activated_logits_batch = pred_activated_logits_batch[:, 1]
            else:
                raise RuntimeError()

            metric_batch = self.__metric( *self._get_confusions(y_masks_batch, pred_activated_logits_batch),
                                          **hyperparameters )
        elif average == 'macro':
            if activation == 'sigmoid':
                raise NotImplementedError()
            elif activation == 'softmax':
                assert nclasses > 1

                if parameters.get('averaged_classes', None) is None:
                    averaged_classes = range(1, nclasses)

                metric_batch_stack = tuple()

                for class_ in averaged_classes:
                    class_y_masks_batch = (y_masks_batch == class_).long()
                    class_pred_activated_logits_batch = pred_activated_logits_batch[:, class_]

                    class_metric_batch = self.__metric(
                        *self._get_confusions(class_y_masks_batch, class_pred_activated_logits_batch),
                        **hyperparameters
                    )

                    metric_batch_stack = (*metric_batch_stack, class_metric_batch)

                metric_batch_stack = torch.stack(metric_batch_stack, dim=-1)
                metric_batch = torch.mean(metric_batch_stack, dim=-1)
            else:
                raise RuntimeError()
        else:
            raise ValueError()

        loss = self._reduce(metric_batch, reduction)

        return loss

def dice(tp, fp, fn, tn, *, smooth=1):
    """ NOTE
        binary is common used implementation
        macro is described in https://ieeexplore.ieee.org/document/9433991
    """

    numerator = 2 * tp + smooth
    denominator = tp + fp + tp + fn + smooth

    return numerator / denominator

dice_with_logits_loss = ConfusionLossTemplate(dice)

def matthews(tp, fp, fn, tn, *, smooth=1):
    """ NOTE
        binary is described in https://www2.cs.sfu.ca/~hamarneh/ecopy/isbi2021.pdf
    """

    numerator = tp * tn - fp * fn + smooth
    denominator = ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) + smooth)**0.5

    return numerator / denominator

matthews_with_logits_loss = ConfusionLossTemplate(matthews)