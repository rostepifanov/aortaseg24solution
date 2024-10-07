
def __get_args_dict(func, args, kwargs):
    args_count = func.__code__.co_argcount
    args_names = func.__code__.co_varnames[:args_count]

    if func.__defaults__:
        args_defaults_count = len(func.__defaults__)
        args_defaults_names = args_names[-args_defaults_count:]

        args_dict = { **dict(zip(args_defaults_names, func.__defaults__)),
                      **dict(zip(args_names, args)) }
    else:
        args_dict = dict(zip(args_names, args))

    if func.__code__.co_kwonlyargcount:
        kwargs_dict = {**func.__kwdefaults__, **kwargs}
    else:
        kwargs_dict = kwargs

    return {**args_dict, **kwargs_dict}


def check_arguments(func, checkers=[]):
    def wrapper_func(*args, **kwargs):
        for checker in checkers:
            checker(__get_args_dict(func, args, kwargs))

        return func(*args, **kwargs)
    return wrapper_func

def _average_value_checker(args):
    if args['average'] not in {'binary', 'macro'}:
        raise ValueError('average value is incorrect')

def _reduction_value_checker(args):
    if args['reduction'] not in {'sum', 'mean'}:
        raise ValueError('reduction value is incorrect')

def _3d_shape_asserter(args):
    if not (len(args['y_masks_batch'].shape) == 4):
        raise RuntimeError(f"y_masks_batch length is {len(args['y_masks_batch'].shape)} not 4")

    if not (len(args['pred_logits_batch'].shape) == 5):
        raise RuntimeError(f"pred_logits_batch length is {len(args['pred_logits_batch'].shape)} not 5")

    if not (args['y_masks_batch'].shape[0] == args['pred_logits_batch'].shape[0]):
        raise RuntimeError(f"y_masks_batch batch size is {args['y_masks_batch'].shape[0]}, "
                           f"but pred_logits_batch batch size is {args['pred_logits_batch'].shape[0]}")

    assert args['y_masks_batch'].shape[1] == args['pred_logits_batch'].shape[2]
    assert args['y_masks_batch'].shape[2] == args['pred_logits_batch'].shape[3]
    assert args['y_masks_batch'].shape[3] == args['pred_logits_batch'].shape[4]
