import os
import sys
import timm
import copy
import time
import yaml
import click
import torch, torch.nn.functional as F
import random
import imgaug
import numpy as np
import statistics as s
import voxelmentations as V
import torch.optim.swa_utils as tsu

from tqdm import tqdm
from pathlib import Path
from clearml import Task
from collections import defaultdict
from torch.utils.data import DataLoader
from monai.transforms import AsDiscrete
from nnspt.segmentation.unet import Unet
from nnspt.segmentation.unetpp import Unetpp
#from monai.metrics import DiceMetric, SurfaceDiceMetric

from convertors import convert_inplace, LayerConvertorSm, LayerConvertorNNSPT
from losses.threedim import matthews_with_logits_loss, dice_with_logits_loss, focal_with_logits_loss
from dataset import aortaIMMhaDataset, VoxelWeightedSampler,VoxelRandomSampler, VoxelSequentialSampler

config = dict()

def init_determenistic(seed=1996, precision=10):
    """ NOTE options

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determenistic = True

        may lead to numerical unstability
    """
    random.seed(seed)
    np.random.seed(seed)
    imgaug.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determenistic = True
    torch.backends.cudnn.enabled = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.use_deterministic_algorithms(True)

    os.environ['PYTHONHASHSEED'] = str(seed)

def init_kwargs(config, kwargs):
    for key, value in kwargs.items():
        if key.upper().endswith('PATH'):
            if value is not None:
                config[key.upper()] = Path(value)
            else:
                raise RuntimeError(f'Path option {key} is None')
        else:
            config[key.upper()] = value

def init_device(config):
    if torch.cuda.is_available():
        config['DEVICE'] = torch.device('cuda')
    else:
        config['DEVICE'] = torch.device('cpu')

def init_verboser(config):
    if config['VERBOSE']:
        config['VERBOSER'] = lambda x, **lkwargs: tqdm(x, **lkwargs)
    else:
        config['VERBOSER'] = lambda x, **lkwargs: x

def load_yaml_config(path):
    assert path.is_file()

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data

def init_options(config):
    for key in list(config.keys()):
        if key.endswith('OPTIONS_PATH'):
            option_key = key[:-5]
            config[option_key] = load_yaml_config(config[key])

def init_run_command(config):
    config['SCRIPT'] = ' '.join(sys.argv)

def init_timestamp(config):
    config['PREFIX'] = time.strftime("%d-%m-%y:%H-%M_", time.gmtime())

def init_global_config(**kwargs):
    init_timestamp(config)
    init_run_command(config)
    init_kwargs(config, kwargs)
    init_device(config)
    init_verboser(config)
    init_options(config)

def load_data():
    data = { }

    augs = None

    data = {
        'train': aortaIMMhaDataset(
            config['DATAPATH'],
            # [ "001", "002", "003", "004", "005",
            #   "006", "007", "008", "009", "010",
            #   "011", "012", "013", "015", "016",
            #   "017", "018", "019", "020", "021",
            #   "022", "023", "024", "025", "026",
            #   "027", "028", "029", "030", "031",
            #   "032", "033", "034", "035", "036",
            #   "038", "039", "040", "041", "042", ],
              # "044", "045", "046", "047", "048",
              # "049", "050", "051", "052", "053", ],
            [ "001", "002", "003", "004", "005",
              "006", "007", "008", "009", "010",
              "011", "012", "013", "015", "016",
              "017", "018", "019", "020", "021",
              "022", "023", "024", "025", "026",
              "027", "028", "029", "030", "031",
              "032", "033", "034", "035", "036",
              "038", "039", "040", "041", "042",
              "044", "045", "046", "047", "048",
              "049", "050", "051", "052", "053", ],
            # [ '001' ],
            augs=augs,
            channels=config['LOADER_OPTIONS']['channels']
        ),
        # 'val': aortaIMMhaDataset(
        #     config['DATAPATH'],
        #     [ "044", "045", "046", "047", "048" ],
        #     # [ '001' ],
        #     channels=config['LOADER_OPTIONS']['channels']
        # ),
        # 'test': aortaIMMhaDataset(
        #     config['DATAPATH'], 
        #     [1],
        #     channels=config['LOADER_OPTIONS']['channels']
        # ),
    }

    return data

def create_model(num_classes):
    model = Unetpp( in_channels=len(config['LOADER_OPTIONS']['channels']),
                    out_channels=num_classes,
                    encoder=config['BACKBONE'] )

    convert_inplace(model, LayerConvertorNNSPT)
    convert_inplace(model, LayerConvertorSm)

    donor = timm.create_model('tf_efficientnetv2_m', drop_path_rate=config['DROP_PATH_RATE'])

    for name, node in donor.named_modules():
        if 'drop_path' in name:
            tokens = name.split('.')
            idx = int(tokens[1])
            jdx = int(tokens[2])

            model.encoder.blocks[idx][jdx].drop_path = donor.blocks[idx][jdx].drop_path

    def decoder_forward(self, *feats):
        xs = dict()
    
        for idx, x in enumerate(feats):
            xs[f'x_{idx}_{idx-1}'] = x
    
        for idx in range(self.nblocks):
            for jdx in range(self.nblocks - idx):
                depth = jdx
                layer = idx+jdx
    
                block = self.blocks[f'b_{depth}_{layer}']
    
                if depth == 0:
                    skip = None
                    shape = xs[f'x_{0}_{-1}'].shape
                else:
                    skip = torch.concat([ xs[f'x_{depth}_{layer-sdx-1}'] for sdx in range(layer-depth+1) ], axis=1)
                    shape = xs[f'x_{depth}_{layer-1}'].shape
    
                x = xs[f'x_{depth+1}_{layer}']
                x = block(x, skip, shape)
                xs[f'x_{depth}_{layer}'] = x
    
        return xs
    
    heads = {}
    
    for idx in range(model.decoder.nblocks):
        heads[f'{idx}'] = copy.deepcopy(model.head)
    
    model.heads = torch.nn.ModuleDict(heads)

    del model.head
    
    def model_forward(self, x):
        f = self.encoder(x)
        xs = self.decoder(*f)
    
        out = tuple()
        
        for idx in range(self.decoder.nblocks):
            x = xs[f'x_{0}_{idx}']
            x = self.heads[f'{idx}'](x)
    
            out = (*out, x)
    
        if self.training:
            return out
        else:
            return x
    
    import types
    
    model.decoder.forward = types.MethodType(decoder_forward, model.decoder)
    model.forward = types.MethodType(model_forward, model)

    del model.decoder.blocks.b_0_0.attention1
    del model.decoder.blocks.b_0_1.attention1
    del model.decoder.blocks.b_0_2.attention1
    del model.decoder.blocks.b_0_3.attention1
    del model.decoder.blocks.b_0_4.attention1

    return model

def dice(tp, fp, fn, tn, *, smooth=1):
    """ NOTE
        binary is common used implementation
        macro is described in https://ieeexplore.ieee.org/document/9433991
    """

    numerator = 2 * tp + smooth
    denominator = tp + fp + tp + fn + smooth

    return numerator / denominator

def matthews(tp, fp, fn, tn, *, smooth=1):
    """ NOTE
        binary is described in https://www2.cs.sfu.ca/~hamarneh/ecopy/isbi2021.pdf
    """

    numerator = tp * tn - fp * fn + smooth
    denominator = ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) + smooth)**0.5

    return numerator / denominator

def get_confusions(y_masks_batch, pred_probs_batch):
    tmp = y_masks_batch * pred_probs_batch
    
    tp = torch.sum(tmp, dim=(1, 2, 3))
    fp = torch.sum(pred_probs_batch-tmp, dim=(1, 2, 3))
    fn = torch.sum(y_masks_batch-tmp, dim=(1, 2, 3))
    tn = torch.sum(1-y_masks_batch-pred_probs_batch+tmp, dim=(1, 2, 3))

    return tp, fp, fn, tn

def combined_loss(y_masks_batch, pred_logitses_batch):
    dice_loss_ = torch.zeros(y_masks_batch.shape[0], dtype=torch.float32, device=config['DEVICE'])
    matthews_loss_ = torch.zeros(y_masks_batch.shape[0], dtype=torch.float32, device=config['DEVICE'])

    pred_activated_logitses_batch = []

    for pred_logits_batch in pred_logitses_batch:
        pred_activated_logits_batch = F.softmax(pred_logits_batch, dim=1)

        pred_activated_logitses_batch.append(pred_activated_logits_batch)

    for class_ in range(1, 24):
        class_y_masks_batch = (y_masks_batch == class_).long()

        for pred_activated_logits_batch in pred_activated_logitses_batch:
            class_pred_activated_logits_batch = pred_activated_logits_batch[:, class_]

            tp, fp, fn, tn = get_confusions(class_y_masks_batch, class_pred_activated_logits_batch) 
            
            dice_loss_ += dice(tp, fp, fn, tn) / 23
            matthews_loss_ += matthews(tp, fp, fn, tn) / 23

    return torch.mean(len(pred_logitses_batch) - dice_loss_), torch.mean(len(pred_logitses_batch) - matthews_loss_)

def inner_supervised(epoch, model, voxels_batch, masks_batch):
    voxels_batch = voxels_batch.to(config['DEVICE'])
    masks_batch = masks_batch.to(config['DEVICE'])

    logitses_batch = model(voxels_batch)

    loss_ = torch.tensor(0, dtype=torch.float32, device=config['DEVICE'])
    focal_loss_ = torch.tensor(0, dtype=torch.float32, device=config['DEVICE'])

    dice_loss_, matthews_loss_ = combined_loss(masks_batch, logitses_batch)
    dice_loss_ = 0.5 * dice_loss_

    for logits_batch in logitses_batch:
        focal_loss = 2 * focal_with_logits_loss(masks_batch, logits_batch, average='macro', activation='softmax')
        focal_loss_ += focal_loss

    loss_ = focal_loss_ + dice_loss_ + matthews_loss_        

    loss = loss_ * 0.2
    matthews_loss = matthews_loss_ * 0.2
    dice_loss = dice_loss_ * 0.2
    focal_loss = focal_loss_ * 0.2

    return loss, matthews_loss, dice_loss, focal_loss

def inner_train_loop(epoch, averaged_model, model, opt, shed, dataset):
    model.train()
    averaged_model.train()

    # augs = V.Sequential([
    #     V.OneOf([
    #         V.RandomGamma(),
    #         V.IntensityShift(shift_limit=0.1),
    #     ], p=p),
    #     V.OneOf([
    #         V.GaussBlur(),
    #         V.GaussNoise(),
    #     ], p=p),
    #     #V.GridDistort(p=p),
    #     V.AxialPlaneAffine(p=p, fill_value=-1000),
    #     V.AxialPlaneDropout(p=p, fill_value=-1000),
    #     #V.AxialPlaneFlip(p=0.1),
    # ], p=1.)

    # augs = V.Sequential([
    #     V.RandomGamma(p=p),
    #     V.IntensityShift(p=p, shift_limit=0.1),
    #     V.GaussNoise(p=p),
    #     V.GridDistort(ncells=5, distort_limit=0.15, p=p),
    #     V.AxialPlaneAffine(p=p, fill_value=-1000),
    #     V.OneOf([
    #         V.AxialPlaneDropout(nplanes=1, fill_value=-1000),
    #         V.PatchShuffle(),
    #         V.PatchDropout()
    #     ], p=p),
    #     # V.AxialPlaneFlip(p=0.75),
    # ], p=1.)

    augs = V.Sequential([
        V.Contrast(
            contrast_limit=0.01,
            p=1.,
        ),
        V.IntensityShift(
            shift_limit=0.1,
            p=1.,
        ),
        V.GaussNoise(
            variance=5.224080277244031,
            p=1.,
        ),
        V.AxialPlaneAffine(
            angle_limit=14.666456843479184,
            shift_limit=0.011608995052867388,
            scale_limit=0.13132223704390011,
            fill_value=-1000,
            p=1.,
        ),
    ], p=1.)

    dataset.augs = augs

    datasampler = VoxelWeightedSampler( dataset,
                                        config['VOXELING_OPTIONS']['voxel_shape'],
                                       dataset.keys,
                                       dataset.shapes,
                                       2 * config['N_ITERATIONS'] * config['BATCH_SIZE'] )

    dataloader = DataLoader( dataset,
                             batch_size=config['BATCH_SIZE'],
                             sampler=datasampler,
                             collate_fn=aortaIMMhaDataset.collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=2 )

    opt.zero_grad(set_to_none=True)

    bloss = torch.tensor(0, dtype=torch.float32, device=config['DEVICE'])
    bmatthews_loss = torch.tensor(0, dtype=torch.float32, device=config['DEVICE'])
    bdice_loss = torch.tensor(0, dtype=torch.float32, device=config['DEVICE'])
    bfocal_loss = torch.tensor(0, dtype=torch.float32, device=config['DEVICE'])

    for step_idx, (voxels_batch, masks_batch, _) in config['VERBOSER'](enumerate(dataloader), total=len(dataloader)):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, matthews_loss, dice_loss, focal_loss = inner_supervised(epoch, model, voxels_batch, masks_batch)

        loss.backward()

        if ( step_idx + 1 ) % 2 == 0:
            opt.step()
            shed.step()

            opt.zero_grad(set_to_none=True)

            averaged_model.update_parameters(model)


        bloss += loss
        bmatthews_loss += matthews_loss
        bdice_loss += dice_loss
        bfocal_loss += focal_loss

        assert torch.isfinite(loss)

        # averaged_model.update_parameters(model)

    datasampler = VoxelRandomSampler( config['VOXELING_OPTIONS']['voxel_shape'],
                                      dataset.keys,
                                      dataset.shapes,
                                      config['N_ITERATIONS'] * config['BATCH_SIZE'] // 4 )

    dataloader = DataLoader( dataset,
                             batch_size=config['BATCH_SIZE'],
                             sampler=datasampler,
                             collate_fn=aortaIMMhaDataset.collate_fn_eval,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=2 )

    tsu.update_bn(dataloader, averaged_model, device=config['DEVICE'])

    bloss = bloss.item() / config['N_ITERATIONS']
    bmatthews_loss = bmatthews_loss.item() / config['N_ITERATIONS']
    bdice_loss = bdice_loss.item() / config['N_ITERATIONS']
    bfocal_loss = bfocal_loss.item() / config['N_ITERATIONS']

    return bloss, bmatthews_loss, bdice_loss, bfocal_loss

def inner_val_loop(model, dataset):
    model.eval()

    metrics = defaultdict(list)

    for idx in config['VERBOSER'](np.arange(len(dataset)), total=len(dataset)):
        key = dataset.keys[idx]
        shape = dataset.shapes[idx]

        masks = dataset.masks[key]

        xsize, ysize, _ = masks.shape

        vxsize, vysize, vzsize = config['VOXELING_OPTIONS']['voxel_shape']
        sxsize, sysize, szsize = config['VOXELING_OPTIONS']['steps']

        # voxel_shape = (xsize, ysize, vzsize)
        # steps = (xsize, ysize, szsize)

        voxel_shape = (vxsize, vysize, vzsize)
        steps = (sxsize, sysize, szsize)
        
        prob_masks = np.zeros((config['N_CLASSES'], *shape), dtype=np.float32)
        count_masks = np.zeros(shape, dtype=np.uint8)

        datasampler = VoxelSequentialSampler( voxel_shape,
                                              [key],
                                              [shape],
                                              steps )

        dataloader = DataLoader( dataset,
                                 batch_size=config['BATCH_SIZE'],
                                 sampler=datasampler,
                                 collate_fn=dataset.collate_fn_eval,
                                 num_workers=config['NJOBS'],
                                 pin_memory=False,
                                 prefetch_factor=2 )

        for voxels_batch, selections in dataloader:
            with torch.no_grad():
                voxels_batch = voxels_batch.to(config['DEVICE'])
                logits_batch = model(voxels_batch)

                prob_masks_batch = logits_batch.softmax(dim=1)
                prob_masks_batch = prob_masks_batch.cpu().float().numpy()

                for prob_masks_tile, selector in zip(prob_masks_batch, selections):
                    prob_masks[(slice(0, None), *selector[-1])] += prob_masks_tile
                    count_masks[selector[-1]] += 1

        prob_masks /= count_masks
        # pred_masks = prob_masks.argmax(axis=0)[None]

        score = 0

        # for idx in range(1, config['N_CLASSES']):
        #     selected_true = ( masks == idx )
        #     selected_pred = ( pred_masks == idx )
        #     TP = np.sum((selected_true == selected_pred) & selected_true)
        #     FP = np.sum((selected_true != selected_pred) & np.logical_not(selected_true))
        #     FN = np.sum((selected_true != selected_pred) & np.logical_not(selected_pred))

        #     score += ( 2*TP / (2*TP + FP + FN) )

        # C = np.product(masks.shape)
        # equal_masks = (masks == pred_masks)
        
        # for idx in range(1, config['N_CLASSES']):
        #     selected_true = ( masks == idx )
        #     selected_pred = ( pred_masks == idx )

        #     TP = np.sum(equal_masks & selected_true)
        #     TN = C - np.sum((selected_true != selected_pred) | selected_true)

        #     score += ( 2*TP / ( C + TP - TN ) )

        #     metrics[f'dice {idx}'].append( 2*TP / ( C + TP - TN ) )

        C = np.product(masks.shape)
        eps = 1

        for idx in range(1, config['N_CLASSES']):
            selected_true = ( masks == idx )
            selected_pred = prob_masks[idx]

            TP = np.sum(selected_true * selected_pred)
            TN = np.sum(( 1 - selected_true ) * ( 1 - selected_pred ))

            scorei = 2*TP / (C-TN+TP+eps)
            score += scorei

            metrics[f'dice {idx}'].append( scorei )

        metrics['dice'].append(score / (config['N_CLASSES'] - 1))

    additional_scores = {}

    for key in metrics:
        mean = float(np.mean(metrics[key]))
        additional_scores[f'{key}'] = mean

    score = additional_scores['dice']

    return score, additional_scores

def get_ema_avg_fn(decay=0.99):
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update

def fit(model, data):
    averaged_model = tsu.AveragedModel(model, avg_fn=get_ema_avg_fn())

    model.to(config['DEVICE'])
    averaged_model.to(config['DEVICE'])

    opt = torch.optim.AdamW(model.parameters(), lr=10**(-3.505932300509118), eps=1e-4)

    # from lion_pytorch import Lion
    # opt = Lion(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1)

    shed = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        pct_start=0.5230937246127159,
        max_lr=10**(-3.505932300509118),
        total_steps=config['EPOCHS']*config['N_ITERATIONS'],
    )

    epochs_without_going_up = 0
    best_score = 0
    best_state = copy.deepcopy(averaged_model.state_dict())

    for epoch in range(config['EPOCHS']):
        start_time = time.perf_counter()

        loss, matthews_loss, dice_loss, focal_loss = inner_train_loop( epoch,
                                                                       averaged_model,
                                                                       model,
                                                                       opt,
                                                                       shed,
                                                                       data['train'] )

        print(f'epoch - {epoch+1} loss - {loss:.6f}')

        config['TASK'].get_logger().report_scalar( title='epoch loss trace',
                                                   series='train loss ' + config['trial_key'],
                                                   iteration=epoch,
                                                   value=loss )

        config['TASK'].get_logger().report_scalar( title='epoch loss trace',
                                                   series='matthews loss ' + config['trial_key'],
                                                   iteration=epoch,
                                                   value=matthews_loss )
        
        config['TASK'].get_logger().report_scalar( title='epoch loss trace',
                                                   series='dise loss ' + config['trial_key'],
                                                   iteration=epoch,
                                                   value=dice_loss )

        config['TASK'].get_logger().report_scalar( title='epoch loss trace',
                                                   series='focal loss ' + config['trial_key'],
                                                   iteration=epoch,
                                                   value=focal_loss )

        # score, additional = inner_val_loop( averaged_model,
        #                                     data['val'] )

        # print(f'epoch - {epoch+1} score - {100 * score:.2f}%')

        # config['TASK'].get_logger().report_scalar( title='epoch score trace',
        #                                            series='val score ' + config['trial_key'],
        #                                            iteration=epoch,
        #                                            value=100*score )

        # for key in additional:
        #     print(f'epoch - {epoch+1} {key} - {100 * additional[key]:.2f}%')

        #     config['TASK'].get_logger().report_scalar( title='epoch additional scores trace',
        #                                                series=f'val {key} ' + config['trial_key'],
        #                                                iteration=epoch,
        #                                                value=100*additional[key] )

        # if best_score <= score:
        #     best_score = score
        #     best_state = copy.deepcopy(averaged_model.state_dict())
        #     epochs_without_going_up = 0

        #     store(averaged_model.module)
        # else:
        #     epochs_without_going_up += 1

        # if epochs_without_going_up == config['STOP_EPOCHS']:
        #     break

        store(averaged_model.module)

        if (epoch + 1) == 10:
            break

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f'elapsed time {elapsed_time:.2f} s')
        print(f'epoch without improve {epochs_without_going_up}')

    averaged_model.load_state_dict(best_state)
    model.load_state_dict(averaged_model.module.state_dict())

    return best_score

def store(model):
    state = model.state_dict()
    path = config['MODELNAME']

    torch.save(state, path)

import optuna

@click.command()
@click.option('--datapath', '-dp', type=str)
@click.option('--loader_options_path', '-lop', type=str, default='loader_options.yaml')
@click.option('--voxeling_options_path', '-vop', type=str, default='voxeling_options.yaml')
@click.option('--learning_rate', '-lr', type=float, default=1e-3)
@click.option('--max_learning_rate', '-mlr', type=float, default=1e-2)
@click.option('--epochs', '-e', type=int, default=10, help='The number of epoch per train loop')
@click.option('--stop_epochs', '-se', type=int, default=5)
@click.option('--n_iterations', '-ni', type=int, default=1000, help='The number of iteration per epoch')
@click.option('--batch_size', '-bs', type=int, default=4)
@click.option('--backbone', '-bone', type=str, default='timm-efficientnetv2-m')
@click.option('--drop_path_rate', '-dpr', type=float, default=0.)
@click.option('--modelname', '-mn', type=str, default='model.pth')
@click.option('--njobs', type=int, default=1, help='The number of jobs to run in parallel.')
@click.option('--verbose', is_flag=True, help='Whether progress bars are showed')
@click.option('--seed', '-seed', type=int, default=1996)
def main(**kwargs):
    config['N_CLASSES'] = 24

    init_global_config(**kwargs)

    for key in config:
        print(f'{key} {config[key]}')

    print(f'load data')
    data = load_data()

    # model = torch.compile(model, mode='reduce-overhead')

    task = Task.init(project_name='AortaSeg', task_name=f"Segmentation_{config['PREFIX']}")
    config['TASK'] = task

    task.upload_artifact( name='train.py', artifact_object='train.py' )

    init_determenistic(seed=kwargs['seed'])

    model = create_model(num_classes=config['N_CLASSES'])

    config['trial_key'] = ''
    
    score = fit(model, data)

    # print(f'create study')
    # study = optuna.create_study()

    # def objective(trial):
    #     init_determenistic(seed=kwargs['seed'])

    #     model = create_model(num_classes=config['N_CLASSES'])

    #     pct_start = trial.suggest_float('pct_start', 0.1, 0.6)
    #     lr = trial.suggest_float('lr', 2, 5)

    #     config['pct_start'] = pct_start
    #     config['lr'] = 10**(-lr)

    #     config['trial_key'] = f'pct_start {pct_start:.2f} lr {lr:.2f}'

    #     score = fit(model, data)

    #     return 1 - score

    # study.enqueue_trial({ 'pct_start': 0.3, 'lr': 3 })
    # study.optimize(objective, n_trials=15)

if __name__ == '__main__':
    main()