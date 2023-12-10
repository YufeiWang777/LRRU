# CONFIG
import argparse
arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='LRRU')
arg.add_argument('-c', '--configuration', type=str, default='train_lrru_small_kitti.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)

# ENVIRONMENT SETTINGS
import os
rootPath = os.path.abspath(os.path.dirname(__file__))
import functools
if len(config.gpus) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = functools.reduce(lambda x, y: str(x) + ',' + str(y), config.gpus)
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = str(config.port)
if not config.record_by_wandb_online:
    os.environ["WANDB_MODE"] = 'dryrun'

# BASIC PACKAGES
import time
import emoji
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# MULTI-GPU AND MIXED PRECISION
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

# MODULES
from dataloaders.kitti_loader import KittiDepth
from model import get as get_model
from optimizer_scheduler import make_optimizer_scheduler
from summary import get as get_summary
from metric import get as get_metric
from utility import *
from loss import get as get_loss

# VARIANCES
pbar, pbar_val = None, None
batch, batch_val = None, None
checkpoint, best_metric_rmse, best_metric_mae = None, None, None
writer_train, writer_val = None, None
warm_up_cnt, warm_up_max_cnt = None, None
loss_sum_val = torch.from_numpy(np.array(0))
sample, sample_val, sample_, output, output_, output_val = None, None, None, None, None, None
log_itr, log_cnt, log_loss, log_cnt_val, log_loss_val, log_val = None, None, None, None, None, 0
loss_jin, loss_an = None, None
val_metric = None
val_metric_rmse,val_metric_mae = None, None

def train(gpu, args):

    # GLOBAL INVARIANCE
    global checkpoint, best_metric_rmse, best_metric_mae, log_val, loss_jin, loss_an, an_val_metric, jin_val_metric, ben_val_metric
    global val_metric_rmse, val_metric_mae
    global warm_up_cnt, warm_up_max_cnt
    global pbar, pbar_val
    global sample, sample_val, output, output_val
    global writer_train, writer_val
    global batch, batch_val
    global log_itr, log_cnt, log_loss, log_cnt_val, log_loss_val
    global loss_sum_val
    if gpu == 0:
        print(args.dump())

    # INITIALIZE
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.num_gpus, rank=gpu)

    # MINIMIZE RANDOMNESS
    rank = torch.distributed.get_rank()
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed(config.seed + rank)
    torch.cuda.manual_seed_all(config.seed + rank)
    np.random.seed(config.seed + rank)
    random.seed(config.seed + rank)
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # WANDB
    if gpu == 0:
        best_metric_rmse = 1e8
        best_metric_mae = 1e8
        import wandb
        wandb.login()
        if not args.resume:
            wandb.init(dir=rootPath, config=args, project=args.project_name)
            args.defrost()
            args.save_dir = os.path.split(wandb.run.dir)[0]
            args.freeze()
            with open(args.save_dir + '/' + 'config.txt', 'w') as f:
                f.write(args.dump())
    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                import wandb
                if len(args.wandb_id_resume) == 0:
                    wandb.init(dir=rootPath, config=args, project=args.project_name, resume=True)
                else:
                    assert len(args.wandb_id_resume) != 0, 'wandb_id should not be empty when resuming'
                    wandb.init(id=args.wandb_id_resume, dir=rootPath, config=args, project=args.project_name,
                               resume='must')
                args.defrost()
                args.save_dir = os.path.split(wandb.run.dir)[0]
                args.freeze()
                print('=> Resume wandb from : {}'.format(args.wandb_id_resume))

    # DATASET
    if gpu == 0:
        print(emoji.emojize('Prepare data... :writing_hand:', variant="emoji_type"))
    batch_size = args.batch_size // args.num_gpus
    data_train = KittiDepth('train', args)
    sampler_train = DistributedSampler(data_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               shuffle=False, num_workers=args.num_threads,
                                               pin_memory=True, sampler=sampler_train, drop_last=True)
    if gpu == 0:
        len_train_dataset = len(loader_train) * batch_size * args.num_gpus
        print('=> Train dataset: {} samples'.format(len_train_dataset))
    data_val = KittiDepth('val', args)
    sampler_val = SequentialDistributedSampler(data_val, batch_size=batch_size)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                                             shuffle=False, num_workers=args.num_threads,
                                             pin_memory=True, sampler=sampler_val)
    if gpu == 0:
        len_val_dataset = len(loader_val) * batch_size * args.num_gpus
        print('=> Val dataset: {} samples'.format(len_val_dataset))

    # NETWORK
    if gpu == 0:
        print(emoji.emojize('Prepare network... :writing_hand:', variant="emoji_type"))
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    models = get_model(args)
    net = models(args)
    net.cuda(gpu)

    if gpu == 0:
        total_params = count_parameters(net)
        if len(args.pretrain) != 0:
            assert os.path.isfile(args.pretrain), "file not found: {}".format(args.pretrain)
            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'], strict=args.load_model_strict)
            print('=> Load network parameters from : {}'.format(args.pretrain))

    # OPTIMIZER
    if gpu == 0:
        print(emoji.emojize('Prepare optimizer... :writing_hand:', variant="emoji_type"))
    optimizer, scheduler = make_optimizer_scheduler(args, net)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O0', verbosity=0)

    # IF RESUME
    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    amp.load_state_dict(checkpoint['amp'])
                    args.defrost()
                    args.start_epoch = checkpoint['epoch'] + 1
                    args.log_itr = checkpoint['log_itr']
                    args.freeze()
                    print('=> Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('=> State dicts for resume are not saved. '
                          'Use --save_full argument')
            del checkpoint

    net = DDP(net)

    # LOSSES
    if gpu == 0:
        print(emoji.emojize('Prepare loss... :writing_hand:', variant="emoji_type"))
        print('=> Loss: {}'.format(args.loss))
    loss = get_loss(args)
    loss = loss(args, args.loss)
    loss.cuda()

    # METRIC
    if gpu == 0:
        print(emoji.emojize('Prepare metric... :writing_hand:', variant="emoji_type"))
    metric = get_metric(args)
    metric = metric(args)

    # SUMMARY
    if gpu == 0:
        print(emoji.emojize('Prepare summary... :writing_hand:', variant="emoji_type"))
        summary = get_summary(args)
        writer_train = summary(args.save_dir, 'train', args,
                               loss.loss_name, metric.metric_name)
        writer_val = summary(args.save_dir, 'val', args,
                             loss.loss_name, metric.metric_name)

    if gpu == 0:
        log_itr = args.log_itr
        backup_source_code(args.save_dir + '/backup_code')
        try:
            assert os.path.isdir(args.save_dir)
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass
        print('=> Save backup source code and makedirs done')

    # GO
    for epoch in range(args.start_epoch, args.epochs + 1):

        # TRAIN
        net.train()
        loader_train.sampler.set_epoch(epoch)

        # LOG
        if gpu == 0:
            print(emoji.emojize('Let\'s do something interesting :oncoming_fist:', variant="emoji_type"))
            current_time = time.strftime('%y%m%d@%H:%M:%S')
            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])
            print('=======> Epoch {:5d} / {:5d} | Lr : {} | {} | {} <======='.format(
                epoch, args.epochs, list_lr, current_time, args.save_dir
            ))
            num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        # ROUND
        for batch, sample in enumerate(loader_train):
            sample = {k: value.cuda(gpu) for k, value in sample.items() if torch.is_tensor(value)}

            output = net(sample)

            # LOSS
            loss_sum, loss_val = loss(output['results'], sample['gt'])
            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # METRIC
            if gpu == 0:
                metric_train = metric.evaluate(output['results'][-1], sample['gt'], 'train')
                writer_train.add(loss_val, metric_train, log_itr)

                log_itr += 1
                log_cnt += 1
                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{}|Ls={:.4f}|Loss={:.4f}'.format(
                    'Train', loss_sum.item(), loss_sum.item())
                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str, list_lr)
                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)

            # VAL
            if  epoch > args.val_epoch:

                # ENVIRONMENT SETTING
                torch.set_grad_enabled(False)
                net.eval()

                # LOG
                if gpu == 0:
                    num_sample_val = len(loader_val) * loader_val.batch_size * args.num_gpus
                    pbar_val = tqdm(total=num_sample_val)
                    log_cnt_val = 0.0
                    log_loss_val = 0.0
                loss_val_list, metric_val_list = [], []

                # ROUND
                for batch_val, sample_val in enumerate(loader_val):
                    sample_val = {key: val.cuda(gpu) for key, val in sample_val.items()
                                  if val is not None}

                    output_val = net(sample_val)

                    # LOG
                    for depth, supervised, gt in zip(torch.chunk(output_val['results'][-1], batch_size, dim=0),
                                                         torch.chunk(sample_val['gt'], batch_size, dim=0),
                                                         torch.chunk(sample_val['gt'], batch_size, dim=0)):

                        loss_sum_val, loss_val = loss(depth, supervised)
                        loss_val_list.append(loss_val)
                        metric_val_list.append(metric.evaluate(depth, gt, 'val'))

                    if gpu == 0:
                        current_time = time.strftime('%y%m%d@%H:%M:%S')
                        error_str = '{}|Loss={:.4f}'.format(
                            'Val', loss_sum_val.item())
                        pbar_val.set_description(error_str)
                        pbar_val.update(loader_val.batch_size * args.num_gpus)

                loss_val_all = distributed_concat(torch.cat(loss_val_list, axis=0),
                                                  len(loader_val.dataset))
                metric_val_all = distributed_concat(torch.cat(metric_val_list, axis=0),
                                                    len(loader_val.dataset))
                if gpu == 0:
                    pbar_val.close()
                    for i, j in zip(loss_val_all, metric_val_all):
                        writer_val.add(i.unsqueeze(0), j.unsqueeze(0))
                    val_metric_rmse, val_metric_mae = writer_val.update(log_val, sample_val, output_val,
                                                   online_loss=True, online_metric=True,
                                                   online_rmse_only=True, online_img=False)

                # SAVE CHECKPOINT
                if gpu == 0:
                    tmp = args.val_iters // (loader_train.batch_size * args.num_gpus)
                    if (epoch <= args.val_epoch and batch == len(loader_train) - 1) or (epoch > args.val_epoch
                        and batch + 1 == (len(loader_train) // tmp) * tmp):
                        writer_val.save(epoch, batch + 1, sample_val, output_val)

                    if val_metric_rmse < best_metric_rmse:
                        best_metric_rmse = val_metric_rmse
                        state = {
                            'net': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'amp': amp.state_dict(),
                            'epoch': epoch,
                            'log_itr': log_itr,
                            'args': args
                        }
                        torch.save(state, '{}/best_rmse_model.pt'.format(args.save_dir))
                    if val_metric_mae < best_metric_mae:
                        best_metric_mae = val_metric_mae
                        state = {
                            'net': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'amp': amp.state_dict(),
                            'epoch': epoch,
                            'log_itr': log_itr,
                            'args': args
                        }
                        torch.save(state, '{}/best_mae_model.pt'.format(args.save_dir))
                    log_val += 1
                torch.set_grad_enabled(True)
                net.train()

        # LOG
        if gpu == 0:
            pbar.close()
            _,_ = writer_train.update(epoch, sample, output,
                                    online_loss=True, online_metric=True,
                                    online_rmse_only=True, online_img=False)
            state = {
                'net': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'amp': amp.state_dict(),
                'epoch': epoch,
                'log_itr': log_itr,
                'args': args
            }
            # torch.save(state, '{}/latest_model.pt'.format(args.save_dir))
            if epoch == args.val_epoch or epoch == 40:
                torch.save(state, '{}/model_epoch{}.pt'.format(args.save_dir, args.val_epoch))
            else:
                torch.save(state, '{}/latest_model.pt'.format(args.save_dir))
        scheduler.step()


def main(args):

    if args.no_multiprocessing:
        train(0, args)
    else:
        assert args.num_gpus > 0

        spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                 join=False)

        while not spawn_context.join():
            pass

        for process in spawn_context.processes:
            if process.is_alive():
                process.terminate()
            process.join()


if __name__ == '__main__':
    main(config)
