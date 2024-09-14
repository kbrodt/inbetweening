import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import shlex
import textwrap
import subprocess

import random
from pathlib import Path

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.jit
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import timm
import timm.scheduler
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.model_selection import KFold

import models
import dataset
import dino


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-dir",
        type=str,
        default="./data/inputs",
        help="path to train df",
    )
    parser.add_argument(
        "--train-img-dir",
        type=str,
        help="path to train df",
        default="./data/inputs",
    )
    parser.add_argument(
        "--test-img-dir",
        type=str,
        help="path to test df",
        default="./data/inputs",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="logs")

    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_s_in21ft1k")

    parser.add_argument("--loss", type=str, default="xent", choices=["bce"])
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dec-channels", type=int, nargs="+", default=[256, 240, 224, 208, 192])
    parser.add_argument("--dec-attn-type", type=str, default=None)
    parser.add_argument("--n-classes", type=int, default=1)
    parser.add_argument("--ds-mult", type=int, default=1)

    parser.add_argument("--optim", type=str, default="adamw", help="optimizer name")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-decay-scale", type=float, default=1e-2)
    parser.add_argument("--warmup-steps-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-t", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-2)  # 5e-5)

    #parser.add_argument("--vit_bb_lr", default={10: 1e-6, 20: 1.25e-6, 26: 2.5e-6, 32: 5e-6})  # ViT-H 5975
    #parser.add_argument("--vit_bb_lr", default={4: 1e-6, 8: 2.5e-6, 10: 5e-6, 12: 1e-5})
    parser.add_argument("--vit_bb_lr", default={8: 1.25e-6, 16: 2.5e-6, 20: 5e-6, 24: 1e-5})  # ViT-L

    parser.add_argument("--scheduler", type=str, default="wucos", help="scheduler name")
    parser.add_argument("--scheduler-mode", type=str, default="step", choices=["step", "epoch"], help="scheduler mode")
    parser.add_argument("--T-max", type=int, default=440)
    parser.add_argument("--eta-min", type=int, default=0)

    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8,
    )
    parser.add_argument(
        "--num-epochs", type=int, help="number of epochs to train", default=440,
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=314159,
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        help="number of folds",
        default=10,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="fold",
        default=0,
    )

    parser.add_argument(
        "--distributed", action="store_true", help="distributed training"
    )
    parser.add_argument("--syncbn", action="store_true", help="sync batchnorm")
    parser.add_argument(
        "--deterministic", action="store_true", help="deterministic training"
    )
    parser.add_argument(
        "--load", type=str, default="", help="path to pretrained model weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")
    parser.add_argument("--img-size", type=int, nargs=2)
    parser.add_argument("--ft", action="store_true", help="finetune")

    args = parser.parse_args(args=args)

    return args


def notify(title, summary, value=-1, server="sun"):
    print(title, summary)
    return
    cmd = textwrap.dedent(
        f"""
            ssh {server} \
                '\
                    export DISPLAY=:0 \
                    && dunstify -t 0 -h int:value:{value} "{title}" "{summary}" \
                '
        """
    )
    cmds = shlex.split(cmd)
    with subprocess.Popen(cmds, start_new_session=True):
        pass


def epoch_step_train(epoch, loader, desc, model, criterion, optimizer, scaler, scheduler=None, fp16=False, local_rank=0, summary_writer=None, prefix="train"):
    model.train()

    if local_rank == 0:
        pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    n_steps_per_epoch = len(loader)
    num_updates = epoch * n_steps_per_epoch
    loc_loss = n = 0
    for item in loader:
        images = item["img"]
        target = item["mask"]
        starget = item["smask"]
        label = item["label"]

        images = images.cuda(local_rank, non_blocking=True)
        images = images.to(memory_format=torch.channels_last)
        target = target.cuda(local_rank, non_blocking=True)
        starget = starget.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(images)
            logits = torch.nn.functional.interpolate(
                logits,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            #loss = criterion(logits, target)
            loss = criterion(logits, target, starget, label)

            #loss = criterion(logits, torch.cat([target, starget], dim=1))
            #loss = label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * loss
            #loss = loss.mean()

        #loss.backward()
        #torch.nn.utils.clip_grad_norm_(
            #model.parameters(), 5.0
        #)  # , error_if_nonfinite=False)
        #optimizer.step()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
            # error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        num_updates += 1
        optimizer.zero_grad(set_to_none=True)

        bs = target.size(0)
        loc_loss += loss.item() * bs
        n += bs

        if scheduler is not None:
            scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        if local_rank == 0:
            postfix = {
                "loss": f"{loc_loss / n:.3f}",
            }
            pbar.set_postfix(**postfix)
            pbar.update()

        if np.isnan(loc_loss) or np.isinf(loc_loss):
            break

    if local_rank == 0:
        pbar.close()

        if summary_writer is not None:
            #images *= 0.5
            #images += 0.5

            logits, slogits = logits.split(1, dim=1)

            images_ori = images.clone()
            images_ori *= torch.from_numpy(dataset.ADE_STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(images.device)
            images_ori += torch.from_numpy(dataset.ADE_MEAN).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(images.device)
            nrow = 3
            grid = torchvision.utils.make_grid(images_ori, nrow=nrow)
            summary_writer.add_image(f"{prefix}/images", grid, epoch)
            grid = torchvision.utils.make_grid(target, nrow=nrow)
            summary_writer.add_image(f"{prefix}/target", grid, epoch)#, dataformats="NHW")
            grid = torchvision.utils.make_grid(logits.sigmoid(), nrow=nrow)
            summary_writer.add_image(f"{prefix}/preds", grid, epoch)#, dataformats="NHW")

            grid = torchvision.utils.make_grid(starget, nrow=nrow)
            summary_writer.add_image(f"{prefix}/starget", grid, epoch)#, dataformats="NHW")
            grid = torchvision.utils.make_grid(slogits.sigmoid(), nrow=nrow)
            summary_writer.add_image(f"{prefix}/spreds", grid, epoch)#, dataformats="NHW")

    return loc_loss, n


def calc_iu(pred, target, thresh=0.5, min_area=float('-inf')):
    pred = pred > thresh
    #if pred.sum(1) < min_area:
    #    pred[:] = False

    pred = pred.flatten(1)
    target = target.flatten(1)

    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)

    return inter, union


def calc_dice(pred, target, thresh=0.5, eps=1e-8, min_area=float('-inf')):
    inter, union = calc_iu(pred, target, thresh=thresh, min_area=min_area)

    return (inter + eps) / (union - inter + eps)


class Metric:
    def __init__(self, thresh=None):
        self.thresh = thresh if thresh is not None else 0.5

        self.clean()

    def clean(self):
        self.dice = 0.0
        self.n = 0

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        preds = preds.sigmoid().cpu()
        targets = targets.cpu()

        self.dice += calc_dice(preds, targets, self.thresh).sum(0).item()

        self.n += len(preds)

    def evaluate(self):
        if self.n == 0:
            return 0.0

        res = self.dice / self.n

        return res


@torch.inference_mode()
def epoch_step_val(epoch, loader, desc, model, criterion, metric, fp16=False, local_rank=0, summary_writer=None, prefix="dev"):
    model.eval()

    if local_rank == 0:
        pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    loc_loss = 0.0
    n = 1
    for item in loader:
        images = item["img"]
        target = item["mask"]
        starget = item["smask"]
        #label = item["label"]

        images = images.cuda(local_rank, non_blocking=True)
        images = images.to(memory_format=torch.channels_last)
        #target = target.cuda(local_rank, non_blocking=True)
        #starget = starget.cuda(local_rank, non_blocking=True)
        #label = label.cuda(local_rank, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(images)
            logits = torch.nn.functional.interpolate(
                logits,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            #loss = criterion(logits, target)

        #bs = target.size(0)
        #n += bs
        #loc_loss += loss.item() * bs

        # logits = logits.softmax(dim=1)
        logits, slogits = logits.split(1, dim=1)
        metric["ov"].update(logits, target)

        metric["mask"].update(slogits, starget)

        # hflip
        #logits += torch.flip(model(torch.flip(images, dims=[-1])), dims=[-1]).softmax(dim=1)
        #metric["tta_2"].update(logits / 2, target)

        #logits += torch.flip(model(torch.flip(images, dims=[-2])), dims=[-2]).softmax(dim=1)
        #metric["tta_3"].update(logits / 3, target)

        #logits += torch.flip(model(torch.flip(images, dims=[-2, -1])), dims=[-2, -1]).softmax(dim=1)
        #metric["tta_4"].update(logits / 4, target)

        torch.cuda.synchronize()

        if local_rank == 0:
            #postfix = {
            #    "loss": f"{loc_loss / n:.3f}",
            #}
            #pbar.set_postfix(**postfix)
            pbar.update()

    if local_rank == 0:
        pbar.close()

        if summary_writer is not None:
            #images *= 0.5
            #images += 0.5
            images_ori = images.clone()
            images_ori *= torch.from_numpy(dataset.ADE_STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(images.device)
            images_ori += torch.from_numpy(dataset.ADE_MEAN).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(images.device)

            nrow = 3
            grid = torchvision.utils.make_grid(images_ori, nrow=nrow)
            summary_writer.add_image(f"{prefix}/images", grid, epoch)
            grid = torchvision.utils.make_grid(target, nrow=nrow)
            summary_writer.add_image(f"{prefix}/target", grid, epoch)#, dataformats="NHW")
            grid = torchvision.utils.make_grid(logits.sigmoid(), nrow=nrow)
            summary_writer.add_image(f"{prefix}/preds", grid, epoch)#, dataformats="NHW")

            grid = torchvision.utils.make_grid(starget, nrow=nrow)
            summary_writer.add_image(f"{prefix}/starget", grid, epoch)#, dataformats="NHW")
            grid = torchvision.utils.make_grid(slogits.sigmoid(), nrow=nrow)
            summary_writer.add_image(f"{prefix}/spreds", grid, epoch)#, dataformats="NHW")

    #fig = plt.figure(figsize=(12, 12))
    #ax = fig.add_subplot(1, 1, 1)
    #x = target[0].cpu().numpy().ravel()
    #y = logits[0, 0].detach().cpu().numpy().ravel()
    #ax.set_title(f"RMSE: {np.sqrt(((x - y) ** 2).mean()):.3f}")
    #ax.plot(np.arange(x.max()), np.arange(x.max()))
    #ax.scatter(x, y)
    #summary_writer.add_figure('dev/hist', fig, epoch)
    #plt.close(fig)

    return loc_loss, n


def create_df(data_dir, suffix="*_img.*", use_animation=True):
    data_dir = Path(data_dir)

    #data_files = set(p.name.split(suffix.strip("*"))[0] for p in data_dir.glob(suffix))
    #_data_files = data_files & set(p.name.split("_mask.")[0] for p in data_dir.glob("*_mask.*"))

    #data_files = sorted(p.name for p in data_dir.glob(suffix) if p.name.split(suffix.strip("*"))[0] in _data_files)
    data_files = sorted(p.name for p in data_dir.glob(suffix))
    df = pd.DataFrame(data_files, columns=["fname"])
    if use_animation:
        df["animation"] = df.fname.str.split("_").str[1].astype("int")

    return df


def train_test_split(df, fold=0, n_folds=10, random_state=314159):
    df = df.groupby("animation").agg(list)
    df["fold"] = None
    n_col = len(df.columns) - 1

    skf = KFold(
        n_splits=n_folds, shuffle=True, random_state=random_state,
    )
    for fold, (_, dev_index) in enumerate(skf.split(df)):
        df.iloc[dev_index, n_col] = fold

    df = df.explode("fname")

    train = df[df.fold != fold].reset_index(drop=True)
    test = df[df.fold == fold].reset_index(drop=True)

    return train, test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(args):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if args.deterministic:
        set_seed(args.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    if args.distributed:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def add_weight_decay(args, model, skip_list=()):
    params = []
    params.extend(
        model.get_parameter_section(
            model.backbone.named_parameters(),
            lr=args.vit_bb_lr,
            wd=args.weight_decay,
            skip_list=skip_list,
        )
    )

    #decay, no_decay = [], []
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            continue

        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            params.append(
                {
                    "params": param,
                    "lr": 1e-1 * args.learning_rate if name.startswith("backbone.") else args.learning_rate,
                    #"lr": args.learning_rate,
                    "weight_decay": 0.0,
                }
            )
            #no_decay.append(param)
        else:
            #decay.append(param)
            params.append(
                {
                    "params": param,
                    "lr": 1e-1 * args.learning_rate if name.startswith("backbone.") else args.learning_rate,
                    #"lr": args.learning_rate,
                    "weight_decay": args.weight_decay,
                }
            )

    #params = [
    #    {"params": no_decay, "weight_decay": 0.0},
    #    {"params": decay, "weight_decay": weight_decay},
    #]

    #backbone_params = list(model.encoder.parameters())
    #params = (
        #list(model.attn.parameters())
        #+ list(model.decoder.parameters())
        #+ list(model.segmentation_head.parameters())
    #)
    #params = [
        #{"params": backbone_params, "lr": 1e-1 * args.learning_rate},
        #{"params": params, "lr": args.learning_rate},
    #]

    return params


def save_jit(model, args, model_path):
    return
    if hasattr(model, "module"):
        model = model.module

    if args.backbone.startswith("efficientnet"):
        model.encoder.set_swish(memory_efficient=False)

    model.eval()
    inp = torch.rand(1, args.in_channels, args.img_size[0], args.img_size[1]).cuda(int(os.environ.get("LOCAL_RANK", 0)))
    # inp = torch.rand(1, args.in_channels, 512, 512).cuda(int(os.environ.get("LOCAL_RANK", 0)))

    with torch.inference_mode():
        #with torch.cuda.amp.autocast(enabled=True):
        traced_model = torch.jit.trace(model, inp)

    #traced_model = torch.jit.freeze(traced_model)

    traced_model.save(model_path)

    if args.backbone.startswith("efficientnet"):
        model.encoder.set_swish(memory_efficient=True)


def all_gather(value, n, is_dist):
    if is_dist:
        if n is not None:
            vals = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(vals, value)
            ns = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(ns, n)

            n = sum(ns)
            if isinstance(value, dict):
                val = {
                    k: sum(val[k] for val in vals) / n
                    for k in value
                }
            else:
                val = sum(vals) / n

        else:
            vals = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(vals, value)
            val = []
            for v in vals:
                val.extend(v)

    elif isinstance(value, dict):
        val = {k: v / n for k, v in value.items()}
    else:
        if n is not None:
            val = value / n
        else:
            val = value

    return val


def mixup_data(x, y, alpha=1):#, mix_y=False):
    bs = x.size(0)
    lam = np.random.beta(alpha, alpha, size=bs)
    lam = x.new_tensor(lam).unsqueeze(1).unsqueeze(1)#.unsqueeze(1)

    index = torch.randperm(bs)
    mixed_x = lam * x + (1 - lam) * x[index]
    # if mix_y:
    y = F.one_hot(y, N_CLASSES).float()
    lam = lam.squeeze(-1)#.squeeze(-1)
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam

    # return mixed_x, (y, y[index]), lam


def mixup_criterion(criterion, logits, target, lam):
    if isinstance(target, tuple):
        if isinstance(lam, tuple):
            target_a, target_b, target_c, target_d = target
            loss_a = criterion(logits, target_a)
            loss_b = criterion(logits, target_b)
            loss_c = criterion(logits, target_c)
            loss_d = criterion(logits, target_d)
            mu, lam1, lam2 = lam
            mu = mu.squeeze()
            lam1 = lam1.squeeze()
            lam2 = lam2.squeeze()
            loss1 = lam1 * loss_a  + (1 - lam1) * loss_b
            loss2 = lam2 * loss_c  + (1 - lam2) * loss_d
            loss = mu * loss1 + (1 - mu) * loss2
        else:
            target_a, target_b = target
            loss_a = criterion(logits, target_a)
            loss_b = criterion(logits, target_b)
            lam = lam.squeeze()
            loss = lam * loss_a  + (1 - lam) * loss_b
    else:
        loss = criterion(logits, target)
        # loss = loss * lam / lam.sum()

    loss = loss.mean()

    return loss


class BCE(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_none = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target, starget, labels):
        logits, slogits = logits.split(1, dim=1)

        loss = self.bce(logits, target)

        loss_none = self.bce_none(slogits, starget)
        loss_none = (labels * loss_none.mean((-1, -2, -3))).mean(0)
        loss = loss + loss_none

        return loss


def train(args):
    init_dist(args)

    torch.backends.cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    summary_writer = None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(args)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(checkpoint_dir / "logs")

        notify(
            checkpoint_dir.name,
            f"start training",
        )

    model = build_model(args)
    model = model.cuda(local_rank)
    model = model.to(memory_format=torch.channels_last)

    checkpoint = None
    if args.load:
        path_to_resume = Path(args.load).expanduser()
        if path_to_resume.is_file():
            print(f"=> loading resume checkpoint '{path_to_resume}'")
            #model = torch.load(args.load, map_location=lambda storage, loc: storage.cuda(local_rank))
            checkpoint = torch.load(
                path_to_resume,
                map_location="cpu", #lambda storage, loc: storage.cuda(local_rank),
            )

            nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["state_dict"], "module.")
            model.load_state_dict(checkpoint["state_dict"])
            print(
                f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{path_to_resume}'")

    # if local_rank == 0:
        # torch.save(model, checkpoint_dir / "last_model.pth")

    #save_jit(model, args, checkpoint_dir / "model_best.pt")
    #return

    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    parameters = add_weight_decay(args, model, skip)

    optimizer = build_optimizer(parameters, args)

    if args.distributed:
        if args.syncbn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            #find_unused_parameters=True,
        )

    img_dir = Path(args.img_dir)
    #suffix = "*_img.png"
    suffix = "*_img.*"
    df = create_df(img_dir, suffix=suffix)

    #suffix = "*_img_s.png"
    #suffix = "*_img_s.*"

    train_df, dev_df = train_test_split(
        df,
        fold=args.fold,
        n_folds=args.n_folds,
        random_state=args.random_state,
    )
    #train_df = train_df.head(100)
    #dev_df = dev_df.head(100)

    if args.ft:
        train_df = df.copy()

    # train_df = train_df[:1000]
    # dev_df = dev_df[:1000]

    if local_rank == 0:
        print(train_df.sample(n=30, random_state=1))
        print(dev_df)

    train_dataset = dataset.DS(
        df=train_df,
        img_dir=img_dir,
        augs=dataset.get_augs(args.img_size),
    )

    val_dataset = dataset.DS(
        df=dev_df,
        img_dir=img_dir,
        augs=dataset.get_test_augs(args.img_size),
    )

    train_img_dir = Path(args.train_img_dir)
    train_df_r = create_df(train_img_dir, suffix=suffix, use_animation=False)
    if local_rank == 0:
        print(train_df_r)

    train_df_r = pd.concat([train_df_r] * args.ds_mult)
    train_dataset_r = dataset.TestDS(
        df=train_df_r,
        img_dir=train_img_dir,
        augs=dataset.get_augs(args.img_size),
    )
    train_dataset = torch.utils.data.ConcatDataset(
        [
            train_dataset,
            train_dataset_r,
        ],
    )
    # train_dataset = train_dataset_r

    test_img_dir = Path(args.test_img_dir)
    test_df = create_df(test_img_dir, suffix=suffix, use_animation=False)
    test_dataset = dataset.TestDSUN(
        df=test_df,
        img_dir=test_img_dir,
        augs=dataset.get_test_augs(args.img_size),
    )

    # proba = train_df.target.value_counts().values
    # proba = proba / proba.sum()
    # to_weights = dict(zip(train_df.target.value_counts().index, 1 / proba))
    # train_df['w'] = train_df.target.apply(lambda x: to_weights[x])
    # weights = torch.from_numpy(train_df.w.values.astype('float32'))
    # print(train_df)

    # train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if args.distributed:
        # train_sampler = DistributedWeightedSampler(train_dataset, weights)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    args.num_workers = min(args.batch_size, 2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
    )

    scheduler = build_scheduler(optimizer, args, n=len(train_loader) if args.scheduler_mode == "step" else 1)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    criterion = build_criterion(args)
    criterion = criterion.cuda(local_rank)

    metric = {
        "ov": Metric(),
        "mask": Metric(),
        #"tta_2": Metric(args.n_classes - 1),
        #"tta_3": Metric(),
        #"tta_4": Metric(),
    }

    def saver(path, score):
        torch.save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "score": score,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict(),
                "args": args,
            },
            path,
        )

    res = 0
    start_epoch = 0
    best_score = 0
    if args.resume and checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        if checkpoint["sched_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["sched_state_dict"])

        optimizer.load_state_dict(checkpoint["opt_state_dict"])
        scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        desc = f"{epoch}/{args.num_epochs}"

        train_loss, n = epoch_step_train(
            epoch,
            train_loader,
            desc,
            model,
            criterion,
            optimizer,
            scaler,
            scheduler=scheduler if args.scheduler_mode == "step" else None,
            fp16=args.fp16,
            local_rank=local_rank,
            summary_writer=summary_writer,
        )

        train_loss = all_gather(train_loss, n, args.distributed)
        if np.isnan(train_loss) or np.isinf(train_loss):
            print('is nan')
            res = 1
            break

        for m in metric.values():
            m.clean()

        epoch_step_val(
            epoch,
            test_loader,
            desc,
            model,
            criterion,
            metric,
            fp16=args.fp16,
            local_rank=local_rank,
            summary_writer=summary_writer,
            prefix="test",
        )

        test_scores = {}
        for key, m in metric.items():
            test_scores[key] = m.evaluate()

        for m in metric.values():
            m.clean()

        dev_loss, n = epoch_step_val(
            epoch,
            val_loader,
            desc,
            model,
            criterion,
            metric,
            fp16=args.fp16,
            local_rank=local_rank,
            summary_writer=summary_writer,
        )

        dev_loss = all_gather(dev_loss, n, args.distributed)

        # TOTO distributed
        #for m in metric.values():
            #m.scores = all_gather(m.scores, None, args.distributed)

        dev_scores = {}
        for key, m in metric.items():
            dev_scores[key] = m.evaluate()

        if scheduler is not None:# and args.scheduler_mode == "epoch":
            scheduler.step(epoch + 1)

        if local_rank == 0:
            for idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                summary_writer.add_scalar(
                    "group{}/lr".format(idx), float(lr), global_step=epoch
                )

            summary_writer.add_scalar("loss/train_loss", train_loss, global_step=epoch)
            summary_writer.add_scalar("loss/dev_loss", dev_loss, global_step=epoch)

            for k, score in dev_scores.items():
                summary_writer.add_scalar(f"loss/dev_score_{k}", score, global_step=epoch)

            for k, score in test_scores.items():
                summary_writer.add_scalar(f"loss/test_score_{k}", score, global_step=epoch)

            score = test_scores["ov"]

            torch.cuda.empty_cache()

            if score > best_score:
                notify(
                    checkpoint_dir.name,
                    f"epoch {epoch}: new score {score:.3f} (old {best_score:.3f}, diff {abs(score - best_score):.3f})",
                    int(100 * (epoch / args.num_epochs)),
                )
                best_score = score

                saver(checkpoint_dir / "model_best.pth", best_score)
                save_jit(model, args, checkpoint_dir / f"model_best.pt")
                #if hasattr(model, "module"):
                #    torch.save(model.module, checkpoint_dir / f"modelo_best.pth")
                #else:
                #    torch.save(model, checkpoint_dir / f"modelo_best.pth")

            saver(checkpoint_dir / "model_last.pth", score)
            save_jit(model, args, checkpoint_dir / "model_last.pt")
            #if hasattr(model, "module"):
            #    torch.save(model.module, checkpoint_dir / f"modelo_last.pth")
            #else:
            #    torch.save(model, checkpoint_dir / f"modelo_last.pth")

            if epoch % (2 * args.T_max) == (args.T_max - 1):
                saver(checkpoint_dir / f"model_last_{epoch + 1}.pth", score)
                save_jit(model, args, checkpoint_dir / f"model_last_{epoch + 1}.pt")

        torch.cuda.empty_cache()

    if local_rank == 0:
        summary_writer.close()

        notify(
            checkpoint_dir.name,
            f"finished training with score {score:.3f} (best {best_score:.3f}) on epoch {epoch}",
        )

    return res


def build_model(args):
    if "dino" in args.backbone:
        model = dino.Unet(
            backbone_name=args.backbone,
            img_size=args.img_size,
            n_classes=args.n_classes,
            decoder_channels=args.dec_channels,
        )
        #for param in model.backbone.parameters():
        #    param.requires_grad = False
    else:
        model = models.Unet(args)
    #cfg = model.pretrained_cfg
    #mean = cfg["mean"]
    #std = cfg["std"]
    #normalizer = A.Normalize(mean=mean, std=std)

    return model#, normalizer


def build_criterion(args):
    if args.loss == "bce":
        #criterion = nn.BCEWithLogitsLoss()#reduction="none")
        criterion = BCE()
    else:
        raise NotImplementedError(f"not yet implemented {args.loss}")

    return criterion


def build_optimizer(parameters, args):
    if args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "fusedadam":
        optimizer = apex.optimizers.FusedAdam(
            parameters,
            adam_w_mode=True,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "fusedsgd":
        optimizer = apex.optimizers.FusedSGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"not yet implemented {args.optim}")

    return optimizer


def build_scheduler(optimizer, args, n=1):
    scheduler = None

    if args.scheduler.lower() == "cosa":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max * n,
            #eta_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 5e-6),
            eta_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-2, 5e-6),
        )
    elif args.scheduler.lower() == "cosawr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_max,
            T_mult=1,
            eta_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 5e-6),
        )
    elif args.scheduler.lower() == "wucos":
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer,
            t_initial=args.T_max * n,
            #lr_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 1e-5),
            #warmup_lr_init=args.learning_rate * 1e-2,
            warmup_t=args.warmup_t,  # int(args.warmup_steps_ratio * args.num_epochs) * n,
            cycle_limit=1,  # args.T_max + 1,
            t_in_epochs=n == 1,
        )
    else:
        print("No scheduler")

    return scheduler


def main():
    args = parse_args()

    train(args)
    # while train(args):
        #continue


if __name__ == "__main__":
    main()
