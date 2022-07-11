import sys
import torch
sys.path.append("/app")


    

from cmath import log
from email.policy import default
from pathlib import Path
import os
import psutil
import datetime
import time
from tqdm import tqdm

from prometheus_client import start_http_server

import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data_loader import PoseDataSetHardDisk

import colossalai
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.trainer import Trainer, hooks
from colossalai.nn.metric import Accuracy
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR

from titans.utils import barrier_context

from colossalai.amp import AMP_TYPE

from utils import save_ckpt
from laplacian import Laplacian
from st_gcn import STGCN

def set_random(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('--data_dir', default = "/data/aist_pad_v8_100/", help='data dir')
    parser.add_argument('--pad', type = int, default = 100, help='we use two side pad, pad = 100, means each 2d seq len is 2 * pad + 1')
    parser.add_argument('--batch_size', type = int, default = 32, help='we use two side pad, pad = 100, means each 2d seq len is 2 * pad + 1')
    parser.add_argument('--epochs', type = int, default = 100, help='we use two side pad, pad = 100, means each 2d seq len is 2 * pad + 1')
    parser.add_argument('--check_points', type = str, default = "./check_points/", help='we use two side pad, pad = 100, means each 2d seq len is 2 * pad + 1')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    check_points_dir = "{}/{}".format(args.check_points, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(check_points_dir, exist_ok=True)
    CONFIG = dict(BATCH_SIZE = args.batch_size, NUM_EPOCHS = args.epochs)
    logger = get_dist_logger()  
    colossalai.launch_from_torch(config=CONFIG)
    logger.info("config {}".format(CONFIG))

    with barrier_context():
        train_pose_data_set = PoseDataSetHardDisk(args.data_dir, "train")
    
    train_dataloader = get_dataloader(
        dataset=train_pose_data_set,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

    pad = args.pad
    laplacian = Laplacian(pad)
    # N = 1, C = 2, H/V = 17, W/T =3
    model = STGCN(torch.from_numpy(laplacian.L).to(dtype = torch.float32), 17, 2, 3, pad)
    model.cuda()

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, _, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader
    )

    if gpc.get_global_rank() == 0:
        from prometheus_client import start_http_server
        from prometheus_client import Gauge
        start_http_server(8080)
        loss_guage = Gauge('loss', 'Description of gauge', ["epch", "lr"])

    for epoch in range(gpc.config.NUM_EPOCHS):
        engine.train()
        if gpc.get_global_rank() == 0:
            train_dl = tqdm(train_dataloader)
        else:
            train_dl = train_dataloader
        for i, (targets_3d, inputs_2d, _, _) in enumerate(train_dl):
            inputs_2d[:, :, [1]] = inputs_2d[:, :, [1]] * -1
            targets_3d[:, :, [1]] = targets_3d[:, :, [1]] * -1

            targets_3d, inputs_2d = targets_3d.cuda(), inputs_2d.cuda()
            inputs_2d = inputs_2d.permute(0, 2, 1, 3)
            targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

            engine.zero_grad()
            outputs_3d = engine(inputs_2d)
            train_loss = engine.criterion(outputs_3d, targets_3d)
            engine.backward(train_loss)
            engine.step()
            if gpc.get_global_rank() == 0:
                train_dl.set_description("Epoch {}".format(epoch))
                train_dl.set_postfix(loss=train_loss.item() * 1000, lr=lr_scheduler.get_last_lr()[0])
                train_dl.update()
                loss_guage.labels(epoch, lr_scheduler.get_last_lr()[0]).set(train_loss.item() * 1000)
        lr_scheduler.step()

        save_ckpt({'state_dict': model.state_dict(), 'epoch': epoch + 1}, check_points_dir, suffix = epoch)

if __name__ == '__main__':
    main()
