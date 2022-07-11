import os
import torch

def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])

    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)