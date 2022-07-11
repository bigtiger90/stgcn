import sys
import torch
sys.path.append("/app")

from laplacian import Laplacian
from st_gcn import STGCN

if __name__ == "__main__":
    # N = 1, C = 2, H/V = 17, W/T =3
    pad = 100
    laplacian = Laplacian(pad)
    stgcn = STGCN(torch.from_numpy(laplacian.L).to(dtype = torch.float32), 17, 2, 3, pad)
    stgcn.cuda()
    x = torch.ones((1, 17, 2 * pad + 1, 2), dtype = torch.float32).cuda()
    y = stgcn(x)
    print(y.shape)
    
