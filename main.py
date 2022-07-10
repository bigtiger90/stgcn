import sys
import torch
sys.path.append("/app")

from laplacian import Laplacian
from st_gcn import STGCN

if __name__ == "__main__":
    # N = 1, C = 2, H/V = 17, W/T =3
    laplacian = Laplacian(0)
    print(laplacian.L.shape)
    stgcn = STGCN(17, 2, 3, 0)
    x = torch.ones((1, 17, 1, 2), dtype = torch.float32)
    y = stgcn(torch.from_numpy(laplacian.L).to(dtype = torch.float32), x)
    
