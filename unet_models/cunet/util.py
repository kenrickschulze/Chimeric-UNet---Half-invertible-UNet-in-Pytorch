""" Utilities """

import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np


class DWTUpsample(nn.Module):
    """
    Wraps discrete wavelet layer.
    Useful for invertibly converting 1 channels to 4 channels
    as pre-processing step.
    """

    def __init__(self):
        super(DWTUpsample, self).__init__()
        self.dwt = DWTForward(wave="db1")
        self.idwt = DWTInverse(wave="db1")
        self.last_output = None


    def forward(self, x):
        low, high = self.dwt(x)
        self.last_output = torch.cat((low, high[0][:, 0, ...]), dim=1)
        return self.last_output

    def inverse(self, x):
        low, high = torch.split(x, [1, 3], dim=1)
        high = [high[:, None, ...]]
        return self.idwt((low, high))
    
class DWTMultiUpsample(nn.Module):
    """
    Wraps discrete wavelet layer.
    Useful for invertibly converting 1 channels to 4 channels
    as pre-processing step.
    """

    def __init__(self, reorder):
        super(DWTMultiUpsample, self).__init__()
        self.dwt = DWTForward(wave="db1")
        self.idwt = DWTInverse(wave="db1")
        self.last_output = None
        self.reorder = reorder


    def forward(self, x):
        
        self.last_output = torch.cat([self.up(x[:,c,...].unsqueeze(1)) for c in range(x.shape[1])], dim=1)
        # reshape indicies
        if self.reorder:
            ind = np.array(range(x.shape[1]*4)).reshape(x.shape[1],4).transpose().ravel()
            return self.last_output[:,ind,...]
        else:
            return self.last_output
        
    def inverse(self, x):
        
         # invert reshaping 
        if self.reorder:
            ind = np.array(range(x.shape[1])).reshape(4, int(x.shape[1]/4)).transpose().ravel()
            x = x[:,ind,...]
        return torch.cat([self.up_inv(x[:,4*c:4*(c+1),...]) for c in range(int(x.shape[1]/4))], dim=1)

    
    def up_inv(self, x):
        low, high = torch.split(x, [1, 3], dim=1)
        high = [high[:, None, ...]]
        return self.idwt((low, high))

    def up(self, x):
        low, high = self.dwt(x)
        return torch.cat((low, high[0][:, 0, ...]), dim=1)
    

class DWTDownsample(nn.Module):
    """
    Wraps discrete wavelet layer.
    Useful for invertibly converting 4 channels to 1 channels
    as post-processing step.
    """

    def __init__(self):
        super(DWTDownsample, self).__init__()
        self.dwt = DWTForward(wave="db1")
        self.idwt = DWTInverse(wave="db1")
        self.last_output = None


    def forward(self, x):
        low, high = torch.split(x, [1, 3], dim=1)
        high = [high[:, None, ...]]
        self.last_output = self.idwt((low, high))
        return self.last_output

    def inverse(self, x):
        low, high = self.dwt(x)
        return torch.cat((low, high[0][:, 0, ...]), dim=1)


class DWTMultiDownsample(nn.Module):
    """
    Wraps discrete wavelet layer.
    Useful for invertibly converting 4 channels to 1 channels
    as post-processing step.
    """

    def __init__(self, reorder):
        super(DWTMultiDownsample, self).__init__()
        self.dwt = DWTForward(wave="db1")
        self.idwt = DWTInverse(wave="db1")
        self.last_output = None
        self.reorder = reorder


    def forward(self, x):
 
        # reshape indicies
        if self.reorder:
            ind = np.array(range(x.shape[1])).reshape(4, int(x.shape[1]/4)).transpose().ravel()
            x = x[:,ind,...]
        self.last_output  = torch.cat([self.down(x[:,4*c:4*(c+1),...]) for c in range(int(x.shape[1]/4))], dim=1) 
        return self.last_output

    def inverse(self, x):
        x_dwt = torch.cat([self.down_inv(x[:,c,...].unsqueeze(1)) for c in range(x.shape[1])], dim=1)
        if self.reorder:
            ind = np.array(range(x.shape[1]*4)).reshape(x.shape[1],4).transpose().ravel()
            return x_dwt[:,ind,...]
        return x_dwt
    
        
    def down(self, x):
        low, high = torch.split(x, [1, 3], dim=1)
        high = [high[:, None, ...]]
        
        return self.idwt((low, high))

    def down_inv(self, x):
        low, high = self.dwt(x)
        return torch.cat((low, high[0][:, 0, ...]), dim=1)
    
    

    

def stacked_block(block_op, depth: int, k_size, increase_channels=False):
    """Returns function that stacks convolutional blocks"""

    def stacked_block_op(channels):
        if increase_channels:
            operants = nn.ModuleList()
            operants.append(block_op(channels,k_size, out_channels=4*channels))
            operants.extend([block_op(4*channels,k_size) for _ in range(depth-1)])
            return nn.Sequential(*operants)
        
        else:
            return nn.Sequential(*[block_op(channels,k_size) for _ in range(depth)])

    return stacked_block_op

# def stacked_block(block_op, depth: int, k_size):
#     """Returns function that stacks convolutional blocks"""

#     def stacked_block_op(channels):
#         return nn.Sequential(*[block_op(channels,k_size) for _ in range(depth)])

#     return stacked_block_op
