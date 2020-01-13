import numpy as np
import os
import mmcv

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

def generate_flow(img1, img2):
    """To be implemented, for online flow generation"""
    return None, None

class Trajectory(object):
    def __init__(self):
        super(Trajectory, self).__init__()

    @staticmethod
    def cat(tensors, tensor):
        return torch.cat([tensors, tensor.unsqueeze(2)], dim=2)

    def get_meshgrid(self):
        B, H, W = self.B, self.H, self.W
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        return grid

    def get_vgrid(self, flow):
        grid = self.get_meshgrid()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(self.W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(self.H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        return vgrid

    def init(self, frame):
        B, _, H, W = frame.shape
        self.B, self.H, self.W = B, H, W
        self.frames = frame.unsqueeze(2)

        self.id_grids = torch.arange(1, H * W + 1).view(1, 1, H, W).float().unsqueeze(2)

    def update(self, frame, flow=None, inv_flow=None):
        if flow is None:
            flow, inv_flow = generate_flow(self.frames[:,:,-1], frame)
        self.frames = self.cat(self.frames, frame)
        id_grid = self.id_grids[:,:,-1]
        vgrid = self.get_vgrid(inv_flow)
        warped_flow = F.grid_sample(flow, vgrid)
        consistency_mask = torch.sum((warped_flow + inv_flow) ** 2, dim=1, keepdim=True) <= \
                           torch.sum(0.01 * (warped_flow ** 2 + inv_flow ** 2), dim=1, keepdim=True) + 0.5
        warped_id_grid = F.grid_sample(id_grid, vgrid, mode='nearest')
        max_id = id_grid.max()
        # consistency fails
        n = warped_id_grid[~consistency_mask].numel()
        warped_id_grid[~consistency_mask] = torch.arange(max_id + 1, max_id + n + 1)
        max_id = max_id + n
        # out of border
        n = warped_id_grid[warped_id_grid == 0].numel()
        warped_id_grid[warped_id_grid == 0] = torch.arange(max_id + 1, max_id + n + 1)
        max_id = max_id + n
        embed()
        self.id_grids = self.cat(self.id_grids, warped_id_grid)

if __name__ == '__main__':
    traj = Trajectory()
    flow_path = '/databack1/KITTI/kitti/tracking/training/Flow/0000/000000.flo'
    inv_flow_path = '/databack1/KITTI/kitti/tracking/training/Inv_Flow/0000/000001.flo'
    flow = mmcv.flowread(flow_path)
    inv_flow = mmcv.flowread(inv_flow_path)
    H, W, _ = flow.shape
    flow = torch.tensor(flow).permute(2, 0, 1).unsqueeze(0)
    inv_flow = torch.tensor(inv_flow).permute(2, 0, 1).unsqueeze(0)
    frame = torch.zeros(1, 3, H, W)
    traj.init(frame)
    traj.update(frame, flow, inv_flow)