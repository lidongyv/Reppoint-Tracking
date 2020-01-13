import torch
import torch.nn.functional as F
import numpy as np

class FlowAug(object):
    def __init__(self, resize, pad, flip):
        self.resize = resize
        self.pad = pad
        self.flip = flip

    def __call__(self, flows, inv_flows):
        flows, inv_flows = self.Resize(flows, inv_flows)
        flows, inv_flows = self.Pad(flows, inv_flows)
        flows, inv_flows = self.Flip(flows, inv_flows)
        return flows, inv_flows

    def Resize(self, flows, inv_flows):
        H, W = flows.shape[-2:]
        size = self.resize
        if size != (H, W):
            flows = F.interpolate(flows, size=size, mode='bilinear')
            flows[:, 0] = flows[:, 0] / W * size[1]
            flows[:, 1] = flows[:, 1] / H * size[0]

            inv_flows = F.interpolate(inv_flows, size=size, mode='bilinear')
            inv_flows[:, 0] = inv_flows[:, 0] / W * size[1]
            inv_flows[:, 1] = inv_flows[:, 1] / H * size[0]

        return flows, inv_flows


    def Pad(self, flows, inv_flows):
        H, W = flows.shape[-2:]
        size = self.pad
        if size != (H, W):
            flows = F.pad(flows, (0, size[1] - W, 0, size[0] - H))
            inv_flows = F.pad(inv_flows, (0, size[1] - W, 0, size[0] - H))

        return flows, inv_flows

    def Flip(self, flows, inv_flows):
        if self.flip:
            flows = torch.flip(flows, dims=(-1,))
            inv_flows = torch.flip(inv_flows, dims=(-1,))
            flows[:, 0] = -flows[:, 0]
            inv_flows[:, 0] = -inv_flows[:, 0]

        return flows, inv_flows