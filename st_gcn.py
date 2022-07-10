import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        in_channels=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, in_channels_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, pad = 0):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels, kernel_size = 2 * pad + 1)
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels, kernel_size = 2 * pad + 1)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        in_channels=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, in_channels=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        print("lfs", lfs.shape)
        t3 = self.temporal2(lfs)
        print("t3", t3.shape)
        return self.batch_norm(t3) 
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    in_channels).
    """

    def __init__(self, num_nodes, in_channels = 2, out_channels = 3, pad = 0):
        """
        :param num_nodes: Number of nodes in the graph.
        :param in_channels: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=in_channels, out_channels=64,
                                 spatial_channels=64, num_nodes=num_nodes, pad = pad)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=64, num_nodes=num_nodes, pad = pad)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64, kernel_size = 2 * pad + 1)
        self.fcn = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size=1)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        in_channels=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        # out2 = self.block2(out1, A_hat)
        # out3 = self.last_temporal(out2)
        # out4 = self.fcn(out3)
        return out1

