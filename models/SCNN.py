import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import random

epsilon = 0.01

        
def SeasonalNorm(x, period_length):
    b, c, n, t = x.shape
    x_period = torch.split(x, split_size_or_sections=period_length, dim=-1)
    x_period = torch.stack(x_period, -2)

    mean = x_period.mean(3)
    var = (x_period ** 2).mean(3) - mean ** 2 + 0.00001

    mean = mean.repeat(1, 1, 1, t // period_length)
    var = var.repeat(1, 1, 1, t // period_length)

    mean = F.pad(mean.reshape(b * c, n, -1), mode='circular', pad=(t % period_length, 0)).reshape(b, c, n, -1)
    var = F.pad(var.reshape(b * c, n, -1), mode='circular', pad=(t % period_length, 0)).reshape(b, c, n, -1) 
    out = (x - mean) / (var + epsilon) ** 0.5

    return out, mean, var ** 0.5
    

class AdaSpatialNorm(nn.Module):
    def __init__(self, embedding_dim, num_nodes):
        super(AdaSpatialNorm, self).__init__()
        self.node_embedding = nn.Parameter(torch.zeros(num_nodes, embedding_dim))
        
    def forward(self, x):
        b, c, n, t = x.shape
        
        adj_mat = torch.matmul(self.node_embedding, self.node_embedding.T)
        adj_mat = adj_mat - 10 * torch.eye(n).cuda()
        adj_mat = torch.softmax(adj_mat, dim=-1)

        adj_mat = adj_mat.unsqueeze(0)
        x_f = x.permute(0, 3, 2, 1).reshape(b * t, -1, c)
        
        mean_f = torch.matmul(adj_mat, x_f)
        var_f = torch.matmul(adj_mat, x_f ** 2) - mean_f ** 2 + 0.00001
          
        mean = mean_f.view(b, t, n, c).permute(0, 3, 2, 1)
        var = var_f.view(b, t, n, c).permute(0, 3, 2, 1)        
        
        out = (x - mean) / (var + epsilon) ** 0.5

        return out, mean, var ** 0.5


def PeriodNorm(x, period_len):
    b, c, n, t = x.shape
    x_patch = [x[..., period_len-1-i:-i+t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    var = (x_patch ** 2).mean(4) - mean ** 2 + 0.00001
    mean = F.pad(mean.reshape(b * c, n, -1), mode='replicate', pad=(period_len-1, 0)).reshape(b, c, n, -1)
    var = F.pad(var.reshape(b * c, n, -1), mode='replicate', pad=(period_len-1, 0)).reshape(b, c, n, -1)
    out = (x - mean) / (var + epsilon) ** 0.5

    return out, mean, var ** 0.5


class ResidualExtrapolate(nn.Module):
    def __init__(self, d_model, input_len, output_len):
        super(ResidualExtrapolate, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.regreesor = nn.Conv2d(in_channels=d_model, out_channels=d_model * output_len, kernel_size=(1, input_len))
        
    def forward(self, x):
        b, c, n, t = x.shape
        proj = self.regreesor(x[..., -self.input_len:]).reshape(b, -1, c, n).permute(0, 2, 3, 1)
        x_proj = torch.cat([x, proj], dim=-1)
        
        return x_proj


def SeasonalExtrapolate(x, cycle_len, pred_len, cycle_num):
    weight = torch.zeros(pred_len // cycle_len + 1, cycle_num).cuda()
    weight = torch.softmax(weight, dim=-1)  
    b, c, n, t = x.shape 
    x_cycle = torch.split(x, split_size_or_sections=cycle_len, dim=-1)
    x_cycle = torch.stack(x_cycle, -1)
    proj_cycle = torch.matmul(weight, x_cycle.permute(0, 2, 3, 4, 1))
    x_proj = torch.cat([x_cycle.permute(0, 2, 3, 4, 1), proj_cycle], dim=-2).permute(0, 4, 1, 3, 2).reshape(b, c, n, -1)[...,: t + pred_len]

    return x_proj


def ConstantExtrapolate(x, num_pred):
    b, c, n, t = x.shape
    x_proj = F.pad(x.reshape(b * c, n, -1), mode='replicate', pad=(0, num_pred)).reshape(b, c, n, -1)
    return x_proj


class EncoderLayer(nn.Module):
    def __init__(self, d_model, seq_len, pred_len, cycle_len, short_period_len, series_num, kernel_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.kernel_size = kernel_size
        self.pred_len = pred_len
        self.series_num = series_num
        self.cycle_len = cycle_len
        self.short_period_len = short_period_len
        self.seq_len = seq_len
        
        self.spatial_norm = AdaSpatialNorm(d_model, series_num)
        self.residual_extrapolate_1 = ResidualExtrapolate(d_model, short_period_len, pred_len)
        self.residual_extrapolate_2 = ResidualExtrapolate(d_model, short_period_len, pred_len)
        self.residual_extrapolate_3 = ResidualExtrapolate(d_model, short_period_len, pred_len)
        self.residual_extrapolate_4 = ResidualExtrapolate(d_model, short_period_len, pred_len)
        
        self.conv_1 = nn.Conv2d(in_channels=13 * d_model, out_channels=d_model, kernel_size=(1, kernel_size), dilation=1)
        self.conv_2 = nn.Conv2d(in_channels=13 * d_model, out_channels=d_model, kernel_size=(1, kernel_size), dilation=1)
        
        self.skip_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.scale_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.residual_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        b, c, n, t = x.shape
        residual = x
        xs = []
        
        x_proj = ConstantExtrapolate(x, self.pred_len)
        xs.append(x_proj)
        
        x, long_term_mean, long_term_std = PeriodNorm(x, self.seq_len)
        x_proj = self.residual_extrapolate_1(x)
        long_term_mean_proj, long_term_std_proj = ConstantExtrapolate(long_term_mean, self.pred_len), ConstantExtrapolate(long_term_std, self.pred_len)
        xs.extend([x_proj, long_term_mean_proj, long_term_std_proj])
        
        x, season_mean, season_std = SeasonalNorm(x, self.cycle_len)
        x_proj = self.residual_extrapolate_2(x)
        season_mean_proj, season_std_proj = SeasonalExtrapolate(season_mean, self.cycle_len, self.pred_len, self.seq_len // self.cycle_len), SeasonalExtrapolate(season_std, self.cycle_len, self.pred_len, self.seq_len // self.cycle_len)
        xs.extend([x_proj, season_mean_proj, season_std_proj])
        
        x, short_term_mean, short_term_std = PeriodNorm(x, self.short_period_len)
        x_proj = self.residual_extrapolate_3(x)
        short_term_mean_proj, short_term_std_proj = ConstantExtrapolate(short_term_mean, self.pred_len), ConstantExtrapolate(short_term_std, self.pred_len)
        xs.extend([x_proj, short_term_mean_proj, short_term_std_proj])
        
        x, spatial_mean, spatial_std = self.spatial_norm(x)
        x_proj = self.residual_extrapolate_4(x)
        spatial_mean_proj, spatial_std_proj = ConstantExtrapolate(spatial_mean, self.pred_len), ConstantExtrapolate(spatial_std, self.pred_len)
        xs.extend([x_proj, spatial_mean_proj, spatial_std_proj])       

        x = torch.cat(xs, dim=1)        
        x = F.pad(x, mode='constant', pad=(self.kernel_size-1, 0))

        x_1 = torch.tanh(self.conv_1(x))
        x_2 = torch.sigmoid(self.conv_2(x))
        
        x_z = (x_1 * x_2)[..., :-self.pred_len]
        pred_z = (x_1 * x_2)[..., -self.pred_len:]
        s = self.skip_conv(pred_z)
        x_z = self.residual_conv(x_z)

        return x_z, s


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_layers = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=configs.d_model,
                                    kernel_size=1)
        self.e_layers = configs.e_layers
        for i in range(configs.e_layers):
            self.enc_layers.append(EncoderLayer(configs.d_model, configs.seq_len, configs.pred_len, configs.cycle_len, configs.short_period_len, configs.enc_in, configs.kernel_size, dropout=0.1))
        self.end_conv = nn.Conv1d(in_channels=self.pred_len * configs.d_model,
                                  out_channels=self.pred_len,
                                  groups=self.pred_len,
                                  kernel_size=1,
                                  bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        input = x_enc.permute(0, 2, 1).unsqueeze(1)
        in_len = input.size(3)
        x = self.start_conv(input)
        
        b, c, n, L = x.shape
                
        out = 0
        s = 0
        
        for i in range(self.e_layers):
            residual = x
            x, s = self.enc_layers[i](x)
            x = x + residual
            out = s + out
        out = out.permute(0, 3, 1, 2).reshape(b, -1, n)
        dec_out = self.end_conv(out)
        
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
        
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                print(name, param.shape)
                pass


