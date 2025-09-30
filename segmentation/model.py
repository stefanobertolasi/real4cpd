import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class SE2d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.leaky_relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s
    
class ChannelNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # x: [N, C, P, W]
        mean = x.mean(dim=-1, keepdim=True)        
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)   
        
        if self.affine:
            x_hat = x_hat * self.gamma + self.beta
        return x_hat
    
class TemporalBlockSep2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride, dilation, dropout, groups):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2 
        self.dw = nn.Conv2d(input_size, input_size, kernel_size=(1, kernel_size),
                            stride=1, padding=(0, padding),
                            dilation=(1, dilation),
                            groups=input_size, bias=False)
        self.pw = nn.Conv2d(input_size, hidden_size, kernel_size=1, bias=False)
        # self.norm = nn.GroupNorm(num_groups=groups, num_channels=hidden_size, affine=True)
        self.norm = ChannelNorm(num_channels=hidden_size, affine=True)
        # self.norm = nn.BatchNorm2d(num_features=hidden_size)
        self.act = nn.LeakyReLU()
        self.se = SE2d(hidden_size, reduction=8)
        self.drop = nn.Dropout2d(p=dropout)
        self.res = nn.Conv2d(input_size, hidden_size, kernel_size=1, bias=False) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        y = self.se(y)
        y = self.drop(y)
        return y + self.res(x)

class TCNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_inputs = 10
        num_channels = [64, 128, 128, 64]
        dropout = args.dropout
        kernel_size = 5
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlockSep2d(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, dropout=dropout, groups=16))
        self.network = nn.Sequential(*layers)

        self.to1 = nn.Conv2d(num_channels[-1], 1, kernel_size=1, bias=True)

        self.optimizer = torch.optim.AdamW(self.parameters(), 
                                           lr=args.lr, 
                                           weight_decay=args.weight_decay, 
                                           betas=(args.betas[0], args.betas[1])
                                           )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    factor=0.5,
                                                                    patience=25, 
                                                                    threshold=1e-4,
                                                                    min_lr=5e-6)

    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        x = self.network(state)
        x = self.to1(x)

        x = x.mean(dim=-1, keepdim=True)

        x = x.squeeze(-1).squeeze(1)
        return x, state
    

class GRUModel(nn.Module):
    def __init__(self, args: dict, num_samples: int):
        nn.Module.__init__(self)
        
        self.gru = nn.GRU(
            input_size=10,
            hidden_size=32, 
            batch_first=True, 
            bidirectional=True, 
            dropout=args.dropout,
            dtype=torch.float32
        )
        
        self.fc = nn.Linear(in_features=64, out_features=1, dtype=torch.float32)
        self.optimizer = optim.Adam(params=self.parameters(), lr=args.lr)

    def forward(self, state):

        # state = [batch_size, sequence_len, channels]
        out, _ = self.gru(state)

        out = self.fc(out).squeeze(-1)

        return out, state
    

