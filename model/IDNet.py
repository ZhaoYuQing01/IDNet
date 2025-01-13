from einops import rearrange
import torch.nn as nn
import torch
Conv2d = nn.Conv2d
from .EDMoE import EDMoE
from .Transformer import ICTransformerBlock, SSICTransformerBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)



class IDnet(nn.Module):
    def __init__(self, classes, dim, patch, num_experts,k, hsi_inchannel=30):
        super(IDnet, self).__init__()
        self.down1=Downsample(dim)
        self.down2=Downsample(dim)
        self.EDMoE=EDMoE(int(dim//2),num_experts,k)
        self.convhs = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
        )
        self.convhs2 = nn.Sequential(
            nn.Conv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            nn.AvgPool3d(2),
            nn.Dropout(0.5),
        )
        self.conv1_1 = nn.Conv3d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
        )
        self.conv1_2 = nn.Conv3d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=(5, 1, 1),
            stride=(1, 1, 1),
            padding=(2, 0, 0),
        )
        self.conv1_3 = nn.Conv3d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=(7, 1, 1),
            stride=(1, 1, 1),
            padding=(3, 0, 0),
        )
        self.conv1_4 = nn.Conv3d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=(9, 1, 1),
            stride=(1, 1, 1),
            padding=(4, 0, 0),
        )
        
        self.convlidar = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.convlidar2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(),   
            nn.AvgPool2d(2),
            nn.Dropout(0.5),   
        )

        self.conv2_1 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv2_3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.conv2_4 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim//4,
            kernel_size=9,
            stride=1,
            padding=4,
        )
                
        self.convhs3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hsi_inchannel*8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.transformer1 = ICTransformerBlock(in_dim=dim,pyramid_levels=3, token_mlp='ffn')
        self.transformer2 = ICTransformerBlock(in_dim=dim*2, pyramid_levels=2, token_mlp='ffn')
        
        self.transformer = SSICTransformerBlock(in_dim=dim, hsi_inchannel=hsi_inchannel//2, pyramid_levels=3, token_mlp='mix_skip')
        

        self.encoder_pos_embed1 = nn.Parameter(torch.randn(1, dim, patch//2, patch//2))
        self.encoder_pos_embed2 = nn.Parameter(torch.randn(1, dim*2, patch//4, patch//4))
        
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, hsi_inchannel//2, (patch//2)**2, dim))
        
        self.dropout = nn.Dropout(0.01)

        self.w1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.w2 = torch.nn.Parameter(torch.Tensor([0.5]))
        
        self.bn1 = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(dim)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=dim*2,
                out_channels=dim*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(8*(patch//4)**2),
            nn.Linear(8*(patch//4)**2, classes),
        )
        

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x1 = self.convhs(x1)
        x1 = torch.cat((self.conv1_1(x1), self.conv1_2(x1), self.conv1_3(x1), self.conv1_4(x1)), dim=1)
        x1=self.bn1(x1)
        x1=self.relu(x1)
        x1=self.convhs2(x1)
        x1 = x1.reshape(x1.shape[0], x1.shape[2], x1.shape[3]*x1.shape[4], x1.shape[1])
        x1 = x1 + self.encoder_pos_embed
        x1 = self.dropout(x1)
        x1 = self.transformer(x1, H=int(x1.shape[2] ** 0.5), W=int(x1.shape[2] ** 0.5))
        x1 = x1.reshape(x1.shape[0], x1.shape[3], x1.shape[1], int(x1.shape[2] ** 0.5), int(x1.shape[2] ** 0.5))
        x1 = rearrange(x1, "b c h w y ->b (c h) w y")
        x1 = self.convhs3(x1)
        x1 = x1 + self.encoder_pos_embed1
        x1 = self.dropout(x1)
        x1 = self.transformer1(x1)
        x1= self.down1(x1)
        x1 = x1 + self.encoder_pos_embed2
        x1 = self.dropout(x1)
        x1 = self.transformer2(x1)

        x2 = self.convlidar(x2)
        x2 = torch.cat((self.conv2_1(x2), self.conv2_2(x2), self.conv2_3(x2), self.conv2_4(x2)), dim=1)
        x2=self.bn2(x2)
        x2=self.relu(x2)
        x2=self.convlidar2(x2)
        x2 = x2 + self.encoder_pos_embed1
        x2 = self.dropout(x2)
        x2 = self.transformer1(x2)
        x2=self.down2(x2)
        x2 = x2 + self.encoder_pos_embed2
        x2 = self.dropout(x2)
        x2 = self.transformer2(x2)

        x=x1*self.w1+x2*self.w2
        x=self.conv(x)
        x=self.EDMoE(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x
