# first one. Gave decent ish results but still not great at all. 
class Deconv(nn.Module):
    def __init__(self, n_in):
        super(Deconv, self).__init__()
        
        self.in_channels = n_in #1792
        b_channels = 256
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=b_channels, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(b_channels)
        
        self.blocks = []
        for _ in range(2):
            self.blocks.append(nn.ConvTranspose2d(in_channels=b_channels, out_channels=b_channels, kernel_size=4, stride=4, padding=0))
            self.blocks.append(nn.BatchNorm2d(b_channels))
            self.blocks.append(nn.ReLU(inplace=True))
            
        self.blocks = nn.Sequential(*self.blocks)
        self.deconv4 = nn.ConvTranspose2d(in_channels=b_channels, out_channels=3, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        bs, bptt, c = x.shape
        x = x.reshape(bs*bptt,c,1,1).contiguous() 
        # x = x.view(-1, self.in_channels, 1, 1)
        
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.blocks(x)
        x = self.deconv4(x) 

        # unpack batch and seq
        x = x.reshape(bs, bptt, 3, 64, 64).contiguous()

        return x
    
# second one, never got anything from it
class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()
        b_dim = 256
        
        # Two upsample blocks w a 1x1 finisher for clf
        
        # Smaller initial kernel to keep params down
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(1792, b_dim, kernel_size=2, stride=2), # no padding by default
            nn.BatchNorm2d(b_dim),
            nn.ReLU()
        )
        block = nn.Sequential(    
            nn.ConvTranspose2d(b_dim, b_dim, kernel_size=2, stride=2), # no padding by default
            nn.BatchNorm2d(b_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[block for _ in range(1)])
        self.final = nn.Sequential( # 1x1 conv to get classes for semseg
            nn.Conv2d(b_dim, 3, kernel_size=1, stride=1),
        )
        
    def forward(self, x):
        # expects (batch, channels, 16, 16)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)

        return x



# third one, decent but xroads not really and never npcs

class DeconvPrep(nn.Module):
    def __init__(self):
        super(DeconvPrep, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.expand = nn.Sequential(
            nn.Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False), # same as the effnet conv_head
            nn.BatchNorm2d(1792),
            nn.ReLU(),
        )
    def forward(self, _in):
        # expects (batch, 448, 12, 45)
        assert _in.shape[1:] == torch.Size([448, 12, 45])

        # slightly stretch so we can manually pool and keep it symmetric
        _in = F.interpolate(_in, size=(16, 48), mode='bilinear', align_corners=False)

        # pool horizontally to output a square. Will be symmetric op
        _in = self.pool(_in)
        # Now we're at size (batch, 448, 16, 16)

        # expand to 1792 channels
        _in = self.expand(_in)
        # now we're at size (batch, 1792, 16, 16)

        return _in

class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()
        b_dim = 256
        
        self.up = nn.Sequential(
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1792, b_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(b_dim),
            nn.ReLU(),
            # another 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(b_dim, b_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(b_dim),
            nn.ReLU(),
        )
        block = nn.Sequential(    
            nn.Conv2d(b_dim, b_dim, kernel_size=3, stride=1, padding=1), # no padding by default
            nn.BatchNorm2d(b_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[block for _ in range(1)])
        self.final = nn.Sequential( # 1x1 conv to get classes for semseg
            nn.Conv2d(b_dim, 3, kernel_size=1, stride=1),
        )
        
    def forward(self, x):
        # expects (batch, channels, 16, 16)
        x = self.up(x)
        x = self.blocks(x)
        x = self.final(x)

        return x