from constants import *
from imports import *

import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        ff_mult = 4
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model*ff_mult),
            nn.ReLU(),
            nn.Linear(d_model*ff_mult, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x, need_weights=False) #, TODO is_causal=True)  # Self-Attention
        x = x + attn_output  # Add
        x = self.norm1(x)  # Normalize
        ff_output = self.feedforward(x)  # Feedforward
        x = x + ff_output  # Add
        x = self.norm2(x)  # Normalize
        return x

class PosEmb(nn.Module):
    def __init__(self, n_tokens, d_model):
        super(PosEmb, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, d_model, n_tokens))
    
    def forward(self, x):
        # bs,d_model,n_tokens
        batch_size, _, _ = x.size()
        positional_embeddings = self.position_embeddings.expand(batch_size, -1, -1)
        return x + positional_embeddings

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        d_model = 128
        nhead = 4
        n_blocks = 8
        input_len = 540
        transf_in_len = 1024
        n_pad_tokens = transf_in_len - input_len
        self.down_conv = nn.Sequential(
            nn.Conv2d(448, d_model, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )
        self.pad_tokens = nn.Parameter(torch.zeros(1, d_model, n_pad_tokens))
        self.pos_emb = PosEmb(transf_in_len, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(n_blocks)
        ])
        self.bev_head = BevHead()

    def forward(self, x):
        # bs,448,12,45
        x = self.down_conv(x)
        # print("down conv", x.shape)
        # bs,128,12,45
        bs,c,h,w = x.shape
        x = x.reshape(bs,c,h*w) #.contiguous() 
        # print("reshape", x.shape)
        # bs,128,540
        x = torch.cat([x, self.pad_tokens.expand(bs,-1,-1)], dim=-1)
        # print("pad tokens", x.shape)
        x = self.pos_emb(x)
        # print("pos emb", x.shape)
        x = x.permute(0,2,1) # swap channels and 'seq'
        # print("permute", x.shape)
        for transformer_block in self.transformer_blocks:
            #x = transformer_block(x)
            x = torch.utils.checkpoint.checkpoint(transformer_block, x, use_reentrant=False)
            # print("t block", x.shape)
        x = x.reshape(bs, 32,32, c).permute(0,3,1,2)
        # bs,d_model,32,32 # back to channels first for bev upsample
        x = self.bev_head(x)

        return x


class BevHead(nn.Module):
    def __init__(self):
        super(BevHead, self).__init__()
        self.up = nn.Sequential(
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1),
        )
    def forward(self, x):
        # print("bev up", x.shape)
        x = self.up(x)
        return x


class SemsegPerspective(nn.Module):
    def __init__(self):
        super(SemsegPerspective, self).__init__()

        self.up1 = nn.Sequential(
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(448, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.resmix1 = nn.Sequential(
            nn.Conv2d(64+56, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.resmix2 = nn.Sequential(
            nn.Conv2d(32+32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.up4 = nn.Sequential(
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.resmix3 = nn.Sequential(
            nn.Conv2d(32+24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.up5 = nn.Sequential(
            # 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 6, kernel_size=1, stride=1),
        )

    def forward(self, b0_out, b1_out, b2_out, x):
        # expects x sz (batch, 448, 12, 45)
        # b0 # 24, 180, 720
        # b1 # 32, 90, 360
        # b2 # 56, 45, 180
        assert x.shape[1:] == torch.Size([448, 12, 45])

        x = self.up1(x)# 64, 48, 180
        x = F.interpolate(x, size=(45, 180), mode='bilinear', align_corners=False) # squeeze down slightly to match input for resmix
        # 64, 45, 180

        x = torch.cat([b2_out, x], dim=1) # cat channelwise
        x = self.resmix1(x) # 64, 45, 180

        x = self.up3(x) # 32, 90, 360

        x = torch.cat([b1_out, x], dim=1) # cat channelwise
        x = self.resmix2(x) # 32, 90, 360

        x = self.up4(x) #  32, 180, 720 

        x = torch.cat([b0_out, x], dim=1) # cat channelwise
        x = self.resmix3(x) # 32, 180, 720

        x = self.up5(x) # 3, 360, 720

        return x
    
# b3 is 1536 features, 14M params or so
# b4 is 1792 features, 21M params
# b5 is 2048 features, 32M params, don't seem to be any weights for models this size and above, actually there are, under the 'tf' prefix

class Finisher(nn.Module):
    def __init__(self, n):
        super(Finisher, self).__init__()
        self.wps_head = nn.Linear(n, N_WPS_TARGETS)
        self.aux_targets_head = nn.Linear(n, len(AUX_TARGET_PROPS))
        self.obsnet = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, len(OBSNET_PROPS)))

    def forward(self, x):
        obsnet_out = self.obsnet(x.detach())
        wps_preds = self.wps_head(x)
        aux_preds = self.aux_targets_head(x)
        return wps_preds, aux_preds, obsnet_out

class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, in_chans=N_CHANNELS_MODEL).to(device)
        # self.backbone = timm.create_model("tf_efficientnet_b6", pretrained=True, in_chans=N_CHANNELS_MODEL).to(device)
        self.backbone.classifier = nn.Identity()
        self.backbone_out = self.backbone.num_features #b3 is 1536
        
        # common cnn feature extractor
        n = self.backbone_out + len(AUX_MODEL_PROPS)

        self.semseg_perspective_head = SemsegPerspective().to(device)
        self.bev_head = Transformer().to(device)

        self.inner_dim = 1024

        self.fcs1_rnn = nn.Sequential(nn.Linear(n, self.inner_dim), nn.ReLU())
        self.fcs1_cnn = nn.Sequential(nn.Linear(n, self.inner_dim), nn.ReLU())
        
        # finishers are same structure
        self.cnn_finisher = Finisher(self.inner_dim)
        self.rnn_finisher = Finisher(self.inner_dim)

        # rnn
        self.rnn = nn.LSTM(self.inner_dim, self.inner_dim, 1, batch_first=True)
        self.h, self.c = None, None

        # Create parameters for learnable hidden state and cell state
        self.hidden_init = nn.Parameter(torch.zeros((1,1,self.inner_dim)), requires_grad=True)
        self.cell_init = nn.Parameter(torch.zeros((1,1,self.inner_dim)), requires_grad=True)
        # initialize hidden and cell states with xavier uniform
        nn.init.xavier_uniform_(self.hidden_init)
        nn.init.xavier_uniform_(self.cell_init)

        self.use_rnn = True

        self.AUX_MODEL_IXS = torch.LongTensor(AUX_MODEL_IXS); self.AUX_CALIB_IXS = torch.LongTensor(AUX_CALIB_IXS)
    
        self.backbone_is_trt = False

        self.is_for_viz = False
        self.activations, self.gradients = None, None 
        self.grads = {}
        self.acts = {}
        self.viz_ix = 5

        self.carousel = False

    def copy_cnn_params_to_rnn(self):
        for p1, p2 in zip(self.cnn_finisher.parameters(), self.rnn_finisher.parameters()):
            p2.data = p1.data.clone()
        for p1, p2 in zip(self.fcs1_cnn.parameters(), self.fcs1_rnn.parameters()):
            p2.data = p1.data.clone()

    def set_for_viz(self):
        # has to be called AFTER load state dict
        self.is_for_viz = True
        self.backbone = self.backbone.as_sequential()

    def load_trt_backbone(self):
        import torch_tensorrt
        self.trt_backbone = torch.jit.load(TRT_MODEL_PATH).to(device)
        self.backbone_is_trt = True

    def reset_hidden(self, bs):
        # self.h = torch.zeros((1,bs,self.inner_dim)).half().to(device)
        # self.c = torch.zeros((1,bs,self.inner_dim)).half().to(device)
        self.h = self.hidden_init.expand(1, bs, self.inner_dim).contiguous()
        self.c = self.cell_init.expand(1, bs, self.inner_dim).contiguous()

    def reset_hidden_carousel(self, bs): #NOTE, only used for inference, bptt always one, bs always one
        # self.h = torch.zeros((1,bs,self.inner_dim)).half().to(device)
        # self.c = torch.zeros((1,bs,self.inner_dim)).half().to(device)
        self.h_carousel = [self.hidden_init.expand(1, bs, self.inner_dim).contiguous().clone() for _ in range(10)]
        self.c_carousel = [self.cell_init.expand(1, bs, self.inner_dim).contiguous().clone() for _ in range(10)]
        self.carousel = True
        self.carousel_ix = 0
        print("resetting hidden carousel")

    def get_hook(self, name):
        EPS = .99
        def hook(grad):
            # if name in self.grads.keys():
            #     self.grads[name] = self.grads[name]*EPS + grad.detach().cpu().numpy()*(1-EPS)
            # else:
            #     self.grads[name] = grad.detach().cpu().numpy()
            self.grads[name] = grad.detach().cpu().numpy()
        return hook
        
    def activations_hook(self, grad):
        self.gradients = grad.detach().cpu()

    def cnn_features(self, x, aux, return_blocks_out=False):
        # flatten batch and seq for CNNs
        bs, bptt, c, h, w = x.shape
        x = x.reshape(bs*bptt,c,h,w).contiguous() 

        if self.is_for_viz: 
            EPS = .99
            # viz
            for i, mm in enumerate(list(self.backbone)):
                # Run through the model
                x = mm(x)
                name = i
                print(f"{i} isnan:{x.isnan().sum().item()} max:{x.max().item()} min:{x.min().item()} std:{x.std().item()} mean:{x.mean().item()} {x.shape}")
                if i==self.viz_ix:
                    # Store the activations
                    self.acts[name] = x.detach().cpu().numpy()
                    # Instruct to store the gradients if necessary
                    x.register_hook(self.get_hook(name))

        elif self.backbone_is_trt:
            # inference
            x = self.trt_backbone(x)
        else: 
            # trn
            x = self.backbone.conv_stem(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)

            # backbone_out = checkpoint_sequential(self.backbone.blocks, 7, x)

            # # This works, brings memory down to 11gb. Keep this in back pocket for when need it. Check this again.
            c = 0
            b0_out, b1_out, b2_out = None,None,None
            # print("\n\n\n")
            for block in self.backbone.blocks:
                for module in block: # each block is also a sequential module
                    x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False) # False is recommended

                    name = type(module).__name__
                    # print(name, x.shape)
                # print("block end", c, x.shape, "\n")
                if c==0: b0_out = x # 24, 180, 720
                if c==1: b1_out = x # 32, 90, 360
                if c==2: b2_out = x # 56, 45, 180
                c+=1
            backbone_out = x
            # print("\n\n\n\n")

            # shape here is (bs*bptt, 448, 12, 45)
            # branching off here for bev semseg

            x = self.backbone.conv_head(backbone_out) # just an upsamping 1x1 conv: Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)
            x = self.backbone.bn2(x)
            x = self.backbone.act2(x) # shape coming out of here is (1792, 12, 45) w img size 360 x 1440. This is ~ 1m activations.
            x = self.backbone.global_pool(x)
            x = self.backbone.classifier(x)
        
        # unpack seq and batch
        x = x.reshape(bs, bptt, self.backbone_out)

        # cat in aux model (speed, has_maps, has_route), this still has calib params in it
        x = torch.cat([x, aux[:,:,self.AUX_MODEL_IXS]], dim=-1)

        if return_blocks_out:
            # semseg_perspective_p = self.semseg_perspective_head(b0_out, b1_out, b2_out, backbone_out)
            semseg_perspective_p = torch.utils.checkpoint.checkpoint(self.semseg_perspective_head, 
                                                                     b0_out, b1_out, b2_out, backbone_out, use_reentrant=False)
            SEMSEG_CHANNELS = 6
            semseg_perspective_p = semseg_perspective_p.reshape(bs, bptt, SEMSEG_CHANNELS, SEMSEG_PERSP_H, SEMSEG_PERSP_W)

            bev = self.bev_head(backbone_out)
            bev = bev.reshape(bs,bptt,3,BEV_HEIGHT,BEV_WIDTH)
            return x, bev, semseg_perspective_p
        
        # if return_blocks_out:
        #     semseg_p = self._deconv_prep(backbone_out) # outputs (bs*bptt, 1792, 16, 16) 
        #     semseg_p = self._deconv(semseg_p) # outputs (bs*bptt, 3, 64, 64)
        #     semseg_p = semseg_p.reshape(bs, bptt, 3, BEV_HEIGHT, BEV_WIDTH)
        #     return x, semseg_p
        else:
            return x
    

    def rnn_head(self, z):
        z = self.fcs1_rnn(z)

        #if self.training and not self.is_for_viz: x = dropout_no_rescale(x, p=.2) 
        if self.carousel: #NOTE only used for inference. Awkward. Bptt always one.
            cc = self.carousel_ix % 10

            test_ix = 0
            if cc==test_ix: 
                print(self.carousel_ix)
                print(self.h_carousel[test_ix][0,0,:3])

            x, h_c = self.rnn(z, (self.h_carousel[cc], self.c_carousel[cc]))
            self.h_carousel[cc], self.c_carousel[cc] = h_c[0].detach().contiguous(), h_c[1].detach().contiguous()

            if cc==test_ix: 
                print(self.h_carousel[test_ix][0,0,:3])
                print("\n")

            self.carousel_ix += 1
        else:
            x, h_c = self.rnn(z, (self.h, self.c))
            self.h, self.c = h_c[0].detach().contiguous(), h_c[1].detach().contiguous()

        wps_preds, aux_preds, obsnet_out = self.rnn_finisher(x)
        return wps_preds, aux_preds, obsnet_out

    def cnn_head(self, z):
        z = self.fcs1_cnn(z)

        wps_preds, aux_preds, obsnet_out = self.cnn_finisher(z)
        return wps_preds, aux_preds, obsnet_out

    def forward_cnn(self, x, aux):
        x = self.cnn_features(x, aux)
        return self.cnn_head(x)

    def forward_rnn(self, x, aux):
        x = self.cnn_features(x, aux)
        return self.rnn_head(x)

    def forward(self, x, aux):
        return self.forward_rnn(x, aux) if self.use_rnn else self.forward_cnn(x, aux)


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def try_load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        try:
            own_state[name].copy_(param)
            #print(f"loaded {name}")
        except:
            try:
                print(f"could not fully load {name}, patching what we can")
                s = own_state[name].shape
                sp = param.shape
                print(s, sp)

                if len(s)==4:
                    # conv layer
                    m = min(s[1], sp[1])
                    own_state[name][:,:m, :,:].copy_(param[:,:m, :,:])
                else:
                    param_n_out = param.shape[0] # assumes loaded weights have fewer than new model, ie we've added something to final layer
                    own_state[name][:param_n_out].copy_(param)
            except:
                print(f"could not load {name}")
                continue
    
    model.load_state_dict(own_state, strict=False)
    return model

"""
# Some bespoke model surgery. Loading our prev wps, aux_targets and obsnet weights into the new ones prefixed w "finisher"

saved_sd = torch.load(f"{BESPOKE_ROOT}/models/m.torch")
sd = m.state_dict()

for name, param in sd.items():
    if "finisher" in name:
        print(name)
        nn = name[13:]; print(nn)
        print(sd[name].shape, saved_sd[nn].shape, "\n\n")
        sd[name].copy_(saved_sd[nn])
m.load_state_dict(sd)

sd["rnn_finisher.wps_head.weight"]
saved_sd["wps_head.weight"]
"""

def add_noise(activations, std=.1):
    noise = torch.randn(activations.shape, dtype=activations.dtype).to(device) * std
    noise += 1 # centered at one w std of std
    return activations * noise

def dropout_no_rescale(activations, p=.2): 
    mask = (torch.rand_like(activations) > p).half()
    return activations * mask