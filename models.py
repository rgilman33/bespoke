from constants import *
from imports import *


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

    def cnn_features(self, x, aux):
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
            x = checkpoint_sequential(self.backbone.blocks, 7, x)
            x = self.backbone.conv_head(x)
            x = self.backbone.bn2(x)
            x = self.backbone.act2(x)
            x = self.backbone.global_pool(x)
            x = self.backbone.classifier(x)
        
        # unpack seq and batch
        x = x.reshape(bs, bptt, self.backbone_out)

        # cat in aux model (speed, has_maps, has_route), this still has calib params in it
        x = torch.cat([x, aux[:,:,self.AUX_MODEL_IXS]], dim=-1)

        return x

    def rnn_head(self, z):
        z = self.fcs1_rnn(z)

        #if self.training and not self.is_for_viz: x = dropout_no_rescale(x, p=.2) 
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