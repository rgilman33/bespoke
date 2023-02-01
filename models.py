from constants import *
from imports import *

VIZ_IX = 5 # viz_ix from 3 (granular) to at least 5, maybe more?

# b3 is 1536 features, 14M params or so
# b4 is 1792 features, 21M params
# b5 is 2048 features, 32M params, don't seem to be any weights for models this size and above, actually there are, under the 'tf' prefix

class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, in_chans=N_CHANNELS_MODEL).to(device)
        # self.backbone = timm.create_model("tf_efficientnet_b6", pretrained=True, in_chans=N_CHANNELS_MODEL).to(device)
        self.backbone.classifier = nn.Identity()
        self.backbone_out = self.backbone.num_features #b3 is 1536

        n = self.backbone_out + len(AUX_MODEL_PROPS)
        self.wps_head = nn.Linear(n, N_WPS_TARGETS)
        self.aux_targets_head = nn.Linear(n, len(AUX_TARGET_PROPS))
        self.obsnet = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, len(OBSNET_PROPS)))

        self.activations, self.gradients = None, None 

        self.backbone_is_trt = False

        self.is_for_viz = False
        self.save_backbone_out = False
        self.backbone_out_acts = None
        self.AUX_MODEL_IXS = torch.LongTensor(AUX_MODEL_IXS)
    
        self.grads = {}
        self.acts = {}
        self.viz_ix = 5

    def set_for_viz(self):
        # has to be called AFTER load state dict
        self.is_for_viz = True
        self.backbone = self.backbone.as_sequential()

    def load_trt_backbone(self):
        import torch_tensorrt
        self.trt_backbone = torch.jit.load(TRT_MODEL_PATH).to(device)
        self.backbone_is_trt = True

    def reset_hidden(self, bs):
        pass

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

    def forward(self, x, aux):
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
                #print(f"{i} isnan:{x.isnan().sum().item()} max:{x.max().item()} min:{x.min().item()} std:{x.std().item()} mean:{x.mean().item()} {x.shape}")
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
        if self.save_backbone_out:
            self.backbone_out_acts = x

        aux = aux[:,:,self.AUX_MODEL_IXS]
        x = torch.cat([x, aux], dim=-1) # cat in aux
        #if self.backbone_is_trt: x = x.half()
        wps_preds = self.wps_head(x)
        aux_preds = self.aux_targets_head(x)
        obsnet_out = self.obsnet(x if self.is_for_viz else x.detach())

        return wps_preds, aux_preds, obsnet_out


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
            print(f"could not fully load {name}, patching what we can")
            own_state_n_out = own_state[name].shape[0]
            param_n_out = param.shape[0] # assumes loaded weights have fewer than new model, ie we've added something to final layer
            own_state[name][:param_n_out].copy_(param)
            print(own_state[name].shape, param.shape)
    
    model.load_state_dict(own_state, strict=False)
    return model