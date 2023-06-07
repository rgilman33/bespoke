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


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.feedforward = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Linear(dim_feedforward, d_model)
        # )
        # self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        bs, seq_len, _ = x.shape
        # print("transf input shape", x.shape)
        # mask = (torch.triu(torch.ones(seq_len, seq_len)) != 1).transpose(0, 1).to("cuda")
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to("cuda")
        attn_output, _ = self.attention(x, x, x, need_weights=False, attn_mask=mask) #, TODO is_causal=True)  # Self-Attention
        # print("attn_output shape", attn_output.shape)
        x = x + attn_output  # Add
        #x = self.norm1(x)  # Normalize

        # ff_output = self.feedforward(x)  # Feedforward
        # print("ff_output shape", ff_output.shape)
        # x = x + ff_output  # Add
        #x = self.norm2(x)  # Normalize

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    
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

        self.fcs1_transformer = nn.Sequential(nn.Linear(n, self.inner_dim), nn.ReLU())
        self.fcs1_cnn = nn.Sequential(nn.Linear(n, self.inner_dim), nn.ReLU())
        
        # finishers are same structure
        self.cnn_finisher = Finisher(self.inner_dim)
        self.transformer_finisher = Finisher(self.inner_dim)

        # transformer
        emb_dim = self.inner_dim
        n_head = 8
        n_blocks = 1
        dim_ff = emb_dim*2
        self.transformer = nn.Sequential(*[TransformerBlock(emb_dim, n_head, dim_ff) for _ in range(n_blocks)])
        self.pe = PositionalEncoding(emb_dim)
        self.use_transformer = True

        self.AUX_MODEL_IXS = torch.LongTensor(AUX_MODEL_IXS); self.AUX_CALIB_IXS = torch.LongTensor(AUX_CALIB_IXS)
    
        self.backbone_is_trt = False

        self.is_for_viz = False
        self.activations, self.gradients = None, None 
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

    def copy_cnn_fcs_to_transformer_fcs(self):
        for p1, p2 in zip(self.cnn_finisher.parameters(), self.transformer_finisher.parameters()):
            p2.data = p1.data.clone()
        for p1, p2 in zip(self.fcs1_cnn.parameters(), self.fcs1_transformer.parameters()):
            p2.data = p1.data.clone()

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

            # checkpointing each block separately has same memory requirements as checkpoint_sequential (24gb)
            # For each module 24 gb to 11gb. We could go even lower by checkpointing even finer, though I'm unable to get it, don't understand the implementation
            # doing each module in the list we get memory blowing up

            # x = checkpoint_sequential(self.backbone.blocks, 7, x)

            for block in self.backbone.blocks: # blocks is a sequential module # TODO check this, need to see we get good perf still. Burned here in the past. Also check how much slower it is.
                for module in block: # each block is also a sequential module
                    x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False) # False is recommended

            # for module in get_children(self.backbone.blocks): # this runs out of memory also. Why?
            #     x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)

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

    def transformer_head(self, z):
        z = self.fcs1_transformer(z) 
        z = self.pe(z)
        z = self.transformer(z)
        wps_preds, aux_preds, obsnet_out = self.transformer_finisher(z)
        return wps_preds, aux_preds, obsnet_out

    def cnn_head(self, z):
        z = self.fcs1_cnn(z)

        wps_preds, aux_preds, obsnet_out = self.cnn_finisher(z)
        return wps_preds, aux_preds, obsnet_out

    def forward_cnn(self, x, aux):
        x = self.cnn_features(x, aux)
        return self.cnn_head(x)

    def forward_transformer(self, x, aux):
        x = self.cnn_features(x, aux)
        return self.transformer_head(x)

    def forward(self, x, aux):
        return self.forward_transformer(x, aux) if self.use_transformer else self.forward_cnn(x, aux)


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