from constants import *
from imports import *

class ObserverNet(nn.Module):
    def __init__(self):
        super(ObserverNet, self).__init__()
        act = nn.ReLU()
        not_drop = nn.Identity() #nn.Dropout(0.2)
        self.fcs = nn.Sequential(nn.Linear(512, 256), act, not_drop, nn.Linear(256, 3)) # error, pitch, yaw

    def forward(self, x):
        return self.fcs(x)

VIZ_IX = 5 # viz_ix from 3 (granular) to at least 5, maybe more?

def add_noise(activations, std=.1):
    noise = torch.randn(activations.shape, dtype=activations.dtype).to(device) * std
    noise += 1 # centered at one w std of std
    return activations * noise

def dropout_no_rescale(activations, p=.2): 
    mask = (torch.rand_like(activations) > p).half()
    return activations * mask

# b3 is 1536 features, 14M params or so
# b4 is 1792 features, 21M params
# b5 is 2048 features, 32M params, don't seem to be any weights for models this size and above

class EffNet(nn.Module):
    def __init__(self, model_arch='efficientnet_b3', is_for_viz=False):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model(model_arch, pretrained=True, in_chans=N_CHANNELS).to(device)
        self.backbone.classifier = nn.Identity()

        backbone_out = self.backbone.num_features #b3 is 1536
        act = nn.ReLU()
        self.inner_dim = 512
        self.n_fcs_2_out = 512

        self.fcs_1 = nn.Sequential(nn.Linear(backbone_out + N_AUX_MODEL_IN, self.inner_dim), act)        
        self._rnn = nn.LSTM(self.inner_dim, self.inner_dim, 1, batch_first=True)
        self.fcs_2 = nn.Sequential(nn.Linear(self.inner_dim + N_AUX_CALIB_IN, self.n_fcs_2_out), act)
        self.wps_head = nn.Linear(self.n_fcs_2_out, N_TARGETS)
        self.aux_targets_head = nn.Linear(self.n_fcs_2_out, N_AUX_TARGETS)


        self.hidden_init = nn.Parameter(torch.randn((1,1,self.inner_dim))/10, requires_grad=True)
        self.cell_init = nn.Parameter(torch.randn((1,1,self.inner_dim))/10, requires_grad=True)
        self.use_trainable_hidden = True
        self.h, self.c = None, None

        self.obsnet = ObserverNet()

        self.activations = None # just storing them here so don't have to return them
        self.gradients = None
        self.rnn_activations = None
        self.rnn_gradients = None

        self.is_for_viz = is_for_viz

    def convert_backbone_to_sequential(self):
        self.backbone = self.backbone.as_sequential()

    # def set_random_hidden(self, bs, std=.1):
    #     self.h = torch.randn((1,bs,self.inner_dim)).half().to(device)*std

    def reset_hidden(self, bs):
        self.use_trainable_hidden = True

    def activations_hook(self, grad):
        self.gradients = grad.detach().cpu()

    def rnn_activations_hook(self, grad):
        self.rnn_gradients = grad.detach().cpu()
    
    def set_dropout_prob(self, p):
        for mm in self.modules():
            if type(mm) == nn.Dropout:
                mm.p = p
                print(mm.p)

    def forward(self, x, aux_model, aux_calib):
        # flatten batch and seq for CNNs
        bs, bptt, c, h, w = x.shape
        x = x.reshape(bs*bptt,c,h,w).contiguous() # contig required otherwise get error when using non-swarm dataloaders. "can;t something on something set w detach or .data"
        
        aux_model = aux_model.reshape(bs*bptt,N_AUX_MODEL_IN)

        if self.is_for_viz:
            bb = self.backbone
            x = bb[:VIZ_IX](x)
            x = bb[VIZ_IX](x)

            activations = x
            activations.register_hook(self.activations_hook)
            self.activations = activations.detach().cpu()
            x = activations

            x = bb[VIZ_IX+1:](x)
        else:
            x = self.backbone.conv_stem(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            x = checkpoint_sequential(self.backbone.blocks, 7, x)
            x = self.backbone.conv_head(x)
            x = self.backbone.bn2(x)
            x = self.backbone.act2(x)
            x = self.backbone.global_pool(x)
            x = self.backbone.classifier(x)

        x = torch.cat([x, aux_model], dim=-1)
        x_fcs1_out = self.fcs_1(x)

        # unpack seq and batch for rnn
        x_fcs1_out = x_fcs1_out.reshape(bs, bptt, self.inner_dim)

        if self.use_trainable_hidden:
            hidden = self.hidden_init.expand(1, bs, self.inner_dim)
            cell = self.cell_init.expand(1, bs, self.inner_dim)
            self.use_trainable_hidden = False
        else:
            hidden, cell = self.h, self.c

        if self.training and not self.is_for_viz:
           x_fcs1_out = dropout_no_rescale(x_fcs1_out, p=.2)

        x, h_c = self._rnn(x_fcs1_out, (hidden, cell))
        self.h, self.c = h_c[0].detach(), h_c[1].detach()

        if self.is_for_viz:
            # Viz
            rnn_activations = x
            rnn_activations.register_hook(self.rnn_activations_hook)
            self.rnn_activations = rnn_activations.detach().cpu()
            x = rnn_activations

        obsnet_out = self.obsnet(x if self.is_for_viz else x.detach())
        #obsnet_out = self.obsnet(x)

        x = torch.cat([x, aux_calib], dim=-1) # cat in calib params

        x = self.fcs_2(x)

        wps_preds = self.wps_head(x)
        aux_preds = self.aux_targets_head(x)

        return wps_preds, aux_preds, obsnet_out
