from constants import *
from imports import *

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
# b5 is 2048 features, 32M params, don't seem to be any weights for models this size and above, actually there are, under the 'tf' prefix

class EffNet(nn.Module):
    def __init__(self, is_for_viz=False):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, in_chans=N_CHANNELS).to(device)
        # self.backbone = timm.create_model("tf_efficientnet_b6", pretrained=True, in_chans=N_CHANNELS).to(device)
        self.backbone.classifier = nn.Identity()
        self.backbone_out = self.backbone.num_features #b3 is 1536

        n = self.backbone_out + N_AUX_CALIB_IN + N_AUX_MODEL_IN
        self.wps_head = nn.Linear(n, N_TARGETS)
        self.aux_targets_head = nn.Linear(n, N_AUX_TARGETS)
        self.obsnet = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 3))

        self.activations = None # just storing them here so don't have to return them
        self.gradients = None
        self.rnn_activations = None
        self.rnn_gradients = None

        self.is_for_viz = is_for_viz

    def reset_hidden(self, bs):
        pass

    def convert_backbone_to_sequential(self):
        self.backbone = self.backbone.as_sequential()

    def activations_hook(self, grad):
        self.gradients = grad.detach().cpu()

    def forward(self, x, aux_model, aux_calib):
        # flatten batch and seq for CNNs
        bs, bptt, c, h, w = x.shape
        x = x.reshape(bs*bptt,c,h,w).contiguous() 
        
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

        # unpack seq and batch for rnn
        x = x.reshape(bs, bptt, self.backbone_out)
        x = torch.cat([x, aux_calib, aux_model], dim=-1) # cat in calib params
        
        wps_preds = self.wps_head(x)
        aux_preds = self.aux_targets_head(x)
        obsnet_out = self.obsnet(x if self.is_for_viz else x.detach())

        if self.is_for_viz:
            # Quick hack so can still use viz apparatus as usual
            self.rnn_activations = torch.zeros((bs, bptt, 512))
            self.rnn_gradients = torch.zeros((bs, bptt, 512))

        return wps_preds, aux_preds, obsnet_out
