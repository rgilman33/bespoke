from constants import *
from imports import *

class ObserverNet(nn.Module):
    def __init__(self):
        super(ObserverNet, self).__init__()
        act = nn.ReLU()
        drop = nn.Dropout(0.2)
        self.fcs = nn.Sequential(nn.Linear(512, 256), act, drop, nn.Linear(256, 3))

    def forward(self, x):
        return self.fcs(x)

VIZ_IX = 5 # viz_ix from 3 (granular) to at least 5, maybe more?

class EffNet(nn.Module):
    def __init__(self, model_arch='efficientnet_b3', is_for_viz=False):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model(model_arch, pretrained=True, in_chans=N_CHANNELS).to(device)
        self.backbone.classifier = nn.Identity()

        backbone_out = self.backbone.num_features #b3 is 1536
        act = nn.ReLU()
        drop = nn.Dropout(0.2)
        self.inner_dim = 512
        self.fcs_1 = nn.Sequential( nn.Linear(backbone_out + len(aux_properties), self.inner_dim), act, drop)
        self.use_rnn = True
        
        self.rnn = nn.GRU(self.inner_dim, self.inner_dim, 1, batch_first=True)
        self.fcs_2 = nn.Sequential(nn.Linear(self.inner_dim, 256), act, drop, nn.Linear(256, N_PRED))

        self.obsnet = ObserverNet()

        self.activations = None # just storing them here so don't have to return them
        self.gradients = None
        self.rnn_activations = None
        self.rnn_gradients = None

        self.is_for_viz = is_for_viz

    def convert_backbone_to_sequential(self):
        self.backbone = self.backbone.as_sequential()
        #self.backbone = ModuleWrapperIgnores2ndArg(self.backbone)

    def reset_hidden(self, bs):
        self.h = torch.zeros((1,bs,self.inner_dim)).to(device)

    def activations_hook(self, grad):
        self.gradients = grad.detach().cpu()

    def rnn_activations_hook(self, grad):
        self.rnn_gradients = grad.detach().cpu()

    def forward(self, x, aux): #x is img
        # reshape for CNNs
        bs, bptt, c, h, w = x.shape
        x = x.reshape(bs*bptt,c,h,w).contiguous() # contig required otherwise get error when using non-swarm dataloaders. "can;t something on something set w detach or .data"
        aux = aux.reshape(bs*bptt,len(aux_properties))

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
            #x = self.backbone(img)

            # #modules = [module for k, module in self.backbone._modules.items()]
            # x = checkpoint_sequential(modules, 4, img)
            # x = checkpoint_sequential(self.backbone, 4, (img, self.dummy_tensor))

            x = self.backbone.conv_stem(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            #x = torch.utils.checkpoint.checkpoint(self.backbone.blocks, x)
            #x = self.backbone.blocks(x)
            x = checkpoint_sequential(self.backbone.blocks, 7, x)
            x = self.backbone.conv_head(x)
            x = self.backbone.bn2(x)
            x = self.backbone.act2(x)
            x = self.backbone.global_pool(x)
            x = self.backbone.classifier(x)
            
            #x = torch.utils.checkpoint.checkpoint(self.backbone, img, self.dummy_tensor)


        x = torch.cat([x, aux], dim=-1)
        x = self.fcs_1(x)

        # unpack seq and bs for rnn
        x = x.reshape(bs, bptt, self.inner_dim)
        x, self.h = self.rnn(x, self.h)
        self.h.detach_()

        if self.is_for_viz:
            # Viz
            rnn_activations = x
            rnn_activations.register_hook(self.rnn_activations_hook)
            self.rnn_activations = rnn_activations.detach().cpu()
            x = rnn_activations

        obsnet_out = self.obsnet(x if self.is_for_viz else x.detach())
        x = self.fcs_2(x)
        x = x[:,:,:N_WPS_TO_USE]
        return x, obsnet_out
