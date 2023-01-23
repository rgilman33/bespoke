from __future__ import absolute_import, division, print_function


import deprecated.optviz_utils as optviz_utils
from constants import *
from imports import *

from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch


def render_viz(model, img_sz=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS_MODEL), viz_ix=0):
    thresholds = (512,)
    param_f = lambda: optviz_utils.image(*img_sz)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    optimizer = torch.optim.Adam(params, lr=5e-2)

    # hook = hook_model(model, image_f)
    # objective_f = optviz_utils.as_objective(objective_f)

    images = []
    for i in tqdm(range(1, max(thresholds) + 1)):
        def closure():
            optimizer.zero_grad()
            backbone_out = model(image_f()) #TODO aug here
            # loss = objective_f(hook)
            loss = backbone_out[0][viz_ix]
            loss.backward()
            return loss
            
        optimizer.step(closure)
        if i in thresholds:
            image = tensor_to_img_array(image_f())
            images.append(image)

    return images


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


# class ModuleHook:
#     def __init__(self, module):
#         self.hook = module.register_forward_hook(self.hook_fn)
#         self.module = None
#         self.features = None

#     def hook_fn(self, module, input, output):
#         self.module = module
#         self.features = output

#     def close(self):
#         self.hook.remove()


# def hook_model(model, image_f):
#     features = OrderedDict()

#     # recursive hooking function
#     def hook_layers(net, prefix=[]):
#         if hasattr(net, "_modules"):
#             for name, layer in net._modules.items():
#                 if layer is None:
#                     # e.g. GoogLeNet's aux1 and aux2 layers
#                     continue
#                 features["_".join(prefix + [name])] = ModuleHook(layer)
#                 hook_layers(layer, prefix=prefix + [name])

#     hook_layers(model)

#     def hook(layer):
#         if layer == "input":
#             out = image_f()
#         elif layer == "labels":
#             out = list(features.values())[-1].features
#         else:
#             assert layer in features
#             out = features[layer].features
#         assert out is not None
#         return out

#     return hook