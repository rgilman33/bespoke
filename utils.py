from constants import *
from imports import *

################
# BEV
################

def _edge_mask(bev):
    mask = torch.zeros_like(bev)
    _bev = bev.clone()
    _bev[_bev>.5] = 1
    _bev[_bev<=.5] = 0
    h = abs(_bev[:,:, :,:,1:] - _bev[:,:, :,:,:-1]) > 0
    v = abs(_bev[:,:, :,1:,:] - _bev[:,:, :,:-1,:]) > 0
    mask[:,:, :,:,:-1] += h
    mask[:,:, :,:,1:] += h
    mask[:,:, :,1:,:] += v
    mask[:,:, :,:-1,:] += v
    return mask

def is_edge(bev):
    mask_1 = _edge_mask(bev)
    mask_2 = _edge_mask(mask_1)
    #mask_3 = _edge_mask(mask_2)
    mask = (mask_1 + mask_2)
    mask = mask.sum(2, keepdim=True).clamp(0,1).bool()
    #mask = mask.expand(-1,-1, 3,-1,-1) # channels dim
    return mask

# stopsign 1, 1, 0
# rd surface 0, 0, 1
# rd lines 1, 0, 1
# npc 0, 1, 0

def is_npc(bev):
    is_npc = ((bev[:,:, 0,:,:]<.5) & (bev[:,:, 1,:,:]>.2) & (bev[:,:, 2,:,:]<.5)).unsqueeze(2)
    is_npc = (is_npc | is_edge(is_npc.int()))
    return is_npc

def is_stopsign(bev):
    _is_stopsign = ((bev[:,:, 0,:,:]>.2) & (bev[:,:, 1,:,:]>.2) & (bev[:,:, 2,:,:]<.5)).unsqueeze(2)
    _is_stopsign = (_is_stopsign | is_edge(_is_stopsign.int()))
    _is_stopsign = (_is_stopsign | is_edge(_is_stopsign.int()))
    return _is_stopsign

def is_rd_markings(bev):
    _is_rd_markings = ((bev[:,:, 0,:,:]>.2) & (bev[:,:, 1,:,:]<.5) & (bev[:,:, 2,:,:]>.2)).unsqueeze(2) # why can't have blue exclusion?
    _is_rd_markings = (_is_rd_markings | is_edge(_is_rd_markings.int()))
    return _is_rd_markings

def is_rd_surface(bev):
    _is_rd = ((bev[:,:, 0,:,:]<.5) & (bev[:,:, 1,:,:]<.5) & (bev[:,:, 2,:,:]>.2)).unsqueeze(2) # rd or rd marking counts as rd
    return _is_rd

def is_rd(bev):
    _is_rd = is_rd_surface(bev) | is_rd_markings(bev)
    return _is_rd

def is_background(bev):
    return ((bev[:,:, 0,:,:]<.1) & (bev[:,:, 1,:,:]<.1) & (bev[:,:, 2,:,:]<.1)).unsqueeze(2)

def depth_loss_mask(bev):
    mask = ~is_background(bev)
    b = 8
    for _ in range(b):
        mask = (mask | _edge_mask(mask.int()).bool())
    return mask


def clean_up_bev(bev):

    # # Determine what is what
    # _is_background = is_background(bev)
    # _is_rd = is_rd(bev)
    # _is_rd_markings = is_rd_markings(bev)
    # _is_npc = is_npc(bev)
    # _is_stopsign = is_stopsign(bev)

    # # Clean up semseg. These layer on, latter ones may override former

    # # Background
    # bev[:,:,0:1,:,:][_is_background] = 0
    # bev[:,:,1:2,:,:][_is_background] = 0
    # bev[:,:,2:3,:,:][_is_background] = 0

    # # # rd
    # bev[:,:,2:3,:,:][_is_rd] = 1.

    # # rd markings
    # bev[:,:,0:1,:,:][_is_rd_markings] = 1.
    # bev[:,:,0:1,:,:][_is_stopsign] = 1. # overloading this for now

    # # NPCs
    # bev[:,:,1:2,:,:][_is_npc] = 1.

    bev[:,:, :3,:,:] = bev[:,:, :3,:,:].round() # round the semseg in place, keep continuous for depth

    return bev

def get_m_sz(m):
    return sum([torch.numel(p) for p in m.parameters()]) // 1e6