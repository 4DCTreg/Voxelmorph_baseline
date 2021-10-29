#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu.
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math

# import voxelmorph with pytorch backenda
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8

# device handling
gpu = '-1'

if gpu and (gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images

multichannel = False
add_feat_axis = not multichannel

save_path = 'D:/DeepRegProject/Voxelmorph_baseline/Results/'
n_index = 0
model_name = 'D:/DeepRegProject/Voxelmorph_baseline/models/MI_add_all_mask/0100.pt'
# load and set up model

model = vxm.networks_xp.VxmDense.load(model_name, device)
model.to(device)
model.eval()

Dice_func = vxm.losses.Dice()
moving_all_name = 'D:/LiverDataset/UII_liver/train_UII_preprocess/image_Ap01.nii'
fixed_all_name = 'D:/LiverDataset/UII_liver/train_UII_preprocess/image_Pvp01.nii'

def get_patches_all(patch_size, sizeI, overlap):
    n_r = math.floor((sizeI[0] - patch_size[0]) / overlap)
    n_c = math.floor((sizeI[1] - patch_size[1]) / overlap)
    n_s = math.floor((sizeI[2] - patch_size[2]) / overlap)
    x_1 = range(0, n_r * overlap+1, overlap)
    y_1 = range(0, n_c * overlap+1, overlap)
    z_1 = range(0, n_s * overlap+1, overlap)
    x_2 = range(patch_size[0], (x_1[-1] + patch_size[0]+1), overlap)
    y_2 = range(patch_size[1], (y_1[-1] + patch_size[1]+1), overlap)
    z_2 = range(patch_size[2], (z_1[-1] + patch_size[2]+1), overlap)
    s_x = math.floor((sizeI[0] - x_2[-1]) / 2)
    s_y = math.floor((sizeI[1] - y_2[-1]) / 2)
    s_z = math.floor((sizeI[2] - z_2[-1]) / 2)
    x_1 = np.array(x_1) + s_x
    x_2 = np.array(x_2) + s_x
    y_1 = np.array(y_1) + s_y
    y_2 = np.array(y_2) + s_y
    z_1 = np.array(z_1) + s_z
    z_2 = np.array(z_2) + s_z
    return x_1, y_1, z_1, x_2, y_2, z_2


class VecInt_all(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer_new(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class SpatialTransformer_new(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        #self.grid=self.grid.cuda()
        #flow=flow.cuda()
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

trshape = [80, 80, 80]
fullshape = [320, 256, 256]
warp_mask_function = SpatialTransformer_new(trshape)
warp_full_function = SpatialTransformer_new(fullshape)


Dice_or_all = 0
Dice_after_all = 0

data_path = 'D:/LiverDataset/UII_liver/train_UII_preprocess3/'
for n_patients in range(6):
    image_Ap_name = data_path + 'image_Ap0' + str(n_patients + 1) + '.nii'
    # image_Pre_name = data_path + 'image_Pre0' + str(n_patients + 1) + '.nii'
    image_Pvp_name = data_path + 'image_Pvp0' + str(n_patients + 1) + '.nii'
    mask_Ap_name = data_path + 'liver_Ap0' + str(n_patients + 1) + '.nii'
    # mask_Pre_name = data_path + 'liver_Pre0' + str(n_patients + 1) + '.nii'
    mask_Pvp_name = data_path + 'liver_Pvp0' + str(n_patients + 1) + '.nii'
    mask_artery_name = data_path + 'hepatic_artery0' + str(n_patients + 1) + '.nii'
    mask_vein_name = data_path + 'hepatic_vein0' + str(n_patients + 1) + '.nii'
    mask_portal_vein_name = data_path + 'portal_vein0' + str(n_patients + 1) + '.nii'
    mask_all_vessel_moving = data_path + 'mask_movingall_0'+str(n_patients + 1) + '.nii'
    mask_all_vessel_fixed = data_path + 'mask_fixedall_0'+str(n_index + 1) + '.nii'

    moving_img = vxm.py.utils.load_volfile(image_Ap_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed_img, fixed_affine_all = vxm.py.utils.load_volfile(
        image_Pvp_name, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    mask_moving = vxm.py.utils.load_volfile(mask_Ap_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    mask_fixed, mask_fixed_affine = vxm.py.utils.load_volfile(
        mask_Pvp_name, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    mask_artery = vxm.py.utils.load_volfile(mask_artery_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_all_vessel = vxm.py.utils.load_volfile(mask_all_vessel_moving, add_batch_axis=True,
                                                  add_feat_axis=add_feat_axis)
    fixed_all_vessel = vxm.py.utils.load_volfile(mask_all_vessel_fixed, add_batch_axis=True,
                                                 add_feat_axis=add_feat_axis)
    mask_vein = vxm.py.utils.load_volfile(mask_vein_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    mask_portal_vein = vxm.py.utils.load_volfile(mask_portal_vein_name, add_batch_axis=True,
                                                 add_feat_axis=add_feat_axis)

    input_moving_full = torch.from_numpy(moving_img).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed_full = torch.from_numpy(fixed_img).to(device).float().permute(0, 4, 1, 2, 3)
    input_mask_moving_full = torch.from_numpy(mask_moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_mask_fixed_full = torch.from_numpy(mask_fixed).to(device).float().permute(0, 4, 1, 2, 3)
    input_mask_artery = torch.from_numpy(mask_artery).to(device).float().permute(0, 4, 1, 2, 3)
    input_moving_vessel = torch.from_numpy(moving_all_vessel).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed_vessel = torch.from_numpy(fixed_all_vessel).to(device).float().permute(0, 4, 1, 2, 3)
    mask_vein = torch.from_numpy(mask_vein).to(device).float().permute(0, 4, 1, 2, 3)
    mask_portal_vein = torch.from_numpy(mask_portal_vein).to(device).float().permute(0, 4, 1, 2, 3)

    Dice_or_full = Dice_func.loss(input_mask_moving_full, input_mask_fixed_full)
    with torch.no_grad():
        warp_full = model(input_moving_full, input_fixed_full, input_moving_vessel, input_fixed_vessel,
                          input_mask_moving_full, regist_full=True)

    mask_moved_all_after = warp_full_function(input_mask_moving_full, warp_full)
    mask_artery_moved = warp_full_function(input_moving_vessel,warp_full)
    moved_img = warp_full_function(input_moving_full, warp_full)
    Dice_after_full = Dice_func.loss(mask_moved_all_after, input_mask_fixed_full)
    image_Ap_after = moved_img.detach().cpu().numpy().squeeze()
    mask_moved_img = mask_moved_all_after.detach().cpu().numpy().squeeze()
    mask_artery_moved = mask_artery_moved.detach().cpu().numpy().squeeze()
    mask_vein = mask_vein.detach().cpu().numpy().squeeze()
    mask_portal_vein = mask_portal_vein.detach().cpu().numpy().squeeze()
    mask_artery_align = input_mask_artery.detach().cpu().numpy().squeeze()
    mask_fixed = input_mask_fixed_full.detach().cpu().numpy().squeeze()
    warp_full_save = warp_full.detach().cpu().numpy().squeeze()

    mask_artery_moved_name = save_path + 'mask_artery0'+str(n_patients+1) + 'MI_v1.nii'
    mask_moved_name = save_path + 'mask_Ap0'+str(n_patients+1)+'MI_v1.nii'
    vxm.py.utils.save_volfile(mask_artery_moved.round(), mask_artery_moved_name)
    vxm.py.utils.save_volfile(mask_moved_img.round(), mask_moved_name)

    deformation_field_name = save_path + 'deformation_field0' + str(n_patients+1) + 'MI_v1.nii'
    vxm.py.utils.save_volfile(warp_full_save[0, :, :, :], deformation_field_name)

    # visual compare
    mask_vein_align = mask_vein
    mask_vein_align[mask_vein_align == 1] = 2
    mask_portal_vein_align = mask_portal_vein
    mask_portal_vein_align[mask_portal_vein_align == 1] = 3
    mask_all_vessel_after = mask_artery_align.round() + mask_portal_vein_align+mask_vein_align
    mask_all_vessel_name = save_path + 'mask_all_vessel0' + str(n_patients+1) + 'MI_v1.nii'
    vxm.py.utils.save_volfile(mask_all_vessel_after, mask_all_vessel_name)

    mask_moved_img_align = mask_moved_img.round()
    mask_moved_img_align[mask_moved_img_align == 1] = 2
    mask_liver_all = mask_moved_img_align + mask_fixed
    mask_liver_name = save_path + 'mask_all_liver0' + str(n_patients+1) + 'MI_v1.nii'
    vxm.py.utils.save_volfile(mask_liver_all, mask_liver_name)

    print(Dice_after_full)

