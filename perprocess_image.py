'''
This per-porcess code is to align the ground truth e,g.liver_mask and mask_artery
'''
import os
import voxelmorph as vxm
import torch
import numpy
multichannel = False
add_feat_axis = not multichannel

# device handling
gpu = '-1'

if gpu and (gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['VXM_BACKEND'] = 'pytorch'
data_path = 'D:/LiverDataset/UII_liver/train_UII_preprocess3/'
for n_paired in range(6):

    image_Ap_name = data_path + 'image_Ap0' + str(n_paired + 1) + '.nii'
    # image_Pre_name = data_path + 'image_Pre0' + str(n_patients + 1) + '.nii'
    image_Pvp_name = data_path + 'image_Pvp0' + str(n_paired + 1) + '.nii'
    mask_Ap_name = data_path + 'liver_Ap0' + str(n_paired + 1) + '.nii'
    # mask_Pre_name = data_path + 'liver_Pre0' + str(n_patients + 1) + '.nii'
    mask_Pvp_name = data_path + 'liver_Pvp0' + str(n_paired + 1) + '.nii'
    mask_artery_name = data_path + 'hepatic_artery0' + str(n_paired + 1) + '.nii'
    mask_vein_name = data_path + 'hepatic_vein0' + str(n_paired + 1) + '.nii'
    mask_portal_vein_name = data_path + 'portal_vein0' + str(n_paired + 1) + '.nii'

    mask_moving = vxm.py.utils.load_volfile(mask_Ap_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    mask_artery = vxm.py.utils.load_volfile(mask_artery_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    mask_fixed = vxm.py.utils.load_volfile(mask_Pvp_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    mask_vein = vxm.py.utils.load_volfile(mask_vein_name, add_batch_axis=True, add_feat_axis=add_feat_axis)
    mask_portal_vein = vxm.py.utils.load_volfile(mask_portal_vein_name, add_batch_axis=True, add_feat_axis=add_feat_axis)

    mask_moving_full = torch.from_numpy(mask_moving).to(device).float().permute(0, 4, 1, 2, 3)
    mask_fixed_full = torch.from_numpy(mask_fixed).to(device).float().permute(0, 4, 1, 2, 3)
    mask_artery_full = torch.from_numpy(mask_artery).to(device).float().permute(0, 4, 1, 2, 3)
    mask_vein_full = torch.from_numpy(mask_vein).to(device).float().permute(0, 4, 1, 2, 3)
    mask_portal_vein_full = torch.from_numpy(mask_portal_vein).to(device).float().permute(0, 4, 1, 2, 3)
    mask_moving_np = mask_moving_full.detach().cpu().numpy().squeeze()
    mask_fixed_np = mask_fixed_full.detach().cpu().numpy().squeeze()
    mask_artery_np = mask_artery_full.detach().cpu().numpy().squeeze()
    mask_vein_np = mask_vein_full.detach().cpu().numpy().squeeze()
    mask_portal_vein_np = mask_portal_vein_full.detach().cpu().numpy().squeeze()

    mask_artery_np[mask_artery_np == 1] = 2
    mask_vein_np[mask_vein_np == 1] = 3
    mask_portal_vein_np[mask_portal_vein_np == 1] = 4

    mask_moving_all = mask_moving_np+mask_artery_np
    mask_moving_all[mask_moving_all == 1] = 0.2
    mask_moving_all[mask_moving_all == 2] = 0.4
    mask_moving_all[mask_moving_all == 3] = 0.4

    mask_fixed_all = mask_vein_np+mask_fixed_np
    mask_fixed_all[mask_fixed_all == 1] = 0.2
    mask_fixed_all[mask_fixed_all == 3] = 0.6
    mask_fixed_all[mask_fixed_all == 4] = 0.6
    mask_fixed_all = mask_fixed_all + mask_portal_vein_np
    mask_fixed_all[mask_fixed_all == 4] = 0.8
    mask_fixed_all[mask_fixed_all == 4.2] = 0.8
    mask_fixed_all[mask_fixed_all == 4.6] = 0.6
    mask_moving_all_name = data_path+'mask_movingall_0'+str(n_paired + 1) + '.nii'
    mask_fixed_all_name = data_path+'mask_fixedall_0'+str(n_paired + 1) + '.nii'

    vxm.py.utils.save_volfile(mask_fixed_all, mask_fixed_all_name)
    vxm.py.utils.save_volfile(mask_moving_all, mask_moving_all_name)

