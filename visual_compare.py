# import voxelmorph with pytorch backenda
import os
import torch
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
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

save_path = 'D:/DeepRegProject/Voxelmorph_baseline/Results/'
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
    mask_all_vessel_fixed = data_path + 'mask_fixedall_0'+str(n_patients + 1) + '.nii'

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

    input_moving_full = torch.from_numpy(moving_img).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed_full = torch.from_numpy(fixed_img).to(device).float().permute(0, 4, 1, 2, 3)
    input_mask_moving_full = torch.from_numpy(mask_moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_mask_fixed_full = torch.from_numpy(mask_fixed).to(device).float().permute(0, 4, 1, 2, 3)
    input_mask_artery = torch.from_numpy(mask_artery).to(device).float().permute(0, 4, 1, 2, 3)
    input_moving_vessel = torch.from_numpy(moving_all_vessel).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed_vessel = torch.from_numpy(fixed_all_vessel).to(device).float().permute(0, 4, 1, 2, 3)





    image_Ap_after = moved_img.detach().cpu().numpy().squeeze()
    mask_moved_img = mask_moved_all_after.detach().cpu().numpy().squeeze()
    mask_artery_moved = mask_artery_moved.detach().cpu().numpy().squeeze()

    mask_artery_moved_name = save_path + 'mask_artery'+str(n_patients+1) + 'MI_v1.nii'
    mask_moved_name = save_path + 'mask_Ap0'+str(n_patients+1)+'MI_v1.nii'

    vxm.py.utils.save_volfile(mask_artery_moved.round(), mask_artery_moved_name)
    vxm.py.utils.save_volfile(mask_moved_img, mask_moved_name)

    print(Dice_or_full)
    print(Dice_after_full)
