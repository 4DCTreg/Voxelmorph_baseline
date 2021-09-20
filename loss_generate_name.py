path_patches = 'D:/LiverDataset/UII_liver/train_UII_preprocess3/'

with open('list_image_add_all.txt', 'w') as f:
    for n_index in range(6):
        Ap_name = 'image_Ap0'+str(n_index+1)+'.nii'
        Pvp_name = 'image_Pvp0' + str(n_index + 1) + '.nii'
        mask_Ap = 'liver_Ap0' + str(n_index + 1) + '.nii'
        mask_Pvp = 'liver_Pvp0' + str(n_index + 1) + '.nii'
        mask_artery = 'hepatic_artery0' + str(n_index + 1) + '.nii'
        mask_vein = 'hepatic_vein0' + str(n_index + 1) + '.nii'
        mask_portal_vein = 'portal_vein0' + str(n_index + 1) + '.nii'
        mask_moving_all = 'mask_movingall_0'+str(n_index + 1) + '.nii'
        mask_fixed_all = 'mask_fixedall_0'+str(n_index + 1) + '.nii'

        f.write(path_patches + Ap_name)
        f.write('\n')
        f.write(path_patches + mask_moving_all)
        f.write('\n')
        f.write(path_patches + Pvp_name)
        f.write('\n')
        f.write(path_patches + mask_fixed_all)
        f.write('\n')
        f.write(path_patches + mask_Ap)
        f.write('\n')
        f.write(path_patches + mask_Pvp)
        f.write('\n')
        f.write(path_patches + mask_artery)
        f.write('\n')
        f.write(path_patches + mask_vein)
        f.write('\n')
        f.write(path_patches + mask_portal_vein)
        f.write('\n')

f.close()