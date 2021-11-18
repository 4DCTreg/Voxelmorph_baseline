import numpy as np
path_patches = 'D:/LiverDataset/UII_train_all/Step3_crop_UII_all/'
train_index = np.random.choice(np.arange(84), size=66, replace=False)
validation_test = np.delete(np.arange(84), train_index)
test_index = np.random.choice(validation_test, size=9, replace=False)
indices = np.argwhere(np.isin(validation_test, test_index))
validation_index = np.delete(validation_test, indices)

train_index = train_index+1
validation_index = validation_index+1
test_index = test_index+1
with open('UII_all_train.txt', 'w') as f:
    for n_index in train_index:
        mov_img_name = 'image_Ap'+str(n_index)+'.nii'
        mov_img_seg_name = 'liver_Ap'+str(n_index)+'.nii'
        artery_name = 'hepatic_artery'+str(n_index)+'.nii'
        fix_img_name = 'image_Pvp'+str(n_index)+'.nii'
        fix_img_seg_name = 'liver_Pvp'+str(n_index)+'.nii'
        vein_name = 'vein'+str(n_index)+'.nii'
        f.write(path_patches+mov_img_name)
        f.write('\n')
        f.write(path_patches+mov_img_seg_name)
        f.write('\n')
        f.write(path_patches+artery_name)
        f.write('\n')
        f.write(path_patches+fix_img_name)
        f.write('\n')
        f.write(path_patches+fix_img_seg_name)
        f.write('\n')
        f.write(path_patches+vein_name)
        f.write('\n')
f.close()

with open('UII_all_test.txt', 'w') as h:
    for n_index in test_index:
        mov_img_name = 'image_Ap' + str(n_index) + '.nii'
        mov_img_seg_name = 'liver_Ap' + str(n_index) + '.nii'
        artery_name = 'hepatic_artery' + str(n_index) + '.nii'
        fix_img_name = 'image_Pvp' + str(n_index) + '.nii'
        fix_img_seg_name = 'liver_Pvp' + str(n_index) + '.nii'
        vein_name = 'vein' + str(n_index) + '.nii'
        h.write(path_patches + mov_img_name)
        h.write('\n')
        h.write(path_patches + mov_img_seg_name)
        h.write('\n')
        h.write(path_patches + artery_name)
        h.write('\n')
        h.write(path_patches + fix_img_name)
        h.write('\n')
        h.write(path_patches + fix_img_seg_name)
        h.write('\n')
        h.write(path_patches + vein_name)
        h.write('\n')
h.close()

with open('UII_all_validation.txt', 'w') as g:
    for n_index in validation_index:
        mov_img_name = 'image_Ap' + str(n_index) + '.nii'
        mov_img_seg_name = 'liver_Ap' + str(n_index) + '.nii'
        artery_name = 'hepatic_artery' + str(n_index) + '.nii'
        fix_img_name = 'image_Pvp' + str(n_index) + '.nii'
        fix_img_seg_name = 'liver_Pvp' + str(n_index) + '.nii'
        vein_name = 'vein' + str(n_index) + '.nii'
        g.write(path_patches + mov_img_name)
        g.write('\n')
        g.write(path_patches + mov_img_seg_name)
        g.write('\n')
        g.write(path_patches + artery_name)
        g.write('\n')
        g.write(path_patches + fix_img_name)
        g.write('\n')
        g.write(path_patches + fix_img_seg_name)
        g.write('\n')
        g.write(path_patches + vein_name)
        g.write('\n')
g.close()