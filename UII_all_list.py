import numpy as np
path_patches = 'D:/LiverDataset/UII_train_all/Step3_crop_UII_all/'
train_index = np.random.choice(np.arange(84), size=67, replace=False)
test_index = np.delete(np.arange(84), train_index)

with open('UII_all_train.txt', 'w') as f:
    for n_index in train_index:
        fix_img_name = 'volume-'+str(n_index)+'.nii'
        fix_img_seg_name = 'segmentation-'+str(n_index)+'.nii'
        mov_img_name = 'volume-'+str(n_index)+'.nii'
        mov_img_seg_name = 'segmentation-'+str(n_index)+'.nii'
        f.write(path_patches+mov_img_name)
        f.write('\n')
        f.write(path_patches+mov_img_seg_name)
        f.write('\n')
        f.write(path_patches+fix_img_name)
        f.write('\n')
        f.write(path_patches+fix_img_seg_name)
        f.write('\n')
f.close()

with open('UII_all_test.txt', 'w') as h:
    for n_index in test_index:
        fix_img_name = 'volume-'+str(n_index)+'.nii'
        fix_img_seg_name = 'segmentation-'+str(n_index)+'.nii'
        mov_img_name = 'volume-'+str(n_index)+'.nii'
        mov_img_seg_name = 'segmentation-'+str(n_index)+'.nii'
        h.write(path_patches+mov_img_name)
        h.write('\n')
        h.write(path_patches+mov_img_seg_name)
        h.write('\n')
        h.write(path_patches+fix_img_name)
        h.write('\n')
        h.write(path_patches+fix_img_seg_name)
        h.write('\n')
h.close()