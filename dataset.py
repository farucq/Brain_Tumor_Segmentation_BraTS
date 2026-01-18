#Handles data loading, normalization, cropping
import os
import numpy as np
import nibabel as nib
import torch

def load_nifti(path):
    return nib.load(path).get_fdata()

def z_norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def random_crop(img, mask, size=(128,128,64)):
    x = np.random.randint(0, img.shape[1] - size[0])
    y = np.random.randint(0, img.shape[2] - size[1])
    z = np.random.randint(0, img.shape[3] - size[2])

    return (
        img[:, x:x+size[0], y:y+size[1], z:z+size[2]],
        mask[x:x+size[0], y:y+size[1], z:z+size[2]]
    )

def load_sample(folder):
    t1 = z_norm(load_nifti(folder + "/T1.nii.gz"))
    t1ce = z_norm(load_nifti(folder + "/T1ce.nii.gz"))
    t2 = z_norm(load_nifti(folder + "/T2.nii.gz"))
    flair = z_norm(load_nifti(folder + "/FLAIR.nii.gz"))
    seg = load_nifti(folder + "/seg.nii.gz")

    image = np.stack([t1, t1ce, t2, flair])
    image, seg = random_crop(image, seg)

    return torch.tensor(image).float(), torch.tensor(seg).long()
