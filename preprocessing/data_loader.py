#data_loader.py
import os
import re
import numpy as np
import pydicom
import nibabel as nib
import cv2
import torch
from torch.utils.data import Dataset


class PerfusionDataset(Dataset):
    def __init__(self, dicom_dir, mask_dir, img_size=128, negative_keep_prob=0.30, max_samples=None):
        self.images = []
        self.masks = []
        self.img_size = img_size
        self.negative_keep_prob = negative_keep_prob
        self.max_samples = max_samples

        dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]

        print("Found DICOM files:", len(dicom_files))
        print("Found mask files:", len(mask_files))

        dicom_dict = {}
        for f in dicom_files:
            uid = f.replace(".dcm", "")
            dicom_dict[uid] = os.path.join(dicom_dir, f)

        mask_dict = {}
        for f in mask_files:
            uid = re.sub(r"_mask.*", "", f)
            mask_dict[uid] = os.path.join(mask_dir, f)

        common_uids = sorted(list(set(dicom_dict.keys()) & set(mask_dict.keys())))
        print("Matched UID pairs:", len(common_uids))

        rng = np.random.default_rng(42)

        stop = False
        for uid in common_uids:
            if stop:
                break

            dicom_path = dicom_dict[uid]
            mask_path = mask_dict[uid]

            try:
                dicom = pydicom.dcmread(dicom_path)
                volume = dicom.pixel_array.astype(np.float32)

                if volume.ndim == 2:
                    volume = np.expand_dims(volume, axis=0)
                elif volume.ndim != 3:
                    print(f"Skipping unsupported DICOM shape for UID {uid}: {volume.shape}")
                    continue

                mask = nib.load(mask_path).get_fdata().astype(np.float32)

                if mask.ndim == 2:
                    mask = np.expand_dims(mask, axis=0)
                elif mask.ndim == 3:
                    if mask.shape[-1] == volume.shape[0]:
                        mask = mask.transpose(2, 0, 1)
                    elif mask.shape[0] == volume.shape[0]:
                        pass
                    else:
                        print(f"Skipping UID {uid}: mask shape mismatch {mask.shape} vs volume {volume.shape}")
                        continue
                else:
                    print(f"Skipping unsupported mask shape for UID {uid}: {mask.shape}")
                    continue

                mask = (mask > 0).astype(np.float32)
                num_slices = min(volume.shape[0], mask.shape[0])

                for s in range(num_slices):
                    img = volume[s]
                    m = mask[s]

                    if np.sum(m) == 0 and rng.random() > self.negative_keep_prob:
                        continue

                    mean = img.mean()
                    std = img.std()
                    if std > 1e-8:
                        img = (img - mean) / std
                    else:
                        img = np.zeros_like(img, dtype=np.float32)

                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                    m = (m > 0.5).astype(np.float32)

                    self.images.append(img.astype(np.float32))
                    self.masks.append(m.astype(np.float32))

                    if self.max_samples is not None and len(self.images) >= self.max_samples:
                        stop = True
                        break

            except Exception as e:
                print(f"Error loading UID {uid}: {e}")

        print("Total slices:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float()
        return image, mask