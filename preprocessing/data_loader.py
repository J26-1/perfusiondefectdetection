import os
import re
import numpy as np
import pydicom
import nibabel as nib
import cv2

from torch.utils.data import Dataset


class PerfusionDataset(Dataset):

    def __init__(self, dicom_dir, mask_dir, img_size=128):

        self.images = []
        self.masks = []
        self.img_size = img_size

        # ----------------------------
        # Collect files
        # ----------------------------
        dicom_files = [
            f for f in os.listdir(dicom_dir)
            if f.endswith(".dcm")
        ]

        mask_files = [
            f for f in os.listdir(mask_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]

        print("Found DICOM files:", len(dicom_files))
        print("Found mask files:", len(mask_files))

        # ----------------------------
        # Build UID lookup tables
        # ----------------------------
        dicom_dict = {}

        for f in dicom_files:
            uid = f.replace(".dcm", "")
            dicom_dict[uid] = os.path.join(dicom_dir, f)

        mask_dict = {}

        for f in mask_files:
            uid = re.sub(r"_mask.*", "", f)
            mask_dict[uid] = os.path.join(mask_dir, f)

        # ----------------------------
        # Match UID pairs
        # ----------------------------
        common_uids = list(set(dicom_dict.keys()) & set(mask_dict.keys()))

        print("Matched UID pairs:", len(common_uids))

        # ----------------------------
        # Load data
        # ----------------------------
        for uid in common_uids:

            dicom_path = dicom_dict[uid]
            mask_path = mask_dict[uid]

            try:

                # ----------------------------
                # Load DICOM image
                # ----------------------------
                dicom = pydicom.dcmread(dicom_path)
                volume = dicom.pixel_array.astype(np.float32)

                # Normalize image
                vmin = volume.min()
                vmax = volume.max()

                if vmax - vmin > 0:
                    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                else:
                    volume = np.zeros_like(volume)

                # Ensure shape (Z,H,W)
                if volume.ndim == 2:
                    volume = np.expand_dims(volume, axis=0)

                elif volume.ndim == 3:
                    pass

                else:
                    continue

                # ----------------------------
                # Load mask
                # ----------------------------
                mask = nib.load(mask_path).get_fdata().astype(np.float32)

                # Ensure mask is (Z, H, W)
                if mask.shape[-1] == volume.shape[0]:
                    mask = mask.transpose(2, 0, 1)

                # Convert to binary
                mask = (mask > 0).astype(np.float32)

                # ----------------------------
                # Slice matching
                # ----------------------------
                slices = min(volume.shape[0], mask.shape[0])

                for s in range(slices):

                    img = volume[s]
                    m = mask[s]

                    # Skip slices with no myocardium
                    if np.sum(m) == 0 and np.random.rand() > 0.3:
                        continue

                    # Resize image
                    img = cv2.resize(
                        img,
                        (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LINEAR
                    )

                    # Resize mask
                    m = cv2.resize(
                        m,
                        (self.img_size, self.img_size),
                        interpolation=cv2.INTER_NEAREST
                    )

                    self.images.append(img.astype(np.float32))
                    self.masks.append(m.astype(np.float32))

            except Exception as e:

                print("Error loading UID:", uid)
                print(e)

        print("Total slices:", len(self.images))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        return image, mask