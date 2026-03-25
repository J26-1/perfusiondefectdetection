import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

mask_path = "data/raw/NIfTI/1.2.840.4267.32.591718155288413424886041785541718021_mask.nii.gz"

mask = nib.load(mask_path).get_fdata()

print("Mask shape:", mask.shape)
print("Max value:", np.max(mask))
print("Min value:", np.min(mask))
print("Unique values:", np.unique(mask))

slice_id = mask.shape[0] // 2

plt.imshow(mask[slice_id], cmap="gray")
plt.title("Mask slice")
plt.show()