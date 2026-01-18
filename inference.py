#Sliding window inference + visualization
import os
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

from dataset import load_sample
from model import get_model

BASE_DIR = "/content/drive/MyDrive/BraTS"
device = "cuda" if torch.cuda.is_available() else "cpu"

cases = sorted(os.listdir(BASE_DIR))
val_case = cases[-1]

model = get_model().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

img, mask = load_sample(os.path.join(BASE_DIR, val_case))
img = img.to(device)

with torch.no_grad():
    pred = sliding_window_inference(
        img.unsqueeze(0),
        roi_size=(128,128,64),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25
    )

pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()

flair = nib.load(os.path.join(BASE_DIR, val_case, "FLAIR.nii.gz")).get_fdata()
slice_id = flair.shape[2] // 2

plt.imshow(flair[:,:,slice_id], cmap="gray")
plt.imshow(pred_mask[:,:,slice_id], alpha=0.4)
plt.title("Predicted Tumor Overlay")
plt.axis("off")
plt.show()
