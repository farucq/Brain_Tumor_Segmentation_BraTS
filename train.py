#Training loop + Dice metric
import os
import torch
import torch.optim as optim
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from dataset import load_sample
from model import get_model

BASE_DIR = "/content/drive/MyDrive/BraTS"

device = "cuda" if torch.cuda.is_available() else "cpu"

cases = sorted(os.listdir(BASE_DIR))
train_cases = cases[:4]

model = get_model().to(device)

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_dice = 0

for epoch in range(10):
    model.train()

    img, mask = load_sample(os.path.join(BASE_DIR, train_cases[0]))
    img, mask = img.to(device), mask.to(device)

    optimizer.zero_grad()

    out = model(img.unsqueeze(0))
    loss = loss_fn(out, mask.unsqueeze(0).unsqueeze(1))

    loss.backward()
    optimizer.step()

    model.eval()
    dice_metric(out, mask.unsqueeze(0).unsqueeze(1))
    dice = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Epoch {epoch} | Loss {loss.item():.4f} | Dice {dice:.4f}")

    if dice > best_dice:
        best_dice = dice
        torch.save(model.state_dict(), "best_model.pth")

print("Training completed")
