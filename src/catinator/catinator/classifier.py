import os, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ——————— 1 · Setup ———————
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ——————— 2 · 299×299 Transforms & DataLoader ———————
val_tfms = transforms.Compose([
    transforms.Resize(358),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

custom_root = Path("/home/ubuntu/robowalle/src/catinator/catinator/testData")
print(custom_root)
dataset     = datasets.ImageFolder(custom_root, transform=val_tfms)
loader      = DataLoader(dataset,
                         batch_size=32,
                         shuffle=False,)

# ——————— 3 · Discover original 21 classes ———————
train_root   = Path("/home/ubuntu/robowalle/src/catinator/catinator/data_split")/"train"
orig_ds      = datasets.ImageFolder(train_root, transform=val_tfms)
orig_classes = orig_ds.classes
# map your 3 test-folder names to their indices in the 21-way head
test_indices = [orig_classes.index(c) for c in dataset.classes]
print("Test folders:", dataset.classes)
print("Mapped to original indices:", test_indices)

# ——————— 4 · Build & load 21-way ResNet50 ———————
# simple factory that only supports resnet50 here
def make_resnet50(n_cls:int) -> nn.Module:
    m = models.resnet50(weights="IMAGENET1K_V1")
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m.to(device)

ckpt_path = "/home/ubuntu/robowalle/src/catinator/catinator/ck_resnet50_all_0.0005_cosine.pth"
# set weights_only=True to silence the warning (requires PyTorch ≥2.0)
state = torch.load(ckpt_path, map_location=device, weights_only=True)

orig_n = state["fc.weight"].size(0)
in_f   = state["fc.weight"].size(1)

model = make_resnet50(orig_n)
model.load_state_dict(state)

# ——————— 5 · Slice out a 3-way head ———————
num_test = len(dataset.classes)
new_fc   = nn.Linear(in_f, num_test).to(device)
with torch.no_grad():
    new_fc.weight.copy_(model.fc.weight[test_indices])
    new_fc.bias.copy_(  model.fc.bias[test_indices])
model.fc = new_fc
model.eval()

# ——————— 6 · Zero‐shot Evaluation ———————
criterion = nn.CrossEntropyLoss()
running_loss = running_corrects = total = 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        running_loss     += criterion(out, y).item() * x.size(0)
        running_corrects += (out.argmax(1) == y).sum().item()
        total            += x.size(0)

loss = running_loss / total
acc  = running_corrects / total
print(f"Zero‐shot loss: {loss:.4f},  accuracy: {acc:.4%}")

# ——————— 7 · (Optional) Per‐image Predictions ———————
for img_path, _ in dataset.samples:
    img = dataset.loader(img_path)
    x   = dataset.transform(img).unsqueeze(0).to(device)
    p   = model(x).argmax(1).item()
    print(f"{img_path} ▶ {dataset.classes[p]}")
