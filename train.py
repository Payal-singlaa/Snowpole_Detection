import torch
from torch.utils.data import DataLoader
from eval_utils import evaluate_map

from dataset import SnowPoleDataset
from model import get_model

# 📁 Pfade
image_dir = "/datasets/tdt4265/Poles2025/roadpoles_v1/train/images"
label_dir = "/datasets/tdt4265/Poles2025/roadpoles_v1/train/labels"

# 📦 Dataset
dataset = SnowPoleDataset(image_dir, label_dir)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# 🚀 Device (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# 🤖 Model
model = get_model(num_classes=2)
model.to(device)

# ⚙️ Optimizer
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

num_epochs = 5

# 🔁 Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    print(f"Epoch {epoch} start")

    for images, targets in train_loader:

        images = [img.to(device) for img in images]

        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device)
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Loss:", total_loss)

# 💾 speichern
torch.save(model.state_dict(), "snowpole_model.pth")
print("Done")

precision, recall, ious = evaluate(model, val_loader, device)
scores, labels = evaluate_map(model, val_loader, device)
print("Done evaluation")

print("Precision:", precision)
print("Recall:", recall)