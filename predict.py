import torch
import os
from PIL import Image, ImageDraw
import numpy as np
from model import get_model

# 📁 Paths
image_dir = "/datasets/tdt4265/Poles2025/roadpoles_v1/test/images"
output_dir = "results/predictions"

os.makedirs(output_dir, exist_ok=True)

# ⚙️ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🤖 Load model
model = get_model(num_classes=2)
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()

# 🔎 Threshold
score_threshold = 0.3

# 🔁 Loop over images
for img_name in os.listdir(image_dir):

    img_path = os.path.join(image_dir, img_name)

    image = Image.open(img_path).convert("RGB")
    image_resized = image.resize((640, 640))

    img_tensor = torch.from_numpy(np.array(image_resized)).permute(2,0,1).float()/255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    preds = outputs[0]

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    draw = ImageDraw.Draw(image_resized)

    for box, score in zip(boxes, scores):

        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{score:.2f}", fill="red")

    save_path = os.path.join(output_dir, img_name)
    image_resized.save(save_path)

print("✅ Predictions saved")