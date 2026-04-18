from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

BASE = "/datasets/tdt4265/Poles2025/roadpoles_v1"

IMG_FOLDER = os.path.join(BASE, "train/images")
LABEL_FOLDER = os.path.join(BASE, "train/labels")

# handle .PNG and .png
images = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(".png")]

img_name = images[0]

img_path = os.path.join(IMG_FOLDER, img_name)
label_path = os.path.join(LABEL_FOLDER, img_name.replace(".PNG", ".txt").replace(".png", ".txt"))

print("Using:", img_path)

# open image
img = Image.open(img_path)
w, h = img.size

# plot
fig, ax = plt.subplots(1)
ax.imshow(img)

# draw boxes
if os.path.exists(label_path):
    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())

            x = x * w
            y = y * h
            bw = bw * w
            bh = bh * h

            x1 = x - bw / 2
            y1 = y - bh / 2

            rect = patches.Rectangle(
                (x1, y1),
                bw,
                bh,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)
else:
    print("No label found")

plt.axis("off")
plt.show()