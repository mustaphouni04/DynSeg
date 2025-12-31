import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def save_training_image(image_tensor, mask_tensor, text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Convert image like matplotlib ---
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)  # keep float [0,1]

    # --- Convert mask like matplotlib ---
    mask = mask_tensor.detach().cpu().numpy().squeeze()
    mask = np.clip(mask, 0, 1)

    print(f"[INFO] mask max={mask.max():.4f} min={mask.min():.4f} mean={mask.mean():.4f}")

    # Convert to binary mask same way you threshold
    bin_mask = (mask > 0.5).astype(np.float32)

    # === SAME VALUES FROM ListedColormap ===
    overlay_color = np.array([1.0, 0.4, 0.4])  # RGB
    alpha = 0.5

    # === Matplotlib-like blending ===
    img_out = img.copy()
    img_out[bin_mask == 1] = (
        (1 - alpha) * img_out[bin_mask == 1] +
        alpha * overlay_color
    )

    # Convert to uint8 after blending
    img_out = (img_out * 255).astype(np.uint8)

    pil_img = Image.fromarray(img_out)

    # === Text ===
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # measure text height
    bbox = font.getbbox(text)
    text_h = bbox[3] - bbox[1] + 10

    # create expanded final image
    final_img = Image.new("RGB", (pil_img.width, pil_img.height + text_h), (255, 255, 255))
    final_img.paste(pil_img, (0, 0))

    draw = ImageDraw.Draw(final_img)
    draw.text((10, pil_img.height + 5), text, fill="black", font=font)

    final_img.save(output_path)
