import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image

device="cuda" if torch.cuda.is_available() else "cpu"

path = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(path)
model = SegformerForSemanticSegmentation.from_pretrained(path).to(device)
model.eval()

def segmentation(input_path,output_path):
  image = Image.open(input_path).convert("RGB")
  inputs = processor(images=image, return_tensors="pt").to(device)

  with torch.no_grad():
    outputs = model(**inputs)

  logits = outputs.logits  # (1, num_labels, H, W)
  pred = logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)

  # resize original image to match prediction size
  pred_resized = Image.fromarray(pred.astype(np.uint8)).resize(image.size, Image.NEAREST)
  pred_resized = np.array(pred_resized)#convert back from tensor to array

  overlay=overlay_segmentation(np.array(image),pred_resized)
  cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
  segmentation("input.jpg","output.jpg")

