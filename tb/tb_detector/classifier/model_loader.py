import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---- Model ----
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 0 = Normal, 1 = TB

state_dict = torch.load("D:/tb/tb_detector/tb_classifier_final.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
torch.set_grad_enabled(False)

# ---- Inference transforms (match training): NO Normalize here ----
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CLASS_NAMES = ["Normal", "Tuberculosis"]

def _predict_tensor(img_tensor, threshold=0.70):
    """Internal: img_tensor is 1xCxHxW on CPU."""
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)[0]        # shape: (2,)
    conf, idx = torch.max(probs, dim=0)             # top-1
    # Optional safety threshold to reduce false TB flags
    if idx.item() == 1 and conf.item() < threshold:
        # Below threshold for TB → treat as Normal
        idx = torch.tensor(0)
        conf = 1.0 - probs[1]
    return {
        "label_index": idx.item(),
        "label": CLASS_NAMES[idx.item()],
        "confidence": float(conf.item()),
        "probs": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(2)}
    }

def predict_image(path, threshold=0.70):
    """Predict from an image file path."""
    image = Image.open(path).convert("RGB")
    img_tensor = infer_transform(image).unsqueeze(0)
    return _predict_tensor(img_tensor, threshold=threshold)

def predict_fileobj(fileobj, threshold=0.70):
    """Predict directly from uploaded file (in-memory)."""
    image = Image.open(io.BytesIO(fileobj.read())).convert("RGB")
    img_tensor = infer_transform(image).unsqueeze(0)
    return _predict_tensor(img_tensor, threshold=threshold)
