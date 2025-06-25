import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import onnxruntime as ort
import numpy as np
import json
import io
import sys
import base64
import os
from PIL import Image
import time 

def convert_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


TOKEN = "13602610"
LAUNCH_URL = "http://34.122.51.94:9090/stealing_launch"
CACHE_PATH = "api_info.json"
DATA_PATH = "ModelStealingPub.pt"
ONNX_PATH = "stolen_encoder.onnx"
SUBMIT_URL = "http://34.122.51.94:9090/stealing"

# -----------------------------------------------------------------------------
# 1. Define TaskDataset (must be available before torch.load)
# -----------------------------------------------------------------------------
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []    # list of PIL.Image
        self.labels = []  # list of ints
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return index, img, label


    def __len__(self):
        return len(self.ids)

# -----------------------------------------------------------------------------
# 2. Query API for victim embeddings
# -----------------------------------------------------------------------------
def query_api(pil_images, port):
    """
    Send exactly 1000 PIL images to the API and return a NumPy array
    of shape (1000, 1024) with the protected encoder's embeddings.
    """
    url = f"http://34.122.51.94:{port}/query"
    b64_list = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        b64_list.append(b64)
    payload = json.dumps(b64_list)
    resp = requests.get(url,
                        files={"file": payload},
                        headers={"token": TOKEN})
    if resp.status_code != 200:
        raise RuntimeError(f"API query failed: {resp.status_code} {resp.text}")
    reps = resp.json()["representations"]
    return np.stack(reps, axis=0).astype(np.float32)

# -----------------------------------------------------------------------------
# 3. Main execution starts here
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Launch or load API info
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            info = json.load(f)
        SEED, PORT = info["seed"], info["port"]
        print(f"[+] Reusing cached API: seed={SEED}, port={PORT}")
    else:
        resp = requests.get(LAUNCH_URL, headers={"token": TOKEN})
        answer = resp.json()
        if "detail" in answer:
            print("Failed to launch API:", answer["detail"])
            sys.exit(1)
        SEED, PORT = str(answer["seed"]), str(answer["port"])
        print(f"[+] Launched API: seed={SEED}, port={PORT}")
        with open(CACHE_PATH, "w") as f:
            json.dump({"seed": SEED, "port": PORT}, f)
            print(f"[+] Cached API info to {CACHE_PATH}")

    # Load and transform dataset
    dataset = torch.load(DATA_PATH)
    transform = transforms.Compose([
    transforms.Lambda(convert_rgb),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


    dataset.transform = transform

    loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    # Define student model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = models.resnet18(pretrained=False)
    in_feats = student.fc.in_features
    student.fc = nn.Linear(in_feats, 1024)
    student = student.to(device)

    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Train model to mimic victim encoder
    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        student.train()
        epoch_loss = 0.0
        for i, (batch_ids, imgs, _) in enumerate(loader):
            pil_batch = [dataset.imgs[i] for i in batch_ids.tolist()]
            with torch.no_grad():
                target_np = query_api(pil_batch, PORT)
            target = torch.from_numpy(target_np).to(device)

            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs = student(imgs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"[+] Waiting 60 seconds for next batch to avoid rate limit...")
            time.sleep(60)  # <-- wait between queries
            avg_loss = epoch_loss / (i + 1)  # i starts at 0
            print(f"Epoch {epoch}/{EPOCHS}, Avg. MSE Loss: {avg_loss:.6f}")

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    torch.onnx.export(
        student.eval(),
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=None
    )
    print(f"[+] Exported ONNX to {ONNX_PATH}")

    # Check ONNX export works
    with open(ONNX_PATH, "rb") as f:
        model_bytes = f.read()
    sess = ort.InferenceSession(model_bytes)
    inp = np.random.randn(1, 3, 32, 32).astype(np.float32)
    out_onnx = sess.run(None, {"x": inp})[0]
    assert out_onnx.shape == (1, 1024), f"Expected (1,1024), got {out_onnx.shape}"
    print("[+] ONNX model passed local shape check")

    # Submit
    with open(ONNX_PATH, "rb") as f:
        files = {"file": f}
        headers = {"token": TOKEN, "seed": SEED}
        resp = requests.post(SUBMIT_URL, files=files, headers=headers)
    print("Submission response:", resp.json())