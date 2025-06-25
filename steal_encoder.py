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

# -----------------------------------------------------------------------------
# Configuration: Query limit
# -----------------------------------------------------------------------------
MAX_QUERIES = 30_000   # Maximum number of images you can query
queried_images = 0      # Counter for images queried so far

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
# 2. Query API for victim embeddings, with query limit enforcement
# -----------------------------------------------------------------------------
def query_api(pil_images, port):
    global queried_images
    batch_size = len(pil_images)

    # Check against query limit
    if queried_images + batch_size > MAX_QUERIES:
        raise RuntimeError(
            f"Query limit exceeded: attempted to query {queried_images + batch_size} images, "
            f"but max allowed is {MAX_QUERIES}."
        )

    # Convert images to base64
    url = f"http://34.122.51.94:{port}/query"
    b64_list = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        b64_list.append(b64)
    payload = json.dumps(b64_list)

    # Send request
    resp = requests.get(url,
                        files={"file": payload},
                        headers={"token": TOKEN})
    if resp.status_code != 200:
        raise RuntimeError(f"API query failed: {resp.status_code} {resp.text}")

    # Parse and count
    reps = resp.json().get("representations", [])
    if len(reps) != batch_size:
        raise RuntimeError(
            f"Unexpected number of representations: expected {batch_size}, got {len(reps)}"
        )

    # Update counter
    queried_images += batch_size
    print(f"[+] Queried {batch_size} images, total used: {queried_images}/{MAX_QUERIES}")

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
    try:
        for epoch in range(1, EPOCHS + 1):
            student.train()
            epoch_loss = 0.0
            for i, (batch_ids, imgs, _) in enumerate(loader):
                pil_batch = [dataset.imgs[idx] for idx in batch_ids.tolist()]

                # Query the victim encoder
                target_np = query_api(pil_batch, PORT)
                target = torch.from_numpy(target_np).to(device)

                imgs = imgs.to(device)
                optimizer.zero_grad()
                outputs = student(imgs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                avg_loss = epoch_loss / (i + 1)
                print(f"Epoch {epoch}/{EPOCHS}, Avg. MSE Loss: {avg_loss:.6f}")

                # Rate limiting between batches
                print(f"[+] Waiting 60 seconds for next batch to avoid rate limit...")
                time.sleep(60)
    except RuntimeError as e:
        print(f"[!] Stopping training: {e}")

    # Export to ONNX if any training happened
    if queried_images > 0:
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        torch.onnx.export(
            student.eval(),
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=12,
            input_names=["x"],
            output_names=["y"]
        )
        print(f"[+] Exported ONNX to {ONNX_PATH}")

        # Check ONNX export
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
    else:
        print("[!] No queries were made; skipping export and submission.")
