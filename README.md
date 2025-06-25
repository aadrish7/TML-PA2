# Model Stealing Attack

A Python implementation that creates a copy of a protected encoder by querying its API and training a student model to mimic its behavior.

## Overview

This code performs a model stealing attack against a protected encoder with query limits and rate limiting. It trains a ResNet-18 model to replicate the victim encoder's 1024-dimensional embeddings.

## Setup

### 1. Create Virtual Environment
```bash
python -m venv model_stealing_env
```

### 2. Activate Virtual Environment
**Windows:**
```bash
model_stealing_env\Scripts\activate
```
**macOS/Linux:**
```bash
source model_stealing_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Files

- `steal_encoder.py` - Main attack implementation
- `requirements.txt` - Python dependencies
- `ModelStealingPub.pt` - Public dataset (required)
- `api_info.json` - Cached API connection details (auto-generated)
- `stolen_encoder.onnx` - Exported trained model (output)

## Configuration

```python
MAX_QUERIES = 30_000    # Query budget limit
TOKEN = "13602610"      # API access token
```

## Usage

1. **Complete setup steps above**
2. **Place `ModelStealingPub.pt` in the same directory**
3. **Run the attack:**
   ```bash
   python stealing_encoder_with_query_limit.py
   ```

## How It Works

1. **Launch API** - Gets access to protected encoder
2. **Load Dataset** - Prepares normalized 32×32 RGB images
3. **Query & Train** - Sends batches of 1000 images, gets embeddings, trains ResNet-18 to match
4. **Export** - Saves model as ONNX file
5. **Submit** - Uploads model for evaluation

## Constraints

- **Query Limit:** 30,000 images maximum
- **Batch Size:** 1000 images per request
- **Rate Limit:** 60-second delay between batches
- **Protection:** Victim encoder uses Bounded Noising (B4B)

## Model Architecture

- **Student:** ResNet-18 with 1024-dim output layer
- **Loss:** Mean Squared Error between embeddings
- **Optimizer:** Adam (lr=1e-4)
- **Training:** Up to 5 epochs or until query budget exhausted

## Output

```
[+] Launched API: seed=12345, port=8080
[+] Queried 1000 images, total used: 1000/30000
Epoch 1/5, Avg. MSE Loss: 0.245631
[+] Exported ONNX to stolen_encoder.onnx
Submission response: {'status': 'success', 'l2_error': 7}
```

## Key Features

- **Budget enforcement** - Prevents exceeding query limits
- **Connection caching** - Reuses API credentials across runs
- **Error handling** - Robust request handling and validation
- **ONNX export** - Standard model format for evaluation
- **Progress tracking** - Real-time loss and query count monitoring

## Notes

- Training time: ~30 minutes for 30K queries (with 60s delays)
- GPU recommended for faster training
- API failures trigger 4-hour cooldown periods
