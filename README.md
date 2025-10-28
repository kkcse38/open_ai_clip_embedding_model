# Image and Text Embedding Generation (CLIP)

Generate vector embeddings from images and text using OpenAI's CLIP model for semantic similarity comparison.

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision pillow numpy
```

### Basic Usage

```python
from image_and_text_embedding_generation import load_model, get_embeddings
import numpy as np

# Load model
model, preprocess, device = load_model("ViT-B/32")

# Generate embeddings
texts = ["a photo of a cat", "a photo of a dog"]
text_emb = get_embeddings(texts, "text", model, preprocess, device)

images = ["./cat.jpg", "./dog.jpg"]
image_emb = get_embeddings(images, "image", model, preprocess, device)

# Calculate similarity (0-1 scale, higher = more similar)
similarity = np.dot(text_emb[0], image_emb[0])
print(f"Similarity: {similarity:.3f}")
```

### Run Demo

```bash
python image_and_text_embedding_generation.py
```

## 📚 Key Functions

### `load_model(model_name="ViT-B/32")`
Loads CLIP model. Returns `(model, preprocess, device)`

### `get_embeddings(input_item, input_type, model, preprocess, device)`
- `input_type`: `"text"` or `"image"`
- `input_item`: list of strings (texts) or paths (images)
- Returns: normalized embeddings (512-dim vectors)

## 💡 Use Cases

- **Image Search**: Find images matching text descriptions
- **Classification**: Zero-shot image classification
- **Similarity**: Compare semantic similarity between images/text
- **Recommendations**: Content-based image recommendations

## ⚙️ Model Options

- `"ViT-B/32"` - Fast, balanced (default)
- `"ViT-B/16"` - Better accuracy, slower
- `"ViT-L/14"` - Best accuracy, slowest

## 📊 Output

Embeddings are 512-dimensional normalized vectors. Similarity scores range from -1 to 1 (higher = more similar).

## 🔧 Requirements

- Python 3.8+
- PyTorch
- CLIP (from GitHub)
- PIL, NumPy

---

**Note**: Ensure sample images (`cat.jpg`, `dog.jpg`) are in the same directory before running the demo.
