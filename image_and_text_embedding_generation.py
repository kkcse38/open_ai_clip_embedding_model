import os
from PIL import Image
import torch
import clip  # assumes you installed via `pip install git+https://github.com/openai/CLIP.git`
import numpy as np

def load_model(model_name: str = "ViT-B/32", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model = model.to(device)
    model.eval()
    return model, preprocess, device

def embed_text(model, device, texts: list[str]) -> np.ndarray:
    """Return embeddings for a list of text strings."""
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    # Normalize embeddings (common practice)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def embed_image(model, preprocess, device, image_paths: list[str]) -> np.ndarray:
    """Return embeddings for a list of image file paths."""
    # load and preprocess
    images = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in image_paths]
    images = torch.cat(images, dim=0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def get_embeddings(input_item, input_type: str, model, preprocess, device) -> np.ndarray:
    """
    input_type: either "text" or "image"
    input_item: list of strings: if text => list of text strings; if image => list of image file paths
    """
    if input_type == "text":
        return embed_text(model, device, input_item)
    elif input_type == "image":
        return embed_image(model, preprocess, device, input_item)
    else:
        raise ValueError(f"Unknown input_type {input_type}. Use 'text' or 'image'.")

def main():
    # Configure numpy to suppress scientific notation
    #np.set_printoptions(suppress=True, precision=10, linewidth=120)
    
    # Example usage:
    model_name = "ViT-B/32"
    model, preprocess, device = load_model(model_name)

    # Example texts
    texts = ["a photo of white cat lying on the green grass", "a photo of a dog", "An animal running on green grass"]
    text_embs = get_embeddings(texts, "text", model, preprocess, device)
    print("Text embeddings shape:", text_embs.shape)

    # Example images
    image_paths = ["./cat.jpg", "./dog.jpg"]
    image_embs = get_embeddings(image_paths, "image", model, preprocess, device)
    print("Image embeddings shape:", image_embs.shape)

    # Example similarity: compute cosine between first image and first text
    cos_cat = np.dot(image_embs[0], text_embs[0])
    cos_dog = np.dot(image_embs[1], text_embs[1])

    print("\n=== COSINE SIMILARITY ===")
    print("Cosine similarity between first image and first text:", cos_cat)
    print("Cosine similarity between second image and second text:", cos_dog)
    
    print("\n=== EMBEDDING DIMENSIONS ===\n")
    print("Text embedding dimension length:", len(text_embs[0]))
    print("Image embedding dimension length:", len(image_embs[0]))
    
    print("\n=== TEXT EMBEDDINGS ===\n")
    #print("\nText Emb for cat\n,", text_embs[0].tolist())
    #print("Text Emb for dog,", text_embs[1].tolist())
    print("Text Emb for animal,", text_embs[2].tolist())
    
    print("\n=== IMAGE EMBEDDINGS ===\n")
    #print("\n\nImage Emb Cat\n", image_embs[0].tolist())
    #print("\nImage Emb Dog:", image_embs[1].tolist())

if __name__ == "__main__":
    main()
