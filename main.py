import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BlipProcessor, AdamW
from torch.optim.lr_scheduler import StepLR

from cxr import CXR  # Custom dataset class for CXR (chest X-ray) data
from train import train  # Training loop function
from vlm import VisionLanguageModel  # Vision-language model class

# -------------------------
# Define paths to dataset files
# -------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))  # Root directory of the current script
DATA_DIR = os.path.join(ROOT, 'dataset', 'mimic-cxr')  # Path to the image directory
TRAIN_FILE = os.path.join(ROOT, 'dataset', 'train.csv')  # Path to training CSV file
VALIDATION_FILE = os.path.join(ROOT, 'dataset', 'validation.csv')  # Path to validation CSV

# -------------------------
# Main training entry point
# -------------------------
def main():
    # -------------------------
    # Define image transformations
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (standard input size for CNNs)
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale (1-channel) to 3-channel (RGB format)
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # -------------------------
    # Load pre-trained BLIP processor
    # -------------------------
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name, do_rescale=False)
    tokenizer = processor.tokenizer  # Used for text tokenization and loss masking (ignore index)

    # -------------------------
    # Load training and validation datasets
    # -------------------------
    train_dataset = CXR(DATA_DIR, TRAIN_FILE, processor, transform)
    val_dataset = CXR(DATA_DIR, VALIDATION_FILE, processor, transform)

    # -------------------------
    # Create DataLoaders
    # -------------------------
    # Custom collate function to filter out invalid samples (e.g., None values)
    collate_fn = lambda x: [item for item in x if item is not None]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8  # Adjust based on your CPU cores
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8
    )

    # -------------------------
    # Initialize the Vision-Language model
    # -------------------------
    model = VisionLanguageModel(vocab_size=tokenizer.vocab_size)

    # -------------------------
    # Define optimizer with differential learning rates
    # -------------------------
    pretrained_params = list(model.vision_encoder.parameters()) + list(model.language_encoder.parameters())
    new_params = list(set(model.parameters()) - set(pretrained_params))

    optimizer = AdamW([
        {'params': pretrained_params, 'lr': 5e-5},  # Lower LR for pre-trained components
        {'params': new_params, 'lr': 1e-3}           # Higher LR for newly added layers
    ], weight_decay=1e-3)

    # -------------------------
    # Learning rate scheduler
    # -------------------------
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)  # Reduce LR by 15% every epoch

    # -------------------------
    # Train the model
    # -------------------------
    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        epochs=10,
        tokenizer=tokenizer  # Needed for loss ignore index and decoding
    )

    # -------------------------
    # Save the final model checkpoint
    # -------------------------
    torch.save(model.state_dict(), "./models/chexsbt.pth")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
