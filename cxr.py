import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CXR(Dataset):
    """
    Custom PyTorch Dataset for loading chest X-ray (CXR) image and report pairs.

    Each data point includes:
    - An X-ray image
    - A textual 'indication' (used as input text)
    - A textual 'finding' (used as target/output text)

    """

    def __init__(self, data_dir, csv_file, processor, transform):
        """
        Args:
            data_dir (str): Directory where image files are stored.
            csv_file (str): Path to the CSV file containing metadata.
            processor (transformers.Processor): Processor (BLIPProcessor) for image + text tokenization.
            transform (callable): Transformations to apply to images (e.g., resizing, normalization).
        """
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads and processes a single data point.

        Returns:
            dict: A dictionary containing:
                - pixel_values: Tensor of the transformed image
                - input_ids: Tokenized input text (indication)
                - attention_mask: Attention mask for the input
                - labels: Tokenized output text (findings) with padding tokens masked as -100
        """
        try:
            # Build full path to the image
            img_path = os.path.join(self.data_dir, self.data.iloc[idx]["image_path"])

            # Extract text fields
            indication = str(self.data.iloc[idx]["indication"])  # Used as input prompt
            findings = str(self.data.iloc[idx]["findings"])      # Used as ground truth / target

            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")  # Convert to RGB regardless of original format
            if self.transform:
                image = self.transform(image)  # Apply resizing, normalization, etc.
            image = torch.clamp(image, min=0.0, max=1.0)  # Ensure pixel values are in valid range

            # Use processor to prepare image + input text
            inputs = self.processor(
                images=image,
                text=indication,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            # Tokenize target text (findings) as labels
            labels = self.processor.tokenizer(
                findings,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )["input_ids"]

            # Set pad tokens to -100 so they're ignored by the loss function
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            # Return all input tensors (squeezed to remove batch dim), along with labels
            return {
                **{key: val.squeeze(0) for key, val in inputs.items()},
                "labels": labels.squeeze(0)
            }

        except Exception as e:
            # In case of any error (e.g., missing file, bad format), skip the sample
            print(f"Error processing image {img_path}: {e}")
            return None
