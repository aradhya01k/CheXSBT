import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BlipProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

from vlm import VisionLanguageModel
from cxr import CXR

# --------------------------------------------
# Metric Computation Function
# --------------------------------------------
def compute_metrics(preds, targets):
    """
    Computes BLEU, METEOR, and ROUGE-L scores between predictions and references.
    
    Args:
        preds (List[str]): Predicted texts.
        targets (List[str]): Ground truth texts.
        
    Returns:
        dict: Averaged scores for each metric.
    """
    rouge = Rouge()
    smoothie = SmoothingFunction().method4  # For BLEU smoothing

    bleu1, bleu2, bleu3, bleu4, meteor, rouge_l_f1 = [], [], [], [], [], []

    for pred, ref in zip(preds, targets):
        ref_tokens = ref.split()
        pred_tokens = pred.split()

        bleu1.append(sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu2.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu3.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu4.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

        meteor.append(single_meteor_score(ref, pred))
        rouge_score = rouge.get_scores(pred, ref)[0]['rouge-l']['f']
        rouge_l_f1.append(rouge_score)

    return {
        "BLEU-1": sum(bleu1) / len(bleu1),
        "BLEU-2": sum(bleu2) / len(bleu2),
        "BLEU-3": sum(bleu3) / len(bleu3),
        "BLEU-4": sum(bleu4) / len(bleu4),
        "METEOR": sum(meteor) / len(meteor),
        "ROUGE-L": sum(rouge_l_f1) / len(rouge_l_f1)
    }

# --------------------------------------------
# Main Testing and Evaluation Pipeline
# --------------------------------------------
def test_and_evaluate():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # Define data and model paths
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT, 'dataset', 'mimic-cxr')
    TEST_FILE = os.path.join(ROOT, 'dataset', 'test.csv')
    MODEL_PATH = os.path.join(ROOT, 'models', 'chexsbt.pth')

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
    ])

    # Initialize BLIP processor and tokenizer
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", do_rescale=False)
    tokenizer = processor.tokenizer

    # Load test dataset
    test_dataset = CXR(DATA_DIR, TEST_FILE, processor, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: [i for i in x if i is not None])

    # Initialize and load model
    model = VisionLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    references = []

    # Inference loop
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            image = torch.stack([b["pixel_values"] for b in batch]).to(device)
            input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)

            # Forward pass through VLM
            outputs = model(image, input_ids, attention_mask)  # [1, seq_len, vocab_size]
            predicted_ids = torch.argmax(outputs, dim=-1)       # Get token indices with max logits

            # Decode predictions and ground truths
            for i, pred in enumerate(predicted_ids):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True)

                # Remove ignored (-100) tokens before decoding target
                target_text = batch[i]["labels"]
                target_decoded = tokenizer.decode(target_text[target_text != -100], skip_special_tokens=True)

                predictions.append(pred_text)
                references.append(target_decoded)

    # Save results to files
    with open("test_predictions.txt", "w") as f:
        for pred in predictions:
            f.write(pred.strip() + "\n")

    with open("test_references.txt", "w") as f:
        for ref in references:
            f.write(ref.strip() + "\n")

    # Compute evaluation scores
    scores = compute_metrics(predictions, references)

    # Display results
    print("\nEvaluation Metrics:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


# Entry point
if __name__ == "__main__":
    test_and_evaluate()
