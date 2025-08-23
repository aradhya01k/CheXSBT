from tqdm import tqdm
import torch
import torch.nn as nn

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, tokenizer):
    """
    Trains a multimodal vision-language model over multiple epochs.
    
    Args:
        model (nn.Module): Vision-language model.
        train_dataloader (DataLoader): DataLoader for training set.
        val_dataloader (DataLoader): DataLoader for validation set.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        tokenizer: Tokenizer used to retrieve pad token ID for loss masking.
    """
    
    # Select computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Move model to the selected device
    model.to(device)

    # Define loss function and ignore padding tokens during loss computation
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Progress bar for training loop
        loop = tqdm(train_dataloader, leave=True)

        for batch in loop:
            # Extract and move batch data to device
            images = torch.stack([b["pixel_values"] for b in batch]).to(device)         # [B, C, H, W]
            input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)         # [B, T]
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)               # [B, T]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)  # [B, T, Vocab]

            # Compute loss (reshape to match CrossEntropy requirements)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
            optimizer.step()

            # Update loss and progress bar
            total_loss += loss.item()
            loop.set_description(f"Loss: {loss.item():.4f}")

        # Update learning rate after each epoch
        scheduler.step()

        # Print epoch summary
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # --------------------------
        # Validation Loop
        # --------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                # Move validation batch to device
                images = torch.stack([b["pixel_values"] for b in batch]).to(device)
                input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
                attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)

                # Forward pass (no gradient computation)
                outputs = model(images, input_ids, attention_mask)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()

        # Report validation performance
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
