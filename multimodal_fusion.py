import torch
import torch.nn as nn

# --------------------------------------------
# Multimodal Fusion Network
# --------------------------------------------
# Fuses vision features
# with language features in a shared space.
# Outputs token-level logits for language generation tasks.
# --------------------------------------------

class MultimodalFusion(nn.Module):
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=1024, output_dim=30522):
        """
        Args:
            vision_dim (int): Dimensionality of vision encoder output.
            text_dim (int): Dimensionality of language encoder hidden states.
            hidden_dim (int): Size of the hidden layer in fusion.
            output_dim (int): Size of the final output (usually vocab size).
        """
        super().__init__()

        # -------------------------------
        # Project vision features into text embedding space
        # -------------------------------
        self.vision_fc = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),  # Map to hidden space
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, text_dim)     # Final projection to match text_dim
        )

        # -------------------------------
        # Fusion refinement block
        # -------------------------------
        self.fusion_fc = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU()
        )

        # -------------------------------
        # Final projection to vocabulary space
        # -------------------------------
        self.output_fc = nn.Linear(text_dim, output_dim)

    def forward(self, vision_features, text_features):
        """
        Args:
            vision_features: Tensor of shape [batch_size, vision_dim]
            text_features: Tensor of shape [batch_size, seq_len, text_dim]

        Returns:
            output: Tensor of shape [batch_size, seq_len, vocab_size]
        """
        # -----------------------------------------
        # Project vision features to match text space
        # -----------------------------------------
        vision_proj = self.vision_fc(vision_features)  # [B, text_dim]

        # -----------------------------------------
        # Expand vision features across the sequence length
        # to match shape of text features
        # -----------------------------------------
        vision_proj = vision_proj.unsqueeze(1).repeat(1, text_features.size(1), 1)  # [B, T, text_dim]

        # -----------------------------------------
        # Fuse by element-wise addition
        # -----------------------------------------
        fused = vision_proj + text_features  # [B, T, text_dim]

        # -----------------------------------------
        # Optional refinement through another layer
        # -----------------------------------------
        fused = self.fusion_fc(fused)  # [B, T, text_dim]

        # -----------------------------------------
        # Final projection to vocab space (logits)
        # -----------------------------------------
        output = self.output_fc(fused)  # [B, T, vocab_size]

        return output
