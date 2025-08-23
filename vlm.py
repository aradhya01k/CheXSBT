from vision_encoder import VisionEncoder
from language_encoder import LanguageEncoder
from multimodal_fusion import MultimodalFusion

import torch
import torch.nn as nn

class VisionLanguageModel(nn.Module):
    """
    A multimodal vision-language model that combines visual features from images
    with textual features from input text, then fuses them to produce token logits.
    """
    def __init__(self, vocab_size=30522, embed_dim=768):
        super().__init__()
        
        # Vision encoder extracts visual embeddings from images
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        
        # Language encoder extracts text embeddings from input tokens
        self.language_encoder = LanguageEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        
        # Fusion module aligns and combines vision and text embeddings, outputting vocab logits
        self.fusion = MultimodalFusion(vision_dim=embed_dim, text_dim=embed_dim, output_dim=vocab_size)
    
    def forward(self, images, input_text, attention_mask):
        """
        Forward pass through the model.

        Args:
            images (Tensor): Batch of images [batch_size, channels, height, width].
            input_text (Tensor): Tokenized input text IDs [batch_size, seq_len].
            attention_mask (Tensor): Attention mask for text input [batch_size, seq_len].

        Returns:
            Tensor: Token logits over vocabulary [batch_size, seq_len, vocab_size].
        """
        # Extract visual features: [batch_size, embed_dim]
        vision_features = self.vision_encoder(images)
        
        # Extract text features and logits from language encoder:
        # decoder_output (logits): [batch_size, seq_len, vocab_size]
        # text_features (embeddings): [batch_size, seq_len, embed_dim]
        decoder_output, text_features = self.language_encoder(input_text, attention_mask)
        
        # Fuse visual and textual embeddings to get final token logits
        fused_output = self.fusion(vision_features, text_features)  # [batch_size, seq_len, vocab_size]
        
        return fused_output
