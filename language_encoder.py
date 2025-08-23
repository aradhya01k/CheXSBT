import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class LanguageEncoder(nn.Module):
    """
    Language Encoder based on BERT.

    This module uses a BERT backbone, followed by additional
    feedforward layers to produce vocabulary-level predictions.

    Returns:
        - output: logits for each token (used for language modeling or generation)
        - text_features: hidden states from the BERT encoder
    """
    def __init__(
        self,
        vocab_size=30522,        # Size of the vocabulary (default: BERT base vocab)
        embed_dim=1024,          # Hidden size of BERT embeddings
        num_decoder_layers=12,   # Number of transformer layers in BERT
        attention_heads=16,      # Number of attention heads
        hidden_layers=2048,      # Size of the intermediate feedforward layers
        dropout_rate=0.2         # Dropout rate for regularization
    ):
        super().__init__()
        
        # ---------------------------------------
        # Define BERT configuration (no pre-trained weights)
        # ---------------------------------------
        self.bert_config = BertConfig(
            hidden_size=embed_dim,
            num_hidden_layers=num_decoder_layers,
            num_attention_heads=attention_heads,
            intermediate_size=hidden_layers,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
        # ---------------------------------------
        # Initialize BERT model from config
        # ---------------------------------------
        self.bert = BertModel(self.bert_config)
        
        # ---------------------------------------
        # Additional feedforward network after BERT
        # ---------------------------------------
        self.fc1 = nn.Linear(embed_dim, hidden_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc3 = nn.Linear(hidden_layers, vocab_size)  # Final projection to vocab logits
        
        # Layer normalization on BERT output
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, input_text, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_text (Tensor): Tokenized input IDs (B, T)
            attention_mask (Tensor): Attention mask for input (B, T)

        Returns:
            output (Tensor): Logits over vocabulary (B, T, vocab_size)
            text_features (Tensor): BERT's last hidden state (B, T, embed_dim)
        """
        # -----------------------
        # BERT encoder forward pass
        # -----------------------
        text_features = self.bert(input_text, attention_mask=attention_mask).last_hidden_state  # Shape: (B, T, embed_dim)
        
        # Apply layer normalization
        text_features = self.norm(text_features)
        
        # -----------------------
        # Feedforward network
        # -----------------------
        x = self.fc1(text_features)   # Shape: (B, T, hidden_layers)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)  # Shape: (B, T, vocab_size)
        
        return output, text_features
