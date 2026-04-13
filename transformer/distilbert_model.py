import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class RuleformerDistilBERT(nn.Module):
    def __init__(self, n_relations, d_model=768, max_seq_len=512):
        super().__init__()

        # Load DistilBERT (pretrained or from scratch)
        config = DistilBertConfig(
            vocab_size=n_relations,   # your KG relations as "tokens"
            max_position_embeddings=max_seq_len,
            hidden_dim=3072,
            dim=d_model,
            n_heads=12,
            n_layers=6,              # DistilBERT default (vs BERT's 12)
        )
        self.encoder = DistilBertModel(config)  # random init for KG task

        # Output head: predict next relation in the rule chain
        self.output_proj = nn.Linear(d_model, n_relations)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # outputs.last_hidden_state: [batch, seq_len, d_model]
        logits = self.output_proj(outputs.last_hidden_state)
        return logits  # [batch, seq_len, n_relations]