import torch
import torch.nn as nn
from transformers import AutoModel

class KoELECTRADifficultyRegressor(nn.Module):
    def __init__(self, model_name="monologg/koelectra-base-v3-discriminator"):
        super(KoELECTRADifficultyRegressor, self).__init__()
        self.koelectra = AutoModel.from_pretrained(model_name)
        hidden_size = self.koelectra.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.koelectra(input_ids=input_ids, attention_mask=attention_mask)

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, 1)
        sum_mask = attention_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        cls_output = sum_embeddings / sum_mask

        return self.regressor(cls_output)

