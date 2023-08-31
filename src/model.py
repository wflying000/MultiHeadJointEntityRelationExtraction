import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoModel


class JointModel(nn.Module):
    def __init__(self, config):
        super(JointModel, self).__init__()
        self.encoder =  AutoModel.from_pretrained(config.pretrained_model_path)
        self.entity_label_embeddings = nn.Embedding(config.num_entity_labels, config.entity_label_embed_dim)
        self.relation_embeddings = nn.Embedding(config.num_relation_labels, config.relation_embed_dim)
        self.entity_proj = nn.Linear(config.hidden_size, config.num_entity_labels)
        self.crf = CRF(config.num_entity_labels, batch_first=True)

        self.selection_u = nn.Linear(config.hidden_size + config.entity_label_embed_dim, config.relation_embed_dim)
        self.selection_v = nn.Linear(config.hidden_size + config.entity_label_embed_dim, config.relation_embed_dim)
        self.selection_uv = nn.Linear(config.relation_embed_dim * 2, config.relation_embed_dim)

        rel_weight = torch.full((config.num_relation_labels,), config.rel_weight, dtype=torch.float)
        rel_weight[0] = 1.0
        rel_pos_weight = torch.full((config.num_relation_labels,), config.rel_pos_weight, dtype=torch.float)
        rel_pos_weight[0] = 1.0

        self.register_buffer("rel_weight", rel_weight)
        self.register_buffer("rel_pos_weight", rel_pos_weight)

        self.config = config
    

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        entity_labels = inputs["entity_labels"]

        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output.last_hidden_state

        entity_logits = self.entity_proj(hidden_state)

        entity_loss = -self.crf(
            emissions=entity_logits,
            tags=entity_labels,
            mask=attention_mask,
        )

        entity_prediction = self.crf.decode(entity_logits, attention_mask)

        entity_label_embedding = self.entity_label_embeddings(entity_labels)
        relation_inputs = torch.cat((hidden_state, entity_label_embedding), dim=2)
        seq_len = input_ids.size(1)
        u = self.selection_u(relation_inputs).unsqueeze(2).expand(-1, -1, seq_len, -1)
        v = self.selection_v(relation_inputs).unsqueeze(1).expand(-1, seq_len, -1, -1)
        uv = torch.tanh(self.selection_uv(torch.cat((u, v), dim=-1)))
        selection_logits = torch.einsum('bijh,rh->birj', uv, self.relation_embeddings.weight)
        selection_logits = selection_logits.permute(0, 1, 3, 2)

        relation_labels = inputs["relation_labels"]
        relation_loss = self.compute_relation_loss(selection_logits, relation_labels, attention_mask)

        selection_score = torch.sigmoid(selection_logits)
        relation_prediction = (selection_score > self.config.relation_threshold).type(torch.long)

        loss = entity_loss + relation_loss

        outputs = {
            "loss": loss,
            "entity_loss": entity_loss,
            "relation_loss": relation_loss,
            "entity_prediction": entity_prediction,
            "relation_prediction": relation_prediction,
        }

        return outputs
    
    def compute_relation_loss(self, logits, labels, mask):
        mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        mask = mask.unsqueeze(3).expand(-1, -1, -1, self.config.num_relation_labels)
        labels_one_hot = F.one_hot(labels, self.config.num_relation_labels)
        loss = F.binary_cross_entropy_with_logits(
            input=logits, 
            target=labels_one_hot.float(), 
            weight=self.rel_weight,
            pos_weight=self.rel_pos_weight,
            reduction="none",
        )

        loss = loss.masked_select(mask).sum() / mask.sum()

        return loss

