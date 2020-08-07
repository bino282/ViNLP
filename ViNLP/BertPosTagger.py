import torch
from torch import nn
from transformers import *
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import torch.nn.functional as F

class BERTPoSTagger(BertPreTrainedModel):
    def __init__(self, config):
       super(BERTPoSTagger, self).__init__(config)
       self.bert = BertModel(config)
       self.num_labels = config.num_labels
       self.dropout = nn.Dropout(config.hidden_dropout_prob)
       self.fc = nn.Linear(config.hidden_size, config.num_labels)
       self.ignore_index = 0
       self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        embedded  = outputs[0]
        embedded  = self.dropout(embedded)
        # embedded = embedded.permute(1, 0, 2)
        logits = self.fc(embedded)
        outputs = (logits,)

        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index = 0)
        #     loss = loss_fct(predictions.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
