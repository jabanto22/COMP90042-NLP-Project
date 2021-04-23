# Model
import torch.nn as nn
from transformers import AutoModel

# Training precision
from torch.cuda.amp import autocast

import bert

class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model=bert.BERT_MODEL, freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        
        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768
        # output dimension is 1 because we're working with a binary classification problem
        self.cls_layer = nn.Linear(768, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        output = self.bert_layer(input_ids, attn_masks, token_type_ids)
        
        # the last layer hidden-state of the first token of the sequence (classification token) 
        # further processed by a Linear layer and a Tanh activation function.
        logits = self.dropout(output['pooler_output'])
        
        # Feeding to the classifier layer 
        logits = self.cls_layer(logits)

        return logits