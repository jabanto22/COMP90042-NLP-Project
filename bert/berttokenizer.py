from transformers import AutoTokenizer
from torch.utils.data import Dataset

import bert

class TweetDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model=bert.BERT_MODEL):

        self.data = data
        self.with_labels = with_labels 
        
        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  
        self.maxlen = maxlen
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'text'])
        sent2 = str(self.data.loc[index, 'comments'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels (train and validation dataset)
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:  # for test set that has no labels
            return token_ids, attn_masks, token_type_ids