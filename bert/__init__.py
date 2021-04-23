"""
BERT model fine-tuning for classification task
"""

# BERT model
BERT_MODEL = "bert-base-uncased"

# Model Parameters
FREEZE = False  # update encoder weights and classification layer weights
MAXLEN = 64  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
BS = 16  # batch size
ITERS_TO_ACCUMULATE = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
LR = 3e-5  # learning rate
EPOCHS = 4  # number of training epochs

# DATA FOLDER
PROJECT_DATA = './project-data/'

# SEED
SEED = 1

from bert.berttokenizer import TweetDataset
from bert.bertmodel import SentencePairClassifier
from bert.finetune import finetuneBERT
from bert.predict import predict_to_file, save_result