# !pip install torch 
# !pip install torchtext 
# !pip install torchvision 
# !pip install transformers
# !pip install tweet-preprocessor

# Libraries
import pandas as pd
import os
import json
import torch

# Training
from torch.utils.data import DataLoader

# Helper functions
import bert
from bert import TweetDataset, SentencePairClassifier, finetuneBERT, predict_to_file, save_result
from dataset import read_csv_datasets


# Global definitions
source_folder = bert.PROJECT_DATA

# BERT model
bert_model = bert.BERT_MODEL

# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


def test_prediction():

    # read train and dev datasets
    train_df = read_csv_datasets(source_folder + 'train.csv')
    dev_df = read_csv_datasets(source_folder + 'dev.csv')
    
    path_to_model = finetuneBERT(train_df, dev_df)

    # load the best model for classification task
    model = SentencePairClassifier(bert_model)
    print("\nLoading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    # use the trained model to predict class labels for the test set
    print("Reading test data...")
    test_df = read_csv_datasets(source_folder + 'test.csv')
    test_set = TweetDataset(test_df, bert.MAXLEN, False, bert_model)
    test_loader = DataLoader(test_set, batch_size=bert.BS, num_workers=2)
    print("Done preprocessing test data.")

    print("Predicting on test data...")
    path_to_output_file = source_folder + 'test-output-probabilities.txt'
    predict_to_file(net=model, device=device, dataloader=test_loader, with_labels=False,
                    result_file=path_to_output_file)
    print("\nTest classification probabilities are available in : {}".format(path_to_output_file))

    save_result(test_df, path_to_output_file)


if __name__ == "__main__":
    test_prediction()