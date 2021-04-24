# Libraries
import pandas as pd
import json
import torch

# Model input
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


def covid_prediction():

    path_to_model = "models/bert-base-uncased_lr_3e-05_val_loss_0.5399_acc_0.88514_ep_2.pt"

    # load the best model for classification task
    model = SentencePairClassifier(bert_model)
    print("\nLoading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.to(device)

    # use the trained model to predict class labels for the test set
    print("Reading test data...")
    covid_df = read_csv_datasets(source_folder + 'covid.csv')
    covid_set = TweetDataset(covid_df, bert.MAXLEN, False, bert_model)
    covid_loader = DataLoader(covid_set, batch_size=bert.BS, num_workers=2)
    print("Done preprocessing covid data.")

    print("Predicting on covid data...")
    path_to_output_file = source_folder + 'covid-output-probabilities.txt'
    predict_to_file(net=model, device=device, dataloader=covid_loader, with_labels=False,
                    result_file=path_to_output_file)
    print("\nTest classification probabilities are available in : {}".format(path_to_output_file))

    path_to_output_json = source_folder + 'covid-output.json'
    save_result(covid_df, path_to_output_file, path_to_output_json)


if __name__ == "__main__":
    covid_prediction()