# Library
import pandas as pd

# Helper functions
from dataset.twitterdata import read_data, read_label, merge_data_label
from dataset import PROJECT_DATA


def save_data_to_csv():
    """
    Read and extract datasets from files.
    """
    # read data (jsonl files)
    train_data = read_data(PROJECT_DATA + 'train.data.jsonl')
    dev_data = read_data(PROJECT_DATA + 'dev.data.jsonl')
    test_data = read_data(PROJECT_DATA + 'test.data.jsonl')
    covid_data = read_data(PROJECT_DATA + 'covid.data.jsonl')

    # read labels (json files)
    train_label = read_label(PROJECT_DATA + 'train.label.json')
    dev_label = read_label(PROJECT_DATA + 'dev.label.json')

    # merge data with class labels
    train_data = merge_data_label(train_data, train_label)
    dev_data = merge_data_label(dev_data, dev_label)

    # write filetered data to csv
    open(PROJECT_DATA + 'train.csv','w', newline='').write(train_data.to_csv(index=False))
    open(PROJECT_DATA + 'dev.csv','w', newline='').write(dev_data.to_csv(index=False))
    open(PROJECT_DATA + 'test.csv','w', newline='').write(test_data.to_csv(index=False))
    open(PROJECT_DATA + 'covid.csv','w', newline='').write(covid_data.to_csv(index=False))


def check_input_files(filename):
    """
    Check input files if they exist.
    """
    try:
        f = open(filename,'r')
        f.close()
    except:
        # read and process all input datasets
        save_data_to_csv()


def read_csv_datasets(filename):
    # check if input files exist
    check_input_files(filename)

    # read datasets
    df = pd.read_csv(filename)

    return df