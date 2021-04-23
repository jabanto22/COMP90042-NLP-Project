import pandas as pd
import json
import preprocessor as p

from dataset import PROJECT_DATA


def read_data(filename):
    """
    Read twitter datasets.
    """
    data = pd.DataFrame()
    with open(filename, 'r', encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            tweet_id = line[0]["id_str"]
            tweet = p.clean(line[0]["text"])
            comments = ""
            for row in line:
                # use tweet preprocessor to clean text
                comments += " " + p.clean(row["text"]) + "."
            data = data.append({"id":tweet_id,"text":tweet,"comments":comments}, ignore_index=True)
    f.close()

    return data

    
def read_label(filename):
    """
    Read class labels.
    """
    label = pd.DataFrame()

    with open(filename, 'r', encoding="utf8") as f:
        label = pd.DataFrame.from_dict(json.load(f), orient="index").reset_index()
        label.columns = ["id", "label"]
    f.close()

    return label

    
def merge_data_label(data, label):
    """
    Merge train data with class labels and class label codes for prediction.
    """
    data = pd.merge(data, label, on="id", how="outer")
    data.label = pd.Categorical(data.label)
    class_labels = dict(enumerate(data.label.cat.categories))
    data['label'] = data.label.cat.codes

    # write predicted labels to json file
    with open(PROJECT_DATA + 'labels.json', 'w') as f:
        json.dump(class_labels, f, separators=(',', ':'))
    f.close()

    return data     