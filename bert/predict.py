import pandas as pd
import json
import torch

import bert


# Global definitions
source_folder = bert.PROJECT_DATA


def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))

    return probs.detach().cpu().numpy()


def predict_to_file(net, device, dataloader, with_labels=True, result_file=source_folder + "output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in dataloader:
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in dataloader:
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()


def extract_class_labels():
    # read class labels from json file
    label = pd.DataFrame()
    with open(source_folder + 'labels.json', 'r', encoding="utf8") as f:
        label = json.load(f)
    f.close()
    return label


def save_result(data, path_to_output_file=source_folder + "output.txt", save_file=source_folder + "output.json"):
    """
    Save predictions on the test data to json file.
    """
    probs = pd.read_csv(path_to_output_file, header=None)[0]  # read prediction probabilities from file
    preds = (probs>=0.5).astype('uint8') # predicted labels using the fixed threshold of 0.5

    labels = extract_class_labels()
    pred_label = {}
    for i in range(len(preds)):
        code = str(preds[i])
        text_id = str(data.iloc[i]['id'])
        pred_label[text_id] = labels[code]
        
    # write predicted labels to json file
    with open(save_file, 'w') as f:
        json.dump(pred_label, f, separators=(',', ':'))
    f.close()

    print("Predictions are available in : {}".format(save_file))