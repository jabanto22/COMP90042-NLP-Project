# Libraries
import numpy as np
import random
import copy
import os
import torch

# Models
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

# Training
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight

# Helper functions
import bert
from bert.berttokenizer import TweetDataset
from bert.bertmodel import SentencePairClassifier
from bert.perfplot import plot_train_perf


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility of results
seed = bert.SEED
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# BERT model
bert_model = bert.BERT_MODEL


def compute_class_weights(train_df):
    #compute the class weights
    class_weights = compute_class_weight(class_weight='balanced', 
                                        classes=np.unique(train_df.label.values), 
                                        y=train_df.label.values)

    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights[1]/class_weights[0], dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    return weights


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def evaluate(net, device, criterion, dataloader):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for (seq, attn_masks, token_type_ids, labels) in dataloader:
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels).item()
            count += 1

    return mean_loss / count, mean_acc / count


def train(net, bert_model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):

    best_loss = np.Inf
    best_acc = 0
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        total_loss = 0
        total_acc = 0
        iter = 0
        for (seq, attn_masks, token_type_ids, labels) in train_loader:
            iter += 1
            #Converting these to tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), labels.float())
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

                # Computing accuracy
                acc = get_accuracy_from_logits(logits, labels)
                total_acc += acc

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if iter % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()

            running_loss += loss.item()
            
            if iter % print_every == 0 and iter != 0:  # Print training loss information
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(iter, nb_iterations, ep+1, running_loss / print_every))
                
                total_loss += running_loss
                
                running_loss = 0.0

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(total_acc / len(train_loader))
        
        val_loss, val_acc = evaluate(net, device, criterion, val_loader)  # Compute validation loss and accuracy
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print("\nEpoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}\n".format(best_loss, val_loss))
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_acc = val_acc
            best_ep = ep + 1

    # Saving the model
    path_to_model='models/{}_lr_{}_val_loss_{}_acc_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), round(best_acc, 5), best_ep)
    best_model = '{}_lr_{}_val_loss_{}_acc_{}_ep_{}'.format(bert_model, lr, round(best_loss, 5), round(best_acc, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("Finished training!")
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()

    # Plot performance of model on each epoch
    plot_train_perf(train_losses, val_losses, train_accuracies, val_accuracies, best_model)

    return path_to_model


def finetuneBERT(train_df, dev_df):
    destination_path = 'models'
    try:
        os.makedirs(destination_path)
        print("Directory:", destination_path, "created.")
    except:
        print("Directory:", destination_path, "already exists.")

    # training parameters
    freeze_bert = bert.FREEZE
    maxlen = bert.MAXLEN
    bs = bert.BS
    iters_to_accumulate = bert.ITERS_TO_ACCUMULATE
    lr = bert.LR
    epochs = bert.EPOCHS

    # Read train and validation datasets
    print("Reading training data...")
    train_set = TweetDataset(train_df, maxlen, bert_model)
    print("Reading validation data...")
    val_set = TweetDataset(dev_df, maxlen, bert_model)
    
    # Create instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=2)
    print("Done preprocessing training and development data.")

    net = SentencePairClassifier(bert_model, freeze_bert=freeze_bert)
    net.to(device)

    # model parameters for fine-tuning
    weights = compute_class_weights(train_df)
    criterion = nn.BCEWithLogitsLoss(weight=weights)
    opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # start training for downstream task
    path_to_model = train(net, bert_model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)

    return path_to_model