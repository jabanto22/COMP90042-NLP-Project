import matplotlib.pyplot as plt


def plot_train_perf(train_losses, val_losses, train_accuracies, val_accuracies, best_model):
    """
    Create a plot analysis of model loss and accuracy across training epochs.
    """
    acc = train_accuracies
    val_acc = val_accuracies
    loss = train_losses
    val_loss = val_losses

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    path_to_fig = 'models/accuracy-' + best_model + '.png'
    fig.savefig(path_to_fig,dpi=300)
    fig.show()
    plt.close()