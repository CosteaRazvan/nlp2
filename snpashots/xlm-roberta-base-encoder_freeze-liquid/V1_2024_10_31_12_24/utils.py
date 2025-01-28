import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import seaborn as sns

from src.config import config
from src.dataset import NitroDataset

def plot_confusion_matrix(epoch, conf_matrix, version, path_for_results):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=config.classes, yticklabels=config.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"{path_for_results}conf_matrix_V{version}_e{epoch}.jpg")
    plt.close()

def plot_metrics(epoch, train_loss, val_loss, f1_score, macro_f1_score, version, path_for_results):
    
    def plot_accuracy():
        plt.plot(range(1, 1+len(f1_score)), f1_score, color='blue', label='f1_score')
        plt.plot(range(1, 1+len(macro_f1_score)), macro_f1_score, color='green', label='macro_f1_score')
        plt.title("Acc over " + str(epoch) + " epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.legend(loc="upper left")
        plt.savefig(f"{path_for_results}acc_V{version}_e{epoch}.jpg")
        plt.close()

    def plot_loss():
        plt.plot(range(1, 1+len(train_loss)), train_loss, color='blue', label='train_loss')
        plt.plot(range(1, 1+len(val_loss)), val_loss, color='green', label='val_loss')
        plt.title("Loss over " + str(epoch) + " epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper left")
        plt.savefig(f"{path_for_results}loss_V{version}_e{epoch}.jpg")
        plt.close()

    plot_loss()
    plot_accuracy()


def weighted_accuracy(all_labels, sample_labels, predicted_labels):
    categories = np.unique(all_labels)
    cls_weights = class_weight.compute_class_weight(class_weight='balanced', classes=categories, y=all_labels)
    cls_weights_dict = {categories[i]: cls_weights[i] for i in range(len(categories))}
    val_sample_weights = class_weight.compute_sample_weight(class_weight=cls_weights_dict, y=sample_labels)
    return accuracy_score(sample_labels, predicted_labels, sample_weight=val_sample_weights)

def get_data(root):
    train_data = pd.read_csv(root)
    text_lists = [train_data[train_data["Final Labels"]==clasa] for clasa in np.unique(train_data["Final Labels"])]

    nonof = text_lists[2].iloc[:5000]
    train_data = pd.concat([text_lists[0], text_lists[1], nonof, text_lists[3], text_lists[4]])

    X, y = train_data['Text'].tolist(), np.array(train_data['Final Labels'].tolist())

    X_replaced = np.array([text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș") for text in X])

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # # Calculate the total number of samples for the validation set
    # n = int(0.05 * len(y)) # there are not enough samples for the reporting class in roder to be equal
    # # Calculate the number of classes
    # num_classes = len(np.unique(y))
    # # Calculate the number of samples per class for the validation set
    # n_per_class = n // num_classes

    # X_train, y_train, X_val, y_val = [], [], [], []

    # for class_label in np.unique(y):
    #     class_indices = np.where(y == class_label)[0]
    #     np.random.shuffle(class_indices)
    #     val_indices = class_indices[:n_per_class]
    #     train_indices = class_indices[n_per_class:]

    #     X_val.extend(X_replaced[val_indices])
    #     y_val.extend(y[val_indices])
    #     X_train.extend(X_replaced[train_indices])
    #     y_train.extend(y[train_indices])

    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_val = np.array(X_val)
    # y_val = np.array(y_val)

    X_train, X_val, y_train, y_val = train_test_split(X_replaced, y, test_size=0.20, random_state=42, shuffle=True)


    return X_train, X_val, y_train, y_val

def get_data_test(root):
    test_data = pd.read_csv(os.path.join(root, 'train_data.csv'))
    X = test_data['Text'].tolist()
    X_replaced = np.array([text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș") for text in X])
    return X_replaced

def get_dataloader(config, tokenizer, root):
    X_train, X_val, y_train, y_val = get_data(root)

    train_data = NitroDataset(X_train, y_train, tokenizer)
    val_data = NitroDataset(X_val, y_val, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader

def get_test_loader(tokenizer, root):
    X = get_data_test(root)
    y = np.zeros(X.shape)
    test_data = NitroDataset(X, y, tokenizer)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return test_loader

def submit_form(predictions):
    a = pd.DataFrame(enumerate(predictions), columns=['Id', 'Label'])
    a.to_csv("SUBMISIE.csv", index=False)

def inference(config, model, test_loader, load_model=True):
    if load_model:
        checkpoint = torch.load('model_e1_mBERT.pth')
        model.load_state_dict(checkpoint['model'])

    model.to(config.device)
    model.eval()
    predictions = []
    for k, data in enumerate(tqdm(test_loader)):
        inputs, masks, labels = data['ids'], data['masks'], data['target']
        inputs, masks, labels = inputs.to(config.device), masks.to(config.device), labels.to(config.device)
        labels = labels.flatten()
        labels = labels.long()
        
        with torch.no_grad():
            outputs = model(inputs, masks)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.append(config.classes[y_pred[0]])
        print(f'Done with {k}/{len(test_loader)}')

    return predictions