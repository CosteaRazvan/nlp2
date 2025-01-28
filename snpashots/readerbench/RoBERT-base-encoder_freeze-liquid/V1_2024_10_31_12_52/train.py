import os
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import ModelConfig
from src.utils import plot_metrics, plot_confusion_matrix


def train(config: ModelConfig, model, train_loader, val_loader, version):

    # region initialize
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler("cuda")

    weights = torch.tensor([0.10915805, 0.07564106, 0.03261642, 0.03791726, 0.74466722])
    weights = weights.to(config.device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_losses, val_losses = [], []
    f1_scores = []
    macro_f1_scores = []

    start_epoch = 0

    optimizer.zero_grad()

    train_steps_per_epoch = len(train_loader)
    num_train_steps = config.num_epochs * train_steps_per_epoch
    num_warmup_steps = int(0.1 * num_train_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)
    
    os.makedirs(f"/mnt/storage/Code/nlp/results/{config.name_model}", exist_ok=True)
    path_for_results = f"/mnt/storage/Code/nlp/results/{config.name_model}/"
    # endregion

    

    for epoch in range(start_epoch+1, config.num_epochs+1):\
    
        model.train()
        epoch_loss = 0.0
        total_inputs = 0

        for k, data in enumerate(tqdm(train_loader)):

            inputs, masks, labels = data['ids'], data['masks'], data['target']
            inputs, masks, labels = inputs.to(config.device), masks.to(config.device), labels.to(config.device)
            labels = labels.flatten()
            labels = labels.long()
            
            

            optimizer.zero_grad(set_to_none=True)

            if config.mix_precision:
                with autocast(device_type='cuda'):
                    outputs = model(inputs, masks)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
            else:
                outputs = model(inputs, masks)
                loss = criterion(outputs, labels)
                loss.backward()

            if (k + 1) % config.gradient_accumulation == 0 or (k + 1) == len(train_loader):
                if config.mix_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)



            epoch_loss += loss.item() * len(inputs)
            total_inputs += len(inputs)
            with open(f'{path_for_results}metrics_log_V{version}.txt', 'a') as f:
                f.write(f"Loss: {loss.item()} Batch: {k} of {len(train_loader)}\n")


        epoch_loss = epoch_loss / total_inputs
        train_losses.append(epoch_loss)

        with open(f'{path_for_results}metrics_log_V{version}.txt', 'a') as f:
            f.write(f"\n  Train epoch {epoch} loss: {epoch_loss}\n")
        print(f"Train epoch {epoch} loss: {epoch_loss}")

        model.eval()
        epoch_loss = 0.0
        total_inputs = 0
        all_labels = []
        all_predictions = []

        for data in val_loader:
            inputs, masks, labels = data['ids'], data['masks'], data['target']
            inputs, masks, labels = inputs.to(config.device), masks.to(config.device), labels.to(config.device)
            labels = labels.flatten()
            labels = labels.long()


            with torch.no_grad():
                outputs = model(inputs, masks)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item() * len(inputs)
                
                predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_labels.extend(labels)
                all_predictions.extend(predicted)
                total_inputs += len(inputs)

        # Calculate average validation loss
        epoch_loss = epoch_loss / total_inputs
        val_losses.append(epoch_loss)

        # Generate classification report
        class_report = classification_report(all_labels, all_predictions, target_names=config.classes, output_dict=True)
        class_report_str = classification_report(all_labels, all_predictions, target_names=config.classes)
        f1_score = class_report['weighted avg']['f1-score']
        macro_f1_score = class_report['macro avg']['f1-score']
        f1_scores.append(f1_score)
        macro_f1_scores.append(macro_f1_score)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Log metrics
        with open(f'{path_for_results}metrics_log_V{version}.txt', 'a') as f:
            f.write(f"  Val epoch {epoch} loss: {epoch_loss} F1: {f1_score} Macro F1: {macro_f1_score}\n")
            f.write(f"Classification Report:\n{class_report_str}\n\n")
        print(f"Val epoch {epoch} loss: {epoch_loss} F1: {f1_score} Macro F1: {macro_f1_score}")

        plot_confusion_matrix(epoch, conf_matrix, version, path_for_results)


        if epoch > 2:
            plot_metrics(epoch, train_losses, val_losses, f1_scores, macro_f1_scores, version, path_for_results)
            path = f"models/model_e{epoch}_{config.name_model}_V{version}.pth"
            torch.save({
                'epoch': config.num_epochs,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'f1_scores': f1_scores,
                'macro_f1_scores': macro_f1_scores,
                'config': config.__dict__
            }, path)