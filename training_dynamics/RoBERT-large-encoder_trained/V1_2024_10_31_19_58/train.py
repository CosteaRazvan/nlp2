import os
import numpy as np
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json

from config import ModelConfig
from utils import plot_metrics, plot_confusion_matrix

def set_bn_eval(m):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.eval()


def train(config: ModelConfig, model, train_loader, val_loader, version, class_weights):
    train_dynamics = {}
    
    {"guid": 50325, "logits_epoch_0": [2.943110942840576, -2.2836594581604004], "gold": 0, "device": "cuda:0"}

    # region initialize
    model.to(config.device)
    model.apply(set_bn_eval)  # Freeze batch norm layers

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler("cuda")

    # weights = torch.tensor([0.10915805, 0.07564106, 0.03261642, 0.03791726, 0.74466722])
    # weights = weights.to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
    
    os.makedirs(f"/kaggle/working/results/{config.name_model}", exist_ok=True)
    os.makedirs(f"/kaggle/working/results/{config.name_model}/train_dynamics", exist_ok=True)
    path_for_results = f"/kaggle/working/results/{config.name_model}/"
    # endregion

    for epoch in range(start_epoch+1, config.num_epochs+1):
    
        model.train()
        epoch_loss = 0.0
        total_inputs = 0

        for k, data in enumerate(tqdm(train_loader)):

            # Unpack batch
            inputs, masks, labels = data['ids'], data['masks'], data['target']
            inputs, masks, labels = inputs.to(config.device), masks.to(config.device), labels.to(config.device)
            labels = labels.flatten().long()

            # Remove zero_grad to only clear it during accumulation step
            # Scaling the loss by `gradient_accumulation`
            if config.mix_precision:
                with autocast(device_type='cuda'):
                    outputs, logits = model(inputs, masks)
                    loss = criterion(outputs, labels) / config.gradient_accumulation
                scaler.scale(loss).backward()
            else:
                outputs, logits = model(inputs, masks)
                loss = criterion(outputs, labels) / config.gradient_accumulation
                loss.backward()

            # Save data for train dynamics
            batch_size = inputs.shape[0]
            for sample_id in range(batch_size):
                guid = sample_id + batch_size * k

                sample_logits = logits[sample_id]
                sample_logits = list(sample_logits.detach().cpu().numpy())

                sample_label = labels[sample_id].item()

                train_dynamics = {"guid": guid, f"logits_epoch_{epoch-1}": sample_logits, "gold": sample_label, "device": inputs.device.type}
                print(train_dynamics)

            # Gradient Accumulation Step
            if (k + 1) % config.gradient_accumulation == 0 or (k + 1) == len(train_loader):
                if config.mix_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()  # Update scheduler only after optimizer step
                optimizer.zero_grad(set_to_none=True)  # Reset gradients after accumulation

                # Optional: Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                # Accumulate total loss only after each effective batch (accumulated)
                epoch_loss += loss.item() * config.gradient_accumulation * len(inputs)
                total_inputs += len(inputs)

                with open(f'{path_for_results}metrics_log_V{version}.txt', 'a') as f:
                    f.write(f"Loss: {loss.item() * config.gradient_accumulation:.4f} Batch: {k} of {len(train_loader)}\n")
 
        with open(os.path.join(path_for_results, f"train_dynamics/dynamics_epoch_{epoch-1}.jsonl")) as out_file:
            json.dumps(train_dynamics, out_file)

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