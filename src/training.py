"""
Module for training and validation routines for a CNN model, including functions for
training, validation, and calculating performance metrics such as F1 score, recall,
and precision.
"""

import os
import torch
import wandb
from .config import PATIENCE_THRESHOLD
from sklearn.metrics import multilabel_confusion_matrix


def train(model, train_loader, criterion, optimizer, epoch, logger, step):
    """
    Trains the model for one epoch.

    Parameters:
        model (torch.nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training set.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epoch (int): Current epoch number.
        logger (callable): Logging function for recording loss and accuracy.
        step (int): Interval at which to log loss and accuracy.

    Returns:
        None
    """
    model.train()
    running_loss = 0.0
    correct = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(model.device), target.to(model.device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(output, 1)
        correct += (predictions == target).float().mean().item()
        running_loss += loss.item()
        if i % step == step - 1:
            accuracy = correct / step
            loss = running_loss / step
            log_step = epoch * len(train_loader) + i
            logger({"train/accuracy": accuracy, "train/loss": loss}, step=log_step)
            running_loss = 0.0
            correct = 0


class ValidationMetrics:
    """
    Class to calculate and store various validation metrics including F1 score, recall,
    precision, false acceptance, false rejection, and accuracy.

    Parameters:
        confusion_matrix (np.ndarray): Multi-label confusion matrix for the validation data.

    Attributes:
        f1 (float): F1 score of the model.
        recall (float): Recall score of the model.
        precision (float): Precision score of the model.
        false_acceptance (float): False acceptance rate.
        false_rejection (float): False rejection rate.
        accuracy (float): Accuracy of the model.
    """

    def __init__(self, confusion_matrix):
        true_neg = confusion_matrix[0, 0, 0]
        false_neg = confusion_matrix[0, 1, 0]
        true_pos = confusion_matrix[0, 1, 1]
        false_pos = confusion_matrix[0, 0, 1]

        real_pos = true_pos + false_neg
        real_neg = true_neg + false_pos
        model_pos = true_pos + false_pos
        total = true_pos + true_neg + false_pos + false_neg

        if real_pos != 0:
            self.false_rejection = false_neg / real_pos
            self.recall = true_pos / real_pos
        else:
            self.false_rejection = self.recall = 0

        if real_neg != 0:
            self.false_acceptance = false_pos / real_neg
        else:
            self.false_acceptance = 0

        if model_pos != 0:
            self.precision = true_pos / model_pos
        else:
            self.precision = 0

        if model_pos + real_pos != 0:
            self.f1 = (2 * true_pos) / (model_pos + real_pos)
        else:
            self.f1 = 0

        if total != 0:
            self.accuracy = (true_pos + true_neg) / total
        else:
            self.accuracy = 0

    def __str__(self):
        """
        Provides a formatted string representation of validation metrics.

        Returns:
            str: Formatted string of metrics including F1 score, recall, precision,
                 false acceptance, and false rejection.
        """
        return rf"""Metrics:
    F1: {self.f1:.2f},
    Accuracy: {self.accuracy:.2f},
    Recall: {self.recall:.2f},
    Precision: {self.precision:.2f},
    False acceptance: {self.false_acceptance:.2f},
    False rejection: {self.false_rejection:.2f}"""


def validate(model, val_loader: torch.utils.data.DataLoader):
    """
    Validates the model on a validation dataset and computes performance metrics.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        ValidationMetrics: Validation metrics including F1 score, recall, precision,
                           false acceptance, and false rejection.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(model.device)
            output = model(data)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.numpy())
    return ValidationMetrics(multilabel_confusion_matrix(targets, preds))


def predict(model, data_loader: torch.utils.data.DataLoader):
    model.eval()
    preds, targets = model_validate(model, data_loader)
    return ValidationMetrics(multilabel_confusion_matrix(targets, preds))


def model_validate(model, data_loader: torch.utils.data.DataLoader):
    """
    processes validation of given model on a validation dataset and returns predictions and targets.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        predictions (np.ndarray): Predictions of the model on the validation dataset.
        targets (np.ndarray): Targets of the model on the validation dataset.
    """
    preds, targets = [], []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to("cpu")
            output = model(data)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.numpy())
            torch.cuda.empty_cache()

    return preds, targets

def process_predictions_and_features(model, data_loader, device="cpu"):
    """
    Processes predictions and feature maps of a given model on a dataset.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to perform computations on (default: "cpu").

    Returns:
        predictions (np.ndarray): Predictions of the model on the dataset.
        feature_vectors (np.ndarray): Flattened feature vectors from the model.
    """
    model.to(device)
    model.eval()

    predictions, feature_vectors = [], []

    with torch.no_grad():
        for data, target in data_loader:
            # Move data to the specified device
            data, target = data.to(device), target.to(device)

            # Get model output
            output = model(data)

            # Extract predictions
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

            # Extract and process feature maps
            feature_maps = output.cpu().numpy()  # Assuming the output is the feature map
            feature_vector = feature_maps.reshape(feature_maps.shape[0], -1)  # Flatten
            feature_vectors.append(feature_vector)

            if device == "cuda":
                torch.cuda.empty_cache()

    return predictions, feature_vectors

def monte_carlo_predictions(model, val_loader: torch.utils.data.DataLoader):
    """
    processes validation for monte carlo predictions on a validation dataset and returns predictions.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        predictions (np.ndarray): Predictions of the model on the validation dataset.
    """
    model.train()
    preds, targets = model_validate(model, val_loader)
    return preds

def do_train(name, train_loader, val_loader, config, model, criterion, optimizer, device, wandb_enabled=False):
    if wandb_enabled:
        run = wandb.init(name=name, project="iml", config=vars(config))
 
    model.device = device
    model.to(device)

    saved = False
    patience = 0
    best_f1 = -1

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")

        if wandb_enabled:
            logger = wandb.log
        else:
            logger = lambda data,step: print(f"  Step {step}: {data}")

        train(model, train_loader, criterion, optimizer, epoch, logger, len(train_loader) // 5 - 1)
        metrics = validate(model, val_loader)
        print(metrics)

        if wandb_enabled:
            wandb.log({"validation/recall": metrics.recall, "validation/accuracy": metrics.accuracy, "validation/precision": metrics.precision, "validation/f1": metrics.f1, "epoch": epoch+1})

        if metrics.f1 < best_f1:
            patience = patience + 1
        else:
            patience = 0
            best_f1 = metrics.f1
        if patience >= PATIENCE_THRESHOLD:
            model_path = f"./models/{name}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            saved = True

    if(saved == False):
            model_path = f"./models/{name}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    if wandb_enabled: 
        wandb.save(model_path)
        wandb.finish()
        return run
    
def do_test(name, test_loader, model_class, run, device, wandb_enabled = False):
    if wandb_enabled:
        wandb.init(name=name, project="iml", resume="must", id=run.id)
 
    model = model_class()
    model.device = device
    model.to(device)

    model_path = f"./models/{name}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    metrics = validate(model, test_loader)
    print(metrics)

    if wandb_enabled:
        wandb.log({"test/recall": metrics.recall, "test/accuracy": metrics.accuracy, "test/precision": metrics.precision, "test/f1": metrics.f1})

    if wandb_enabled: 
        wandb.finish()
