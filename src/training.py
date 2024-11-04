import torch
from sklearn.metrics import multilabel_confusion_matrix


def train(model, train_loader, criterion, optimizer, epoch, logger, step):
    """
    Trains the model for one epoch.

    Parameters:
        model (torch.nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training set.
        criterion: criterion for loss
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epoch (int): Epoch number.
        logger (function): logger function for loss and accuracy statistics
        step (int): every this number of samples loss and accuracy is being calculated
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
            step = epoch * len(train_loader) + i
            logger({"train/accuracy": accuracy, "train/loss": loss}, step=step)
            running_loss = 0.0
            correct = 0


class ValidationMetrics:
    def __init__(self, confusion_matrix):
        true_neg = confusion_matrix[0, 0, 0]
        false_neg = confusion_matrix[0, 1, 0]
        true_pos = confusion_matrix[0, 1, 1]
        false_pos = confusion_matrix[0, 0, 1]

        real_pos = true_pos + false_neg
        real_neg = true_neg + false_pos
        model_pos = true_pos + false_pos

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
            # https://en.wikipedia.org/wiki/F-score#Definition
            self.f1 = (2 * true_pos) / (model_pos + real_pos)
        else:
            self.f1 = 0
        # TODO: Maybe not 0?

    def __str__(self):
        return rf"""Metrics:
    F1: {self.f1:.2f},
    Recall: {self.recall:.2f},
    Precision: {self.precision:.2f},
    False acceptance: {self.false_acceptance:.2f},
    False rejection: {self.false_rejection:.2f}"""


def validate(model, val_loader: torch.utils.data.DataLoader):
    """
    Validates the model and calculates F1 score.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        float: Recall score.
        float: Precision score.
        float: Validation F1 score.
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
