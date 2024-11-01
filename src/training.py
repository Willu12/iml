import torch
from sklearn.metrics import f1_score, recall_score, precision_score


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


def validate(model, val_loader):
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
            data, target = data.to(model.device), target.to(model.device)
            output = model(data)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    recall = recall_score(targets, preds, zero_division=0)
    precision = precision_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, average="macro")
    return recall, precision, f1
