import torch
from sklearn.metrics import f1_score

def train(model, train_loader, criterion, optimizer):
    """
    Trains the model for one epoch.

    Parameters:
        model (torch.nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion: criterion for loss

    Returns:
        float: Training loss.
    """
    model.train()
    for _, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def validate(model, val_loader):
    """
    Validates the model and calculates F1 score.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        float: Validation F1 score.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    f1 = f1_score(targets, preds, average='macro')
    return f1
