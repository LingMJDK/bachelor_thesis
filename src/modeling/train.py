import torch  # Required for tensor operations and model training
from typing import Callable  # For type hints like Callable




try:
    from src.modeling.models import *
except ImportError:
    from modeling.models import *


def accuracy_fn(y_pred, y_true):
    """Calculates accuracy between truth labels and predictions.

    Params:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def accuracy_fnV2(y_pred, y_true):
    """Returns accuracy as percentage between predictions and labels."""
    correct = (y_pred == y_true).sum().item()
    total = y_true.numel()
    return (correct / total) * 100


def train_step(model, train_data, loss_fn, accuracy_fn, optimizer, device, epoch, task="unknown"):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for X, y in train_data:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        y_logits = model(X)
        y_preds = y_logits.argmax(dim=-1)
        if task.lower() == "classification":
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y_preds, y)
        else:
            B, T, V = y_logits.shape
            loss = loss_fn(y_logits.view(B*T, V), y.view(B*T))
            acc = accuracy_fn(y_preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc
    n = len(train_data)
    train_loss /= n
    train_acc  /= n
    print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train accuracy {train_acc:.2f}%")
    return train_loss, train_acc


def test_step(model, test_data, loss_fn, accuracy_fn, device, epoch, task="unknown"):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.inference_mode():
        for X, y in test_data:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_logits = model(X)
            y_preds = y_logits.argmax(dim=-1)
            if task.lower() == "classification":
                loss = loss_fn(y_logits, y)
                acc = accuracy_fn(y_preds, y)
            else:
                B, T, V = y_logits.shape
                loss = loss_fn(y_logits.view(B*T, V), y.view(B*T))
                acc = accuracy_fn(y_preds, y)
            test_loss += loss.item()
            test_acc  += acc
    n = len(test_data)
    test_loss /= n
    test_acc  /= n
    print(f"Epoch: {epoch} | Test loss: {test_loss:.4f} | Test accuracy {test_acc:.2f}%")
    return test_loss, test_acc


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
