import torch


def test_step(model, test_data, loss_fn, accuracy_fn, device):
    """
    Evaluates the model on the test dataset.
    Params:
        model: The trained model to evaluate.
        test_data: DataLoader for the test dataset.
        loss_fn (callable): Loss function to calculate the test loss.
        accuracy_fn (callable): Function to calculate accuracy.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Average test loss and accuracy.
    """

    model.eval()

    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for X, y in test_data:
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass
            y_logits = model(X)  # Output is raw logits
            y_preds = y_logits.argmax(dim=1)  # Turns logits into predictions

            # 2.1 Calculate Loss and Accuracy
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y_pred=y_preds, y_true=y)
            # 2.2 Accumulate Loss and Accuracy
            test_loss += loss.item()
            test_acc += acc

        test_loss /= len(test_data)
        test_acc /= len(test_data)
        # print(
        #     f"Epoch: {epoch} | Test loss: {test_loss:.4f} | Test accuracy {test_acc:.2f}%"
        # )
