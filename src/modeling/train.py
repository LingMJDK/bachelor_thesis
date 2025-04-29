def train_step(model, train_data, loss_fn, accuracy_fn, optimizer, device):
    """ """

    model.train()

    train_loss, train_acc = 0, 0

    for X, y in train_data:
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass
        y_logits = model(X)  # Output is raw logits
        y_preds = y_logits.argmax(dim=1)  # Turns logits into predictions

        # 2.1 Calculate Loss and Accuracy
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y_pred=y_preds, y_true=y)

        # 2.2 Accumulate Loss and Accuracy
        train_loss += loss.item()
        train_acc += acc

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Step
        optimizer.step()

    train_loss /= len(train_data)
    train_acc /= len(train_data)


#   print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train accuracy {train_acc:.2f}%")
