import torch


def train_one_epoch(train_loader, device, optimizer, model, loss_fn, summary_writer, epoch):
    """
    train one epoch:
    1. get batch of data move to device
    2. zero gradients
    3. perform inference
    4. Calculate the loss
    5. backward
    6. optimizer step
    7. report running loss
    8. get the avg loss to validation step
    """
    running_loss = 0.0
    last_loss = 0.0
    for i, data in enumerate(train_loader):
        # 1. Get batch of data
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 2. Zero gradient
        optimizer.zero_grad()
        # 3. perform inference
        outputs = model(inputs)
        # 4. calculate loss
        loss = loss_fn(outputs, labels)
        # 5. backward
        loss.backward()
        # 6.
        optimizer.step()
        # 7.
        running_loss += loss.item()
        # 8 report at every 1000 batch, and return avg loss
        if not (i % 1000):
            last_loss = running_loss / 1000
            print(f"Batch {i} Loss {last_loss}")
            x_tb = epoch * len(train_loader) + i + 1
            summary_writer.add_scalar("Train/Loss", last_loss, x_tb)
            running_loss = 0.0

    return last_loss


def per_epoch_activity(train_loader, test_loader, device, optimizer, model, loss_fn,
                       summary_writer, num_epoch, timestamp):
    """
    1. perform validation
    2. save the best model
    """
    best_loss = 1_000_000
    for epoch in range(num_epoch):
        print("Epoch: ", epoch)

        # 1.1 train model
        model.train(True)
        avg_loss = train_one_epoch(train_loader, device, optimizer, model, loss_fn, summary_writer, epoch)
        model.train(False)

        running_val_loss = 0.0
        # 1. 2
        for i, data in enumerate(test_loader):
            val_inputs, val_labels = data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)

            val_loss = loss_fn(val_outputs, val_labels)
            running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)
        print(f"Epoch {epoch} Training Loss {avg_loss} Validation Loss {avg_val_loss}")
        summary_writer.add_scalar("Training/Validation",{
            "Train loss ": avg_loss, "Validation loss ": avg_val_loss,
        }, epoch)
        summary_writer.flush()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_path = fr'\saved_model\state_dict\model_{timestamp}_{epoch}'
            torch.save(model.sate_dict(), model_path)
