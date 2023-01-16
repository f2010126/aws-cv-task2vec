import numpy as np
import torch
from sklearn.metrics import f1_score
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())


# function to train the model
def train(model, train_dataloader, device, cross_entropy, optimizer):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 10 batches.
        if step % 10 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(input_ids=sent_id, mask=mask, enable_fim=False)
        # pred_labels = torch.argmax(logits, dim=1)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        # append the model predictions
        total_preds += list(preds)
        total_labels += labels.tolist()

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    # total_preds  = np.concatenate(total_preds, axis=0)
    f1 = f1_score(total_labels, total_preds, average='weighted')
    # returns the loss and predictions
    return avg_loss, f1


# function for evaluating the model
def evaluate(model, val_dataloader, device, cross_entropy):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 10 batches.
        if step % 10 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            total_preds += list(preds)
            total_labels += labels.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    # total_preds  = np.concatenate(total_preds, axis=0)

    f1 = f1_score(total_labels, total_preds, average='weighted')
    return avg_loss, f1


def save_checkpoint(filename, epoch, model, optimizer, label_map, id2label):
    state = {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'label_map': label_map,
        'id_map': id2label}
    torch.save(state, filename)


def training_loop(model, optimizer, label_map, id2label, epochs, train_dataloader, cross_entropy, val_dataloader,
                  device):
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, f1_train = train(model, train_dataloader, device, cross_entropy, optimizer)

        # evaluate model
        valid_loss, f1_valid = evaluate(model, val_dataloader, device, cross_entropy)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            file_name = 'topic_saved_weights.pt'
            save_checkpoint(file_name, epoch, model, optimizer, label_map, id2label)

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'\nTraining F1: {f1_train:.3f}')
        print(f'Validation F1: {f1_valid:.3f}')
