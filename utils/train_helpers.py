import torch
from torch import nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model, dataloader, optimizer, criterion):
    train_loss, train_acc = 0, 0
    for batch, (images, labels) in enumerate(dataloader):
        model.train()
        images, labels = images.to(device), labels.to(device)
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = torch.argmax(y_pred, dim=1)
        correct = (preds == labels).sum()
        batch_acc = correct.float() / labels.size(0)
        train_acc += batch_acc.item()
    train_acc /= len(dataloader)
    train_loss /= len(dataloader)
    return train_acc, train_loss

def test_step(model, dataloader, criterion):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            test_loss += loss.item()
            preds = torch.argmax(y_pred, dim=1)
            correct = (preds == labels).sum()
            batch_acc = correct.float() / labels.size(0)
            test_acc += batch_acc.item()
    test_acc /= len(dataloader)
    test_loss /= len(dataloader)
    return test_acc, test_loss

def train(model,train_dataloader,test_dataloader,optimizer,criterion,epochs = 10,writer = None):
    results ={'train_acc':[],'train_loss':[],'test_acc':[],'test_loss':[]}

    for epoch in tqdm(range(epochs)):
        train_acc,train_loss = train_step(model,train_dataloader,optimizer,criterion)
        test_acc,test_loss = test_step(model,test_dataloader,criterion)

        print(f"Epoch : {epoch} | train_acc : {train_acc: .4f} | test_acc :{test_acc : .4f}")

        results['train_acc'].append(train_acc)
        results['train_loss'].append(train_loss)
        results['test_acc'].append(test_acc)
        results['test_loss'].append(test_loss)
        if writer:
            writer.add_scalar("Loss/train",train_loss,epoch)
            writer.add_scalar("Accuracy/train",train_acc,epoch)
            writer.add_scalar("Loss/test",test_loss,epoch)
            writer.add_scalar("Accuracy/test",test_acc,epoch)

    return results