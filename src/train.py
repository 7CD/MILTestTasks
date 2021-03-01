"""Script for training ResNet-20 on CIFAR-10."""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from resnet.resnet20 import resnet20


def parse_arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-path", help="Path for saving/loading of the model.")
    parser.add_argument("--data-dir", "-d", help="Path to dataset dir.", default=None)
    parser.add_argument("--batch-size", "-b", default=128, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=0.1, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    #parser.add_argument("--shed-factor", default=0.1, type=float)
    #parser.add_argument("--shed-patience", default=5, type=int)
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    n_samples, train_loss = 0, 0

    #for batch in tqdm(loader, total=len(loader), desc='training...'):
    for batch in loader:
        input = batch[0].to(device)
        target = batch[1].to(device)

        output = model(input)
        loss = loss_fn(output, target, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        n_samples += batch_size
        train_loss += loss.item() * batch_size
    
    return train_loss / n_samples


def validate(model, loader, loss_fn, device):
    model.eval()
    n_samples, val_loss, accuracy = 0, 0, 0
    
    #for batch in tqdm(loader, total=len(loader), desc='validation...'):
    for batch in loader:
        input = batch[0].to(device)
        target = batch[1].to(device)

        with torch.no_grad():
            output = model(input)
        
        loss = loss_fn(output, target, reduction='mean')
        
        batch_size = target.size(0)
        n_samples += batch_size
        val_loss += loss.item() * batch_size
        accuracy += (torch.max(output, 1)[1] == target).sum().item()
    
    return val_loss / n_samples, accuracy / n_samples


def main(args):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, 
                                                 transform=transforms.Compose([
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.RandomCrop(32, 4),
                                                     transforms.ToTensor(),
                                                     normalize]), download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               num_workers=2, pin_memory=True,
                                               shuffle=True, drop_last=True)

    val_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    normalize]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, 
                                             num_workers=2, pin_memory=True,
                                             shuffle=True, drop_last=True)

    model = resnet20()
    device = torch.device('cuda:0') if args.gpu else torch.device('cpu')
    model.to(device)
    
    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.shed_factor, patience=args.shed_patience)

    start_epoch, best_val_acc = 1, 0
    learning_rate = args.learning_rate
    if args.resume:
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epochs'] + 1
            best_val_acc = checkpoint['accuracy']
            print("loaded checkpoint '{}' (epochs: {}, accuracy: {}, lr: {})"
                  .format(args.model_path, start_epoch - 1, best_val_acc, optimizer.param_groups[0]['lr']))
        else:
            print("no checkpoint found at '{}'".format(args.model_path))
            return
    elif os.path.isfile(args.model_path):
        print("Checkpoint '{}' already exists. Name model differently or set the '--resume' flag.".format(args.model_path))
        return

    model.to(device)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        #scheduler.step(train_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open(f"{args.model_path}", "wb") as fp:
                torch.save({
                            'model_state_dict': model.state_dict(),
                            #'optimizer_state_dict': optimizer.state_dict(),
                            'epochs': epoch,
                            'accuracy': best_val_acc
                            }, fp)
        print("Epoch #{}. lr: {}, train loss: {}, val loss: {}, val accuracy: {} (best: {})".\
              format(epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss, val_acc, best_val_acc))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
