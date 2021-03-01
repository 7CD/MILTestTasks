"""Script for evaluating ResNet-20 on CIFAR-10."""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from .resnet.resnet20 import resnet20


def parse_arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-path", help="Path for saving/loading of the model.")
    parser.add_argument("--data-dir", "-d", help="Path to dataset dir.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def calculate_accuracy(model, loader, device):
    model.eval()
    n_samples, accuracy = 0, 0
    
    for batch in tqdm(loader, total=len(loader)):
        input = batch[0].to(device)
        target = batch[1].to(device)

        with torch.no_grad():
            output = model(input)
                
        batch_size = target.size(0)
        n_samples += batch_size
        accuracy += (torch.max(output, 1)[1] == target).sum().item()
    
    return accuracy / n_samples


def main(args):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.243, 0.262])

    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    normalize]), download=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                             num_workers=2, pin_memory=True,
                                             shuffle=False, drop_last=False)
    
    model = resnet20()
    device = torch.device('cuda:0') if args.gpu else torch.device('cpu')
    model.to(device)

    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("no model found at '{}'".format(args.model_path))
        return

    accuracy = calculate_accuracy(model, test_loader, device)

    print('accuracy = {:.4}'.format(accuracy))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
