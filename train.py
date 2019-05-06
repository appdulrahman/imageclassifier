# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from workspace_utils import active_session
import matplotlib.pyplot as plt
from PIL import Image
import argparse
 
# Argparse arguments
parser = argparse.ArgumentParser(description='Train a model to predict a type of a flower')
parser.add_argument('-d','--data', type=str, help='The data file path')
parser.add_argument('-s','--save_dir', type=str, help='Set directory to save checkpoints')
parser.add_argument('-a','--arch', type=str, help='Choose architecture to be vgg11 or alexnet')
parser.add_argument('-l','--learning_rate', type=float, help='Set the hyperparameter learning rate')
parser.add_argument('-H','--hidden_units', type=int, help='Set the hyperparameter hidden units')
parser.add_argument('-e','--epochs', type=int, help='Set the hyperparameter epochs')
parser.add_argument('-g','--gpu', type=str, help='Train the model on GPU or CPU')
args = parser.parse_args()
    
# files
data_dir = args.data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Defining transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(224,scale=(0.5, 1)),
    transforms.RandomRotation(45),
    transforms.RandomAffine(30, scale=(0.3,0.8)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Loadong the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder(train_dir,data_transforms['train']),
                  'valid': datasets.ImageFolder(valid_dir,data_transforms['valid']),
                  'test': datasets.ImageFolder(train_dir,data_transforms['test'])}

# Using the image datasets and the trainforms to define the dataloaders
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 10, shuffle=True),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 10),
               'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 10)}

# Use GPU if it's available
device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu.lower() == 'gpu' ) else 'cpu')

# Build and train your network
def train_model(model, criterion, optimizer, num_epochs=5):

    for epoch in range(num_epochs):
        print('-' * 20)
        print('>' * 4 +'Epoch {}/{}'.format(epoch+1, num_epochs) + '<' * 4)
        print('-' * 20)
        
        for phase in ['train','valid']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
                
            running_loss = 0.0
            valid_loss = 0.0
            train_accuracy = 0.0
            valid_accuracy = 0.0
            
            for inputs, labels in dataloaders[phase]:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    model_tr = model.forward(inputs)
                    loss = criterion(model_tr,labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                    # Calculate accuracy
                    tr = torch.exp(model_tr)
                    top_p, top_class = tr.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                else:
                    with torch.no_grad():
                        val_model = model.forward(inputs)
                        va_loss = criterion(val_model, labels)
                        valid_loss += va_loss.item()
                        
                        # Calculate accuracy
                        va = torch.exp(val_model)
                        top_p, top_class = va.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            if phase == 'train':
                print('train loss: {}..'.format(running_loss/len(dataloaders['train'])),
                      'train accuracy: {}..'.format(train_accuracy/len(dataloaders['train'])))
            
            else:
                print('validation loss: {}..'.format(valid_loss/len(dataloaders['valid'])),
                      'validation accuracy: {}..'.format(valid_accuracy/len(dataloaders['valid'])))

    return model             

def main():

    #import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    if args.arch.lower() == 'vgg11':
        model = models.vgg11(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(args.hidden_units, args.hidden_units//3),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(args.hidden_units//3, args.hidden_units//9),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(args.hidden_units//9, 102),
                                         nn.LogSoftmax(dim=1))

        
    elif args.arch.lower() == 'alexnet':
        model = models.alexnet(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
    
        model.classifier = nn.Sequential(nn.Linear(9216, args.hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(args.hidden_units, args.hidden_units//3),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(args.hidden_units//3, args.hidden_units//9),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(args.hidden_units//9, 102),
                                         nn.LogSoftmax(dim=1))
    else:
        print('Please choose your model architecture to be vgg11 or alexnet')

    # Only train the classifier parameters, feature parameters are frozen
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    model.to(device);

    with active_session():
        # do long-running work here
        model = train_model(model, criterion, optimizer, num_epochs=args.epochs)
        #  Save the checkpoint
        model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'epochs': args.epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, args.save_dir)

if __name__ == '__main__':
    main()
