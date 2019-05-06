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
parser = argparse.ArgumentParser(description='Test a model to predict a type of a flower in an imags')
parser.add_argument('-l','--load_dir', type=str, help='Set directory to load checkpoints')
parser.add_argument('-c','--category_names', type=str, help='For loading a JSON file that maps the class values to other category names')
parser.add_argument('-i','--input', type=str, help='The test image path')
parser.add_argument('-g','--gpu', type=str, help='Test the model on GPU or CPU')
parser.add_argument('-t','--top_k', type=int, help='The top K predicted classes') 
args = parser.parse_args()

#import json
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Use GPU if it's available
device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu.lower() == 'gpu' ) else 'cpu')

# files
data_dir = 'flowers'
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

#Defined a model 
model = models.alexnet(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(9216, 2700),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(2700, 900),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(900, 300),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(300, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.CrossEntropyLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

model.to(device);


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model    
    img_loader = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    np_image = np.array(pil_image)
    np_image = np.transpose(np_image, (0, 2, 1))

    return torch.from_numpy(np_image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #Implement the code to predict the class from an image file
    input = process_image(image_path)
    input.unsqueeze_(0)
    prediction = model(input)
    ps = torch.exp(prediction)
    prebs, classes = ps.topk(topk, dim=1)
    prebs = np.asarray(prebs.cpu().detach().numpy()).ravel()
    classes = np.asarray(classes.cpu().detach().numpy()).ravel()
    print(prebs)
    print(classes)
    
    return prebs, classes

# A function that loads a checkpoint and rebuilds the model
def load_model(file_path):
    checkpoint = torch.load(file_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    model.eval()
    return model

def main():

    #Save an image to use for testing
    dataiter = iter(dataloaders['test'])
    images, labels = dataiter.next()
    img = images[5]
    torchvision.utils.save_image(img, 'img.jpg')
    
    # Load the model
    model = load_model(args.load_dir)
    
    # Display an image along with the top 5 classes
    probs, classes = predict(args.input, model, args.top_k)

    # get the labels
    top_class = []
    for c in classes:
        c = str(c)
        top_class.append(cat_to_name[c])
    
    print(top_class)
    plt.barh(top_class, classes), imshow(process_image(args.input))

if __name__ == '__main__':
    main()