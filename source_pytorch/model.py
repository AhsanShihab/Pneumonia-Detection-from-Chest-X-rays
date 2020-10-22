import argparse
import json
import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def densnet_pretrained():
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=1),nn.Sigmoid())
    
    return model

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = densnet_pretrained()

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model

# Get training data in batches
def train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(training_dir, transform = train_transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return trainloader

# Get training data in batches
def validation_data_loader(batch_size, validation_dir):
    print("Get validation data loader.")

    val_transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    
    val_data = datasets.ImageFolder(validation_dir, transform = val_transform)
    validationloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return validationloader

def weighted_BCELoss(pos_weight, neg_weight):
    def weighted_loss_func(outputs, labels, pos_weight=pos_weight, neg_weight=neg_weight, epsilon=1e-7):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weight = torch.Tensor([pos_weight]).to(device)
        neg_weight = torch.Tensor([neg_weight]).to(device)

        loss = torch.mean(-pos_weight*labels*torch.log(outputs+epsilon) - neg_weight*(1-labels)*torch.log(1-outputs+epsilon))

        return loss

    return weighted_loss_func

# training function
def train(model, trainloader, criterion, optimizer, device, testing=False):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    testing      - If True, only the first batch will be run
    """
    
    model.train() # Make sure that the model is in training mode.

    training_loss = 0
    for step, batch in enumerate(trainloader):
        images, labels = batch
        labels = labels.float()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        batch_loss = criterion(outputs.squeeze(), labels)
        batch_loss.backward()
        optimizer.step()

        training_loss += batch_loss.item()
        
        if testing:
            break

    return model, training_loss/len(trainloader)

# training function
def validation(model, validationloader, criterion, device, testing=False):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    criterion    - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    testing      - If True, only the first batch will be run
    """
    
    model.eval() # Make sure that the model is in training mode.
    
    validation_loss = 0
    for batch in validationloader:
        images, labels = batch
        labels = labels.float()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            batch_loss = criterion(outputs.squeeze(), labels)
            validation_loss += batch_loss.item()
        if testing:
            break

    return validation_loss/len(validationloader)
        
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val-data-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pos-weight', type=float, metavar='PW',
                        help='positive class weight for loss function')
    parser.add_argument('--neg-weight', type=float, metavar='NW',
                        help='negative class weight for loss function')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    trainloader = train_data_loader(args.batch_size, args.train_data_dir)
    
    # Load the training data.
    validationloader = validation_data_loader(args.batch_size, args.val_data_dir)
    
    # initiate model
    model = densnet_pretrained().to(device)
    
    # define an optimizer and loss function for training
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    # define a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)
    
    # define loss function
    criterion = weighted_BCELoss(args.pos_weight, args.neg_weight)

    # train the model
    for epoch in range(args.epochs):
        print(f"{epoch+1}/{args.epochs}")
        model, training_loss = train(model, trainloader, criterion, optimizer, device)
        validation_loss = validation(model, validationloader, criterion, device)
        print(" - training loss "+ str(training_loss) + " - val. loss " + str(validation_loss))
        print("Learning rate used " + str(optimizer.param_groups[0]['lr']))
        scheduler.step(validation_loss)

    # save the model
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)