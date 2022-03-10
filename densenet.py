import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# Create DenseLayer for dense_block()
class DenseLayer(nn.Module):
    """Following DenseNet-BC Dense Layer using BN-ReLU-Conv"""
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.denseLayer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4*growth_rate, 1, 1),
            nn.BatchNorm2d(4*growth_rate), # 4k channel size each layer
            nn.ReLU(),
            nn.Conv2d(4*growth_rate, growth_rate, 3, 1, 1)
        )

    def forward(self, input):
        output = self.denseLayer(input)
        return torch.cat([input, output], 1) # Add input to output


class TransitionLayer(nn.Module):
    """Following DenseNet-BC Compression using BN-ReLU-Conv"""
    def __init__(self, in_channels):
        super().__init__()
        self.transitionLayer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels//2, 1, 1), # Set theta to 0.5 hence 'in_channels//2'
            nn.AvgPool2d(2, 2)
        )

    def forward(self, input):
        output = self.transitionLayer(input)
        return output


# Implement network architecture
class DesneNet3(nn.Module):
    """Implements a DenseNet of 3 dense blocks of 4 dense layers each
    following original network architecture: https://arxiv.org/abs/1608.06993v5"""
    def __init__(self, num_blocks=3, block_size=4, init_channels=3, growth_rate=32):
        super().__init__()
        self.architecture = nn.Sequential()

        # Initial 7x7 Convultion
        self.architecture.add_module(f"conv0", nn.Sequential(
            nn.Conv2d(init_channels, 2*growth_rate, 7, 1, 3),
            nn.BatchNorm2d(2*growth_rate),
            nn.ReLU()
        ))

        # Initial 2x2 Max Pooling (replaces 3x3 Max Pooling)
        self.architecture.add_module(f"pool0", nn.MaxPool2d(2, 2))

        # Member function to create a dense block of block_size dense layers
        def dense_block(in_channels) -> nn.Sequential:
            denseBlock = nn.Sequential()
            for i in range(block_size):
                denseLayer = DenseLayer(in_channels + i*growth_rate, growth_rate)
                denseBlock.add_module(f"denseLayer{i}", denseLayer)
            return denseBlock

        # Create num_blocks of dense blocks and transition layers
        channels = 2*growth_rate
        for i in range(num_blocks):
            self.architecture.add_module(f"denseBlock{i}", dense_block(channels))
            channels += (block_size)*growth_rate
            if i < (num_blocks-1): # Don't add transition layer after final block
                self.architecture.add_module(f"transitionLayer{i}", TransitionLayer(channels))
                channels //= 2

        # Final classification layer
        self.architecture.add_module(f"classifier", nn.Sequential(
            nn.BatchNorm2d(channels), # Final normalisation
            nn.ReLU(), # Final activation
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(channels, 10),
            nn.Softmax(dim=1)
        ))

    def forward(self, x):
        x = self.architecture(x)
        return x


# Cutout Algorithm
def cutout(img: torch.tensor, s: int) -> torch.tensor:
    """
    Takes an image and applys the cutout algoritihm: https://arxiv.org/abs/1708.04552
	Parameters:
		data (torch.Tensor): A 2xN matrix of x and y values
		s (int): Max size of one side of the square mask applied to image
	Returns:
		img (torch.Tensor): Final image with a square mask applied
    """
    width, height = img[0,:,:].shape
    
    # Uniformly sampled mask size between 0 to s
    maskSize = random.randint(0, s)

    # Randomly selected location for center of mask
    y, x = (random.randint(0, height), random.randint(0, width))

    # Top/bottom & left/right bounds of square mask
    t, b = max(y - maskSize//2, 0), min(y + maskSize//2, height) # Min/Max clips edges
    l, r = max(x - maskSize//2, 0), min(x + maskSize//2, width) # Min/Max clips edges

    mask = torch.zeros((3, b-t, r-l))
    img[:, t:b, l:r] = mask

    return img

if __name__ == "__main__":
    TRAIN_MODEL = False # To skip training (~45mins) set this variable to False

    print("\n#### DenseNet ####\n")
    print(f"Running DenseNet {'by training new model' if TRAIN_MODEL else 'from saved model'}...")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load in the train data for train set
    train_batch_size = 20
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: cutout(x, 20)) # Apply the cutout algo to train data
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)

    # Unormalise the test images
    train_images, _ = iter(train_loader).next()
    train_images = train_images/2 + 0.5

    # Montage of 16 images with cutout applied to cutout.png
    print('Creating montage of 16 images with cutout...') 
    torchvision.utils.save_image(torchvision.utils.make_grid(train_images[:16], nrow=4), fp='cutout_new.png')

    # Load in test data from test set
    test_batch_size = 36
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

    # Create and summarise DenseNet architecture
    denseNet = DesneNet3()
    print("\n=== DenseNet3 Architecture ===")
    print(str(denseNet.architecture))

    # Define loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(denseNet.parameters(), lr=0.001, momentum=0.9)

    # Begin training DenseNet3 model
    if TRAIN_MODEL:
        print('\nBegining model training...\n')
        start = time.time()
        for epoch in range(1, 11): # 2c.II) 10 epochs
            print(f"=== Epoch {epoch} ===")
            running_loss = 0.0
            for batch, (inputs, labels) in enumerate(train_loader, 0):
                optimizer.zero_grad()

                outputs = denseNet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                running_loss += loss.item()
                if batch % 1000 == 999: # Print every 1000 mini-batches
                    print('[%d, %5d] - Loss: %.3f      ' %
                        (epoch, batch + 1, running_loss / 1000))
                    running_loss = 0.0
                currentTime = time.time() - start
                sys.stdout.write("[{:04d}] @ {:.2f}s - Loss: {:<10.3f}\r".format(batch, currentTime, loss))
                sys.stdout.flush()
            
            # Compute accuracy vs. test set after each epoch
            print("Running model on test set...    ")
            true_postives = 0
            for test_images, test_labels in iter(test_loader):
                outputs = denseNet(test_images)
                _, predicted = torch.max(outputs, 1)
                true_postives += sum(test_labels == predicted)
            accuracy = (true_postives/len(test_set)).item()
            print(f"Epoch {epoch} test set accuracy: {round(accuracy, 4)}")

    # Once trained save the model
    # torch.save(denseNet.state_dict(), 'densenet3_model_new.pt') # Un/comment this to overwrite saved model
    denseNet.load_state_dict(torch.load('densenet3_model.pt')) # Un/comment to load in a saved model


    # Montage of 36 test images, captions indicating gt vs. predicted   
    print('\nCreating montage of 36 images...') 
    test_images, test_labels = iter(test_loader).next()
    outputs = denseNet(test_images)
    _, predicted = torch.max(outputs, 1)

    # Map indices to class names
    gt_classes = list(map(lambda x: classes[x], test_labels))
    pred_classes = list(map(lambda x: classes[x], predicted))

    # Create white 32x32 image to place label into
    graphic_labels = torch.zeros((36, 3, 32, 64), dtype=torch.float32)
    for i, (gt, pred) in enumerate(zip(gt_classes, pred_classes)):
        imgLabel = Image.fromarray(255*torch.ones((32, 64, 3)).numpy().astype('uint8'), mode='RGB')
        draw = ImageDraw.Draw(imgLabel)
        draw.text((5, 4), f"GT:{gt}", (0,0,0))
        draw.text((5, 16), f"Pre:{pred}", (0,0,0))
        graphic_labels[i] = transforms.ToTensor()(np.array(imgLabel))

    # Unormalise the test images
    test_images = test_images/2 + 0.5
    
    # Combine graphical labels with test_images
    cnt = 0
    classifiedImages = torch.zeros((72, 3, 32, 64))
    for img, label in zip(test_images, graphic_labels):
        padding = torch.ones((3, 32, 16), dtype=torch.float32)
        newImg = torch.cat([padding, img, padding], dim=2)
        classifiedImages[cnt] = newImg
        classifiedImages[cnt+1] = label
        cnt += 2

    # Create grid of images and labels and save as results.png
    torchvision.utils.save_image(torchvision.utils.make_grid(classifiedImages, nrow=2), fp='results_new.png')

    # Print labels to console
    print('\nGround-truth: ', str(gt_classes))
    print('Predicted:    ', str(pred_classes))
