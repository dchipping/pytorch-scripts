import time
from enum import Enum
from typing import Generator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Net(nn.Module):
    """CNN taken from img_cls Lab"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Modification(Enum):
    """Enum used as flag to set the modification"""
    SGD = 0,
    Adam = 1,


def get_accuracy(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Calculates the accuracy of a predicition from the ground truth
	Paremeters:
		predicted (torch.Tensor): Model predicited values given some input (y_hat)
		ground_truth (torch.Tensor): True values for some input
	Returns:
		accuracy (float): 0 to 1 value of model accuracy given ground truth 
    """
    assert predicted.shape == ground_truth.shape
    return sum(predicted == ground_truth)/len(ground_truth)


def img_cls_train(trainloader: DataLoader, modification: Modification, model_name=None) -> Tuple[nn.Module, List[float]]:
    """
    General function that can be called to train a PyTorch model
	Parameters:
		trainset (DataLoader): Training data to fit model to
		modification (Modification): Either SGD or Adam
		model_name (str): Name of saved model, if None model is not saved
	Returns:
		model (nn.Module): Returns the trained model
    """
    model = Net()

    # Loss and Optimiser
    criterion, optimizer = torch.nn.CrossEntropyLoss(), None
    if modification is Modification.SGD: # SGD + Momentum
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif modification is Modification.Adam: # Adam
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    start = time.time()
    for epoch in range(2): # Loop over the dataset multiple times
        runningAcc = 0.
        runningLoss = 0.
        for i, data in enumerate(trainloader, 0):
            # Get the inputs, data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = get_accuracy(predicted, labels)

            # Loss
            timeElapsed = time.time() - start
            runningLoss += loss.item()
            runningAcc += accuracy
            if i % 1000 == 999: # Print every 1000 mini-batches
                print('[%d, %5d] @ %.2fs - Loss: %.3f, Accuracy: %.3f' %
                    (epoch + 1, i + 1, timeElapsed, runningLoss / 1000, runningAcc / 1000))
                runningLoss = 0.
                runningAcc = 0.          
    
    end = time.time()
    
    # Compute final metrics
    totalTime = end-start
    inputs, labels = iter(trainloader).next()
    outputs = model(inputs)
    finaLoss = criterion(outputs, labels)
    
    _, predicted = torch.max(outputs, 1)
    finalAcc = get_accuracy(predicted, labels)
    metrics = [totalTime, finaLoss, finalAcc]

    # If specified, save model to disk
    if model_name:
        torch.save(model.state_dict(), f"{model_name}_img_cls_model.pt")

    return model, metrics


def img_cls_test(testloader: DataLoader, model: nn.Module) -> List[float]:
    """
    Run a model against a test set to compute the total test time, averaged
    loss & overall accuracy of the test run.
	Parameters:
		testloader (DataLoader): Test set loader
		model (nn.Module): Pre-trained neural network model
	Returns:
		metrics (List[floats]): List of total test time, averaged loss and accuracy
    """
    count = 0
    criterion = torch.nn.CrossEntropyLoss()
    runningLoss, totalTruePositives = 0., 0

    start = time.time()
    for i, (images, labels) in enumerate(testloader, 0):
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        truePositives = sum(predicted == labels)

        runningLoss += loss
        totalTruePositives += truePositives
        count += len(images)

    end = time.time()

    totalTime = end-start
    averagedLoss = runningLoss/len(testloader)
    overallAccuracy = totalTruePositives/count

    return [totalTime, averagedLoss, overallAccuracy]


def k_folds(dataset: Dataset, k: int) -> Generator:
    """
    Implements algorithim to generate k folds of indices based 
    on dataset's total length and a specified number of folds.
	Parameters:
		dataset (DataLoader): Dataset that will be analysed using k-folds
		k (int): Number of desired folds
	Return:
		generator (Generator): Tuple generator containing train and validation indices
    """
    l = len(dataset)
    foldLen = l // k
    remainder = l % k # If division is not exact how many indices remain

    idxs = [] # e.g. [[0,1], [2, 3], [4, 5]]
    idxSum = 0
    for n in range(k): # Produce k number of folds
        start, end = idxSum, idxSum + foldLen
        # Add remainder evenly across newest folds
        if remainder > 0:
            end += 1; remainder -=1
        # Produce list of indices from start to end of fold
        idxs.append(list(range(start, end))) 
        idxSum += end-start # Move to next fold
        print(f"Fold {n+1}: {start}-{end}")

    # Iterate over validation folds and merge rest
    for i in range(k):
        trainIdxs = sum(idxs[:i] + idxs[i+1:], []) # e.g. [0, 1, 2, 3]
        valIdxs = idxs[i] # e.g. [4, 5]
        yield trainIdxs, valIdxs


if __name__ == "__main__":
    print("\n#### SGD vs Adam ####\n")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    global_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Split data into development set
    dev_batch_size = 20
    dev_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=global_transforms)
    dev_dataloader = torch.utils.data.DataLoader(dev_set, batch_size=dev_batch_size, shuffle=False)

    #  Split data into holdout set
    holdout_batch_size = 4
    holdout_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=global_transforms)
    holdout_loader = torch.utils.data.DataLoader(holdout_set, batch_size=holdout_batch_size, shuffle=False)

    # Global store of all metrics during train/val/test
    allMetrics = {}

    # Execute operations twice for each modificiation
    for modification in [Modification.SGD, Modification.Adam]:
        print(f"\n=== Running model with {modification.name} ===")
        
        # Run the cross validation scheme for modification
        folds = 3
        kfolds_dev_set = k_folds(dev_set, folds)
        for k, (train_idxs, val_idxs) in enumerate(kfolds_dev_set, 0):
            print(f"\n=== {modification.name} CV - Fold {k+1} ===")

            # Summary of random split
            print(f"Train set of size {len(train_idxs)}")
            print(f"Validation set of size {len(val_idxs)} (1 fold from index {val_idxs[0]}-{val_idxs[-1]})")

            # Create random samplers based on train/val split using SubsetRandomSampler
            train_set_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
            val_set_sampler = torch.utils.data.SubsetRandomSampler(val_idxs)

            # Create train/val sets using shuffled indices from development set
            train_set = torch.utils.data.DataLoader(dev_set, batch_size=dev_batch_size, sampler=train_set_sampler)
            val_set = torch.utils.data.DataLoader(dev_set, batch_size=dev_batch_size, sampler=val_set_sampler)

            # Train model for using train data
            print("Training model on test set...")
            model, metrics = img_cls_train(train_set, modification)
            print("Model succesfully trained on test split")

            # Loss, Speed & Accuracy on train set during fold
            print(f"\n=== Train Metrics {modification.name} CV Fold {k+1} ===")
            trainTime, trainLoss, trainAcc = metrics
            print("Total Time: {:.2f}s".format(trainTime))
            print("Final Loss: {:.3f}".format(trainLoss))
            print("Final Accuracy: {:.3f}".format(trainAcc))

            # Loss, Speed & Accuracy on validation set during fold
            print(f"=== Validation Metrics {modification.name} CV Fold {k+1} ===")
            print("Running model on validation set...")
            valTime, valLoss, valAcc = img_cls_test(val_set, model)
            print("Total Time: {:.2f}s".format(valTime))
            print("Final Loss: {:.3f}".format(valLoss))
            print("Final Accuracy: {:.3f}".format(valAcc))

            # Save result for overall summary
            allMetrics[k] = metrics

        # Overall summary of 3-Fold CV Scheme
        print(f"\n=== Summary of {modification.name} Cross Validation ===")
        totalTime, totalAcc, totalLoss = 0., 0., 0.
        print("{:<8} {:<7} {:<7} {:<7}".format('k-Fold','Time','Loss','Accuracy'))
        for k in range(folds):
            elapsedTime, loss, accuracy = allMetrics[k]
            print("{:<8} {:<7.2f} {:<7.3f} {:<7.3f}".format(k+1, totalTime, loss, accuracy))
            totalTime += elapsedTime
            totalLoss += loss
            totalAcc += accuracy
        print("----------------------------------------")
        avgLoss = totalLoss/3
        avgAccuracy = totalAcc/3
        print("Overall Time: {:.2f}s".format(totalTime))
        print("Overall Averaged Loss: {:.3f}".format(avgLoss))
        print("Overall Averaged Accuracy: {:.3f}".format(avgAccuracy))
        
        # Train two further models using the entire development set
        print(f"\n=== Training {modification.name} w/o Cross Validation ===")
        model_name = f"{modification.name}_new"
        print("Training model on test set...")
        model, _ = img_cls_train(dev_dataloader, modification, model_name)

        # Model's metrics on holdout set vs. prior cross validation
        print(f"\n=== {modification.name} - Holdout vs. Cross Validation ===")
        print("Running model on holdout set...")
        holdoutTime, holdoutLoss, holdoutAcc = img_cls_test(holdout_loader, model)
        print("Holdout Set - Time: {:.2f}, Loss: {:.2f}, Accuracy {:.2f}".format(holdoutTime, holdoutLoss, holdoutAcc))
        print("Full Cross Validation - Time: {:.2f}, Loss: {:.2f}, Accuracy {:.2f}".format(totalTime, avgLoss, avgAccuracy))
        print("Holdout vs. CV - +/-Loss: {:.2f}, +/-Accuracy {:.2f}".format(holdoutLoss-avgLoss, holdoutAcc-avgAccuracy))
