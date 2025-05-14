import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy
import matplotlib.pyplot as plt # Import matplotlib

def prepare_data(data_dir='EX2/flower_dataset', batch_size=32):
    """Data preparation and loading"""
    # Data augmentation and preprocessing
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir, data_transforms)

    # Split training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    num_workers = 0 if os.name == 'nt' else 4  # Set to 0 for Windows
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers # Changed shuffle to False for val_loader for consistency
    )

    # Generate class files and annotation files
    class_names = full_dataset.classes
    os.makedirs(os.path.dirname(data_dir), exist_ok=True) # Ensure base directory for annotations exists
    with open(os.path.join(os.path.dirname(data_dir), 'classes.txt'), 'w') as f: # Save in the parent of data_dir
        for class_name in class_names:
            f.write(class_name + '\n')

    def _generate_annotation(dataset, filename, base_annotation_dir):
        os.makedirs(base_annotation_dir, exist_ok=True)
        with open(os.path.join(base_annotation_dir, filename), 'w') as f:
            # When using random_split, dataset.dataset refers to the original full_dataset
            # and dataset.indices refers to the indices selected for this subset.
            for idx in dataset.indices:
                img_path, label = dataset.dataset.samples[idx]
                rel_path = os.path.relpath(img_path, data_dir)
                f.write(f"{rel_path} {label}\n")

    annotation_dir = os.path.join(os.path.dirname(data_dir), 'annotations') # Define a specific directory for annotations
    _generate_annotation(train_dataset, 'train.txt', annotation_dir)
    _generate_annotation(val_dataset, 'val.txt', annotation_dir)

    return train_loader, val_loader, len(class_names), class_names

def initialize_model(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Model initialization"""
    # Load pre-trained model
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Modify the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move to specified device
    model = model.to(device)

    return model

def train_model(model, train_loader, val_loader, num_epochs=20, data_dir='EX2/flower_dataset'): # Added data_dir for saving model path
    """Model training"""
    device = next(model.parameters()).device  # Get the device the model is on

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # List for storing history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Determine the actual number of samples in train and val datasets
    # This is important because random_split creates subsets.
    # For DataLoader, len(dataloader.dataset) gives the size of the dataset it wraps.
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)


    start_time = time.time() # Start timer for total training

    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Start timer for this epoch
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
                dataset_size = train_dataset_size
            else:
                model.eval()
                dataloader = val_loader
                dataset_size = val_dataset_size

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step() # Step the scheduler after a training epoch

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Record loss and accuracy
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item()) # .item() to get Python number
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item()) # .item() to get Python number

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Ensure work_dir path is correctly formed relative to data_dir or a fixed path
                    work_dir = os.path.join(os.path.dirname(data_dir), 'work_dir')
                    os.makedirs(work_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pth'))

        epoch_time_elapsed = time.time() - epoch_start_time
        print(f'Epoch completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')
        print()


    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history # Return model and history

def plot_training_history(history, num_epochs, save_dir='EX2/work_dir'):
    """
    Plot training and validation loss and accuracy curves

    Parameters:
    history (dict): Dictionary containing lists for 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
    num_epochs (int): Total number of training epochs.
    save_dir (str): Directory to save the plot.
    """
    epochs_range = range(num_epochs)

    plt.figure(figsize=(12, 5))

    # Plot Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout() # Adjust layout to prevent overlap

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_filename = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_filename)
    print(f"Training history plot saved to {plot_filename}")

    plt.show() # Display the plot

def main():
    # Define data directory and batch size
    data_dir = 'EX2/flower_dataset' # Make sure this path is correct
    num_epochs_to_run = 20 # Define number of epochs
    batch_size_to_use = 32 # Define batch size

    # 1. Prepare data
    print("Preparing data...")
    train_loader, val_loader, num_classes, _ = prepare_data(data_dir=data_dir, batch_size=batch_size_to_use)
    print(f"Data prepared. Number of classes: {num_classes}")

    # 2. Initialize model
    print("Initializing model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(num_classes, device)
    print(f"Model initialized on device: {device}")

    # 3. Train model
    print("Starting model training...")
    # Pass data_dir to train_model so it knows where to save the best_model.pth relative to the dataset
    model, history = train_model(model, train_loader, val_loader, num_epochs=num_epochs_to_run, data_dir=data_dir)
    print("Training complete!")

    # 4. Plot training history
    print("Plotting training history...")
    # Define where to save the plot, e.g., in the same 'work_dir' as the model
    plot_save_dir = os.path.join(os.path.dirname(data_dir), 'work_dir')
    plot_training_history(history, num_epochs_to_run, save_dir=plot_save_dir)
    print("Training history plotted.")


if __name__ == '__main__':
    # Example usage:
    # Make sure you have a dataset in 'EX2/flower_dataset'
    # The structure should be:
    # EX2/flower_dataset/
    # EX2/flower_dataset/class_a/xxx.jpg
    # EX2/flower_dataset/class_a/xxy.jpg
    # EX2/flower_dataset/class_b/yyy.jpg
    # ...
    # If the directory 'EX2' or 'flower_dataset' doesn't exist where your script is,
    # you'll need to create it or adjust the `data_dir` path.

    # For demonstration, let's assume 'EX2/flower_dataset' might not exist
    # and provide a way to quickly test without real data (though training won't be meaningful)
    if not os.path.exists('EX2/flower_dataset'):
        print("Warning: 'EX2/flower_dataset' not found. Creating dummy directories for testing structure.")
        print("Please replace with your actual dataset for meaningful training.")
        os.makedirs('EX2/flower_dataset/dummy_class_1', exist_ok=True)
        os.makedirs('EX2/flower_dataset/dummy_class_2', exist_ok=True)
        # Create dummy image files (optional, ImageFolder might handle empty dirs, but good for robustness)
        open('EX2/flower_dataset/dummy_class_1/dummy1.jpg', 'a').close()
        open('EX2/flower_dataset/dummy_class_2/dummy2.jpg', 'a').close()

    main()