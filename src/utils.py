import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to show images
def show_random_images_batch(dataset, class_names, num_images=8):
    """Randomly selects and displays a batch of images and their labels."""
    indices = random.sample(range(len(dataset)), num_images)  # Randomly select indices
    images, labels = zip(*[dataset[i] for i in indices])  # Get the corresponding images and labels

    images = torch.stack(images)  # Stack images into a single batch
    labels = torch.tensor(labels)  # Stack labels into a tensor
    
    fig, axes = plt.subplots(1, min(len(images), num_images), figsize=(20, 5))  # Show up to 'num_images' images
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        if img.shape[-1] == 1:  # If grayscale, squeeze to remove last channel
            img = img.squeeze()
            ax.imshow(img, cmap='gray')  # Use grayscale color map
        else:
            ax.imshow(img)  # Display RGB image
        ax.set_title(f"Label: {class_names[labels[i]]}")
        ax.axis('off')
    plt.show()


def visualize_predictions(model, test_loader, class_names, device, num_images=16):
    """Visualize a random set of test images along with predictions."""
    model.eval()

    # Collect all data from the test_loader
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)

    # Concatenate all batches into a single dataset
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    # Select random indices for visualization
    random_indices = random.sample(range(len(all_images)), num_images)

    # Prepare the figure
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))  # 4x4 grid for 16 images

    images_shown = 0
    with torch.no_grad():
        for idx in random_indices:
            # Get the image and label for the current random index
            image = all_images[idx]
            label = all_labels[idx]

            # Move image and label to the device
            image, label = image.to(device).unsqueeze(0), label.to(device)

            # Forward pass through the model
            output = model(image)
            _, pred = torch.max(output, 1)

            # Convert image to NumPy format for display
            img = image.cpu().squeeze().numpy()
            if img.shape[0] == 1:  # Grayscale image (1 channel)
                img = img.squeeze()  # Remove channel dimension
                cmap = 'gray'
            else:  # RGB image (3 channels)
                img = img.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                cmap = None

            # Get true and predicted labels
            true_label = class_names[label.item()]
            predicted_label = class_names[pred.item()]

            # Select subplot for the image
            ax = axes[images_shown // 4, images_shown % 4]
            ax.imshow(img, cmap=cmap)

            # Set title with true and predicted labels
            ax.set_title(
                f"True: {true_label}\nPred: {predicted_label}",
                color='green' if true_label == predicted_label else 'red'
            )

            # Turn off the axis
            ax.axis('off')
            images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Seaborn heatmap.
    
    Args:
        cm (numpy.ndarray): Confusion matrix obtained from sklearn.metrics.confusion_matrix
        class_names (list): List of class labels, in order they appear in cm
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def visualize_incorrect_predictions(model, test_loader, class_names, device, num_images=8):
    """Visualize a random set of incorrect predictions from the test set."""
    model.eval()

    # Collect all data from the test_loader
    all_images, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_images.append(images)
            all_labels.append(labels)
            all_preds.append(preds)

    # Concatenate all batches into single tensors
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Find indices of incorrect predictions
    incorrect_indices = (all_preds != all_labels).nonzero(as_tuple=True)[0]

    # Randomly sample up to num_images incorrect predictions
    if len(incorrect_indices) == 0:
        print("No incorrect predictions to display.")
        return
    
    sampled_indices = random.sample(list(incorrect_indices.cpu().numpy()), min(num_images, len(incorrect_indices)))

    # Prepare the figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid for num_images=8
    images_shown = 0

    for idx in sampled_indices:
        # Get the image, true label, and predicted label for the current index
        image = all_images[idx]
        true_label = class_names[all_labels[idx].item()]
        predicted_label = class_names[all_preds[idx].item()]

        # Convert image to NumPy format for display
        img = image.cpu().numpy()
        if img.shape[0] == 1:  # Grayscale image (1 channel)
            img = img.squeeze()  # Remove channel dimension
            cmap = 'gray'
        else:  # RGB image (3 channels)
            img = img.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            cmap = None

        # Select subplot for the image
        ax = axes[images_shown // 4, images_shown % 4]
        ax.imshow(img, cmap=cmap)

        # Set title with true and predicted labels
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color='red')

        # Turn off the axis
        ax.axis('off')
        images_shown += 1

        if images_shown >= num_images:
            break

    plt.tight_layout()
    plt.show()

    
def plot_loss_curves(train_losses, val_losses):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()