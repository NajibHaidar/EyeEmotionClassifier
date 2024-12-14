import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import EmotionClassifier

def main():
    # 2. Configuration / paths
    demo_data_folder = r"demo/demo_data"        # Folder with sample images subfolders (one subfolder per emotion/class)
    results_folder = r"results"   # Where to store output predictions, confusion matrix, etc.
    model_path = r"checkpoints/final_model.pth"  # Pre-trained model weights

    # Ensure results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # 3. Create transforms (matching the transforms used at test time)
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 4. Load the demo dataset
    #    We use `ImageFolder` assuming your demo folder structure is like demo/happy/*.jpg, demo/sad/*.jpg, etc.
    #    Each subfolder is treated as a separate class. 
    demo_dataset = ImageFolder(root=demo_data_folder, transform=test_transforms)
    demo_loader = DataLoader(demo_dataset, batch_size=8, shuffle=False)

    # If you want the classes (subfolder names)
    class_names = demo_dataset.classes
    print("Class names (inferred):", class_names)

    # 5. Load the pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EmotionClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 6. Perform inference on the demo dataset
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in demo_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 7. Calculate metrics on the "demo set" (since it also has labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("Demo Set Metrics:")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    # 8. Save confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Demo Set)")
    cm_path = os.path.join(results_folder, "confusion_matrix_demo.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # 9. Save predictions to a text or CSV file
    prediction_path = os.path.join(results_folder, "predictions_demo.csv")
    with open(prediction_path, "w") as f:
        f.write("filename,true_label,predicted_label\n")
        # Each sample in the dataset has a path:  demo_dataset.samples[i] = (path, class_index)
        for i, (sample_path, true_label_idx) in enumerate(demo_dataset.samples):
            filename = os.path.basename(sample_path)
            true_label_name = class_names[true_label_idx]
            predicted_label_name = class_names[all_preds[i]]
            f.write(f"{filename},{true_label_name},{predicted_label_name}\n")

    print(f"Predictions written to: {prediction_path}")
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()
