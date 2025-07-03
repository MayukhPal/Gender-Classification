import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Parameters
val_dir = 'C:/Users/91974/Downloads/Comys_Hackathon5/Comys_Hackathon5/Task_A/val'
model_path = 'gender_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

# Transformations - match training transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load validation data
val_data = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Load model architecture & weights
model = models.resnet18(weights=None)  # no pretraining during eval
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: male, female
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Classification report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=val_data.classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(val_data.classes))
plt.xticks(tick_marks, val_data.classes, rotation=45)
plt.yticks(tick_marks, val_data.classes)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Save some sample images with predictions
output_dir = 'val_samples_with_preds'
os.makedirs(output_dir, exist_ok=True)

# Unnormalize function for display
def unnormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # reverse normalization
    tensor = tensor.clamp(0, 1)
    return tensor

print(f"Saving sample images with predictions to {output_dir}...")

for i in range(10):  # save 10 samples
    img_path, true_label_idx = val_data.samples[i]
    true_label = val_data.classes[true_label_idx]

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = val_data.classes[pred_idx]

    # Save image with predicted and true label in filename
    save_path = os.path.join(output_dir, f"true_{true_label}_pred_{pred_label}_{os.path.basename(img_path)}")
    img.save(save_path)

print("Sample images saved. Check the folder for qualitative results.")
