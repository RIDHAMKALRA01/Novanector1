import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from PIL import Image
import torch.cuda.amp as amp  # For mixed precision training

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Preparing the Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use num_workers=0 on Windows to avoid multiprocessing issues
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Step 2: Building the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable mixed precision training only if CUDA is available
    use_amp = torch.cuda.is_available()
    scaler = amp.GradScaler(enabled=use_amp)  # Correct initialization

    # Step 3: Training the Model
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with amp.autocast(enabled=use_amp):  # Mixed precision only if CUDA is available
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Step 4: Evaluating the Model
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_true_labels, all_predictions, target_names=class_names))

    # Step 5: Testing on Custom Images
    def preprocess_custom_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((32, 32))
        img = transform(img).unsqueeze(0)
        return img.to(device)

    custom_image_path = 'I:\\AI\\cat.jpg'  # Replace with your image path
    custom_image = preprocess_custom_image(custom_image_path)
    output = model(custom_image)
    _, predicted_class = torch.max(output, 1)
    print(f"\nPredicted Class for Custom Image: {class_names[predicted_class.item()]}")

if __name__ == '__main__':
    # Required for multiprocessing on Windows
    torch.multiprocessing.freeze_support()
    main()

"""
# Description
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This project uses a simple CNN model implemented in PyTorch to classify these images.

# Task Description
The goal of this project is to develop an image classification system using AI. The following steps were performed:

1. Data Collection and Preprocessing:
   - The CIFAR-10 dataset was downloaded and preprocessed using transformations like resizing, normalization, and conversion to tensors.

2. Model Training:
   - A Convolutional Neural Network (CNN) was designed and trained on the dataset.
   - Techniques like dropout were used to handle overfitting.

3. Evaluation:
   - The model was evaluated using metrics like precision, recall, and F1-score.

4. Testing on Unseen Data:
   - The model was tested on custom images to assess its real-world performance.

5. Documentation:
   - Detailed documentation was provided, including the model architecture, training process, and usage instructions.

# Model Architecture
The CNN model used in this project has the following architecture:

1. Convolutional Layers:
   - Conv1: 3 input channels (RGB), 32 output channels, kernel size 3x3, padding 1.
   - Conv2: 32 input channels, 64 output channels, kernel size 3x3, padding 1.

2. Pooling Layer:
   - MaxPooling with a 2x2 kernel and stride 2.

3. Fully Connected Layers:
   - FC1: Input size 64 * 8 * 8 (flattened output from Conv2), output size 512.
   - FC2: Input size 512, output size 10 (one for each class).

4. Dropout:
   - Dropout layer with a probability of 0.5 to prevent overfitting.

5. Activation Function:
   - ReLU activation is used after each convolutional and fully connected layer.

# Installation
To run this project, you need to have Python and PyTorch installed. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cifar10-pytorch.git
   cd cifar10-pytorch
Install dependencies:

pip install torch torchvision numpy matplotlib scikit-learn pillow
Download the CIFAR-10 dataset:
The dataset will be automatically downloaded when you run the script for the first time.

Testing on Custom Images
You can test the model on custom images by providing the path to the image. The script will preprocess the image, pass it through the model, and predict the class.


custom_image_path = 'path/to/your/image.jpg'
output = model(preprocess_custom_image(custom_image_path))
_, predicted_class = torch.max(output, 1)
print(f"Predicted Class: {class_names[predicted_class.item()]}")
Preprocessing Steps:
The image is resized to 32x32 pixels.

It is normalized using the same parameters as the training data.
"""
