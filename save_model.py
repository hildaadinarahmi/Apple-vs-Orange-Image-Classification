import torch
import torch.nn as nn
import pickle

# Definisikan kembali SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Wrapper
class ModelWrapper:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

# Inisialisasi model
model = SimpleCNN()
class_names = ["Apple", "Orange"]
wrapper = ModelWrapper(model, class_names)

# Simpan ke file .pkl
with open("fruit_classifier_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)

print("✅ Model saved as 'fruit_classifier_model.pkl'")
