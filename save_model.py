import torch
import torch.nn as nn
import pickle

class_names = ['Apple', 'Orange']

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 32 * 32, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

class ModelWrapper:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

wrapper = ModelWrapper(model, class_names)

with open("fruit_classifier_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)

print("✅ Model saved to fruit_classifier_model.pkl")
