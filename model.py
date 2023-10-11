import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import PIL.Image
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, label = self.data_list[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class Model:
    def __init__(self, num_classes):
        self.model = models.mobilenet_v2(pretrained=True)
        num_features = self.model.classifier[1].in_features  # Get the number of input features from the pretrained model
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        self.model_state = None  # model-specific state to reset


    def train(self, class_counters, num_epochs):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


        img_list = []
        class_list = []

        for class_num, counter in enumerate(class_counters):
            for i in range(0, counter + 1):
                img = cv.imread(f'{class_num}/frame{i + 1}.jpg')
                img = PIL.Image.fromarray(img)
                img = transform(img)
                img_list.append(img)
                class_list.append(class_num)

        if len(img_list) == 0:
            print("No training data found.")
            return

        dataset = CustomDataset(list(zip(img_list, class_list)))

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Adjust batch size as needed
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Adjust learning rate as needed

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

        print("Training completed!")

    def predict(self, frame):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        frame_pil = PIL.Image.fromarray(frame)

        frame_tensor = transform(frame_pil).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(frame_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    def reset(self):
        # Reset any state or knowledge related to objects
        self.model_state = None  # Reset model-specific state if applicable