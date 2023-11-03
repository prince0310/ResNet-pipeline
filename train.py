import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import argparse
# sys.path.append("/home/whitewalker/Documents/prince/resnet_pipeline")
from model import resnet101

torch.cuda.empty_cache()

class ResNetTrainer:
    def __init__(self, data_dir, num_classes, batch_size=32, learning_rate=0.001, num_epochs=10):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.dataloaders = None

    def _initialize_model(self):
        # Create a ResNet model from scratch
        self.model = resnet101(num_classes = self.num_classes) 
        self.model.to(self.device)
        print(self.model)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _load_data(self):
        # Data augmentation and preprocessing
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # Load the dataset
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    def train(self):
        self._initialize_model()
        self._load_data()

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Save the trained model
        torch.save(self.model, 'best.pt')

# Example usage:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")

    args = parser.parse_args()

    data_directory = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    num_classes = len(os.listdir(data_directory))

    trainer = ResNetTrainer(data_directory, num_classes,  batch_size, learning_rate, num_epochs)
    trainer.train()


# example of training 

# python3 train.py --data_dir /home/whitewalker/Documents/prince/gender/data/demo  --batch_size 32 --learning_rate 0.001  --num_epochs 20
