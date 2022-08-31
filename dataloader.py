from torchvision import datasets, models, transforms
import os
import torch
data_dir = "./"

class Data_Loader():
    def __init__(self, input_size, batch_size):
        self.input_size = input_size
        self.batch_size = batch_size
        
    def dataform(self):
        data_transforms = {
            'train': transforms.Compose([
                # x = 224
                # x = 299
                transforms.Resize([int(self.input_size), int(self.input_size)]), #bilinear
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'validation': transforms.Compose([
                # x = 224
                #x = 299
                transforms.Resize([int(self.input_size), int(self.input_size)]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'validation']}
        return dataloaders_dict


