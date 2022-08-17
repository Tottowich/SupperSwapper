"""
File that contains the dataset class for the Facial Feature Detection project.
This project uses CelebA dataset. Which contains 200000 images of celeberties annotated with 40 facial features.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset,random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
import matplotlib.pyplot as plt
CelebATranslation = "5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young"
CelebATranslation = CelebATranslation.split(",")
def translate(prediction,threshold=0.5):
    attributes = []
    for j, val in enumerate(prediction):
        if val >= threshold:
            attributes.append(CelebATranslation[j])
    return attributes
def transform_image(image):
    return transforms.Compose([transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])(image)
class CelebADataset(Dataset):
    """
    CelebA dataset for facial attribute detection
    """
    def __init__(self,
                label_path:str,
                img_dir:str,
                batch_size:int=32,
                train_ratio:float=0.85,
                test_ratio:float=0.05,
                val_ratio:float=0.1,
                _transforms:transforms.transforms=None,
                device:str=None,
                shuffle:bool=True,
                ) -> None:
        super().__init__()
        self.label_path = label_path
        
        self.img_labels, self.img_names = self.load_labels()
        self.img_dir = img_dir
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device == None else device
        self.transform = transforms.Compose([transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]) if _transforms is None else _transforms
    
    def load_labels(self):
        """
        Load the labels from the label file
        """
        images_attrr = pd.read_csv(self.label_path)
        images_attrr.replace(to_replace=-1,value=-1,inplace=True)
        #print(images_attrr)
        return np.array(images_attrr.to_numpy()[:,1:],dtype=np.float32), images_attrr.to_numpy()[:,0] # Remove the first column which is the image name and return numpy array with image names seperate
    def train_val_test_split(self,):
        """
        Split the dataset into train, validation and test sets
        """
        train_size = int(len(self)*self.train_ratio)+1
        val_size = int(len(self)*self.val_ratio)+1
        test_size = int(len(self)*self.test_ratio)
        
        train_set, val_set, test_set = random_split(self, [train_size, val_size, test_size])
        return train_set, val_set, test_set
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = read_image(img_path)
        image = self.transform(image) # Shape: (3, 224, 224)
        label = self.img_labels[idx, :] # 40 Attributes
        
        #print(image.shape, type(image))
        return image, label
    def get_dataloader(self,):
        """
        Get the dataloader for the dataset
        """
        #dataset = CelebADataset(label_path,img_dir,train_ratio,test_ratio,val_ratio=)
        train,val,test = self.train_val_test_split()
        train_dataloader = DataLoader(train, batch_size=self.batch_size, shuffle=self.shuffle)
        val_dataloader = DataLoader(val, batch_size=self.batch_size, shuffle=self.shuffle)
        test_dataloader = DataLoader(test, batch_size=self.batch_size, shuffle=self.shuffle)

        return {"train":train_dataloader,"val":val_dataloader,"test":test_dataloader}
def unnormalize(image):
        """
        Unnormalize the image
        """
        image = image.numpy()
        image = image.transpose(1,2,0)
        image = image*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return image
if __name__ == "__main__":
    dataset = CelebADataset(label_path="data/list_attr_celeba.csv",img_dir="data/img_align_celeba/img_align_celeba/",train_ratio=0.85,test_ratio=0.05,val_ratio=0.1)
    print(len(dataset))
    dataloader = dataset.get_dataloader()
    for i,(image,label) in enumerate(dataloader["train"]):
        print(label)
        for j,val in enumerate(label[0,:]):
            if val == 1:
                print(CelebATranslation[j])
        plt.imshow(unnormalize(image[0]))
        plt.show()
    