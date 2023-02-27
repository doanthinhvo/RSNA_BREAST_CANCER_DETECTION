from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

class BreastCancerDataset(Dataset): 
    def __init__(self, df, img_dir, train, categories_classes, transform=None):
        super(BreastCancerDataset, self).__init__() # super() returns a proxy object that allows you to refer parent class by 'self'.
        self.df = df
        self.img_dir = img_dir
        # self.img_size = img_size
        self.train = train
        self.categories_classes = categories_classes
        self.transform = transform

    def __getitem__(self, i):

        path = f'{self.img_dir}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        try:
            img = Image.open(path).convert('RGB')
            # print(f"first img in dataset: {img.size}")
            # plt.imshow(np.array(img), cmap='gray')
            # plt.title("Before augmentation")
            # plt.show()
        except Exception as ex:
            print(path, ex)
            return None

        if self.transform is not None:
            img = self.transform(img)
            # print(f"second img in dataset: {img.size}")

        if self.train:
            '''
            cols_values: a tensor of shape (11,), which each element is a class label of each category.(col)'''
            cancer = torch.as_tensor(self.df.iloc[i].cancer)
            cols_values = torch.as_tensor(self.df.iloc[i][self.categories_classes])
            return img, cancer, cols_values
            '''
            cancer: torch.Size([])
            cols_values.shape: torch.Size([11])
            cols_values: tensor([1, 1, 1, 0, 0, 0, 3, 4, 0, 1, 5])
            '''
            
        return img

    def __len__(self):
        return len(self.df)
