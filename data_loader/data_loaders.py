from torchvision import datasets, transforms
from .dataset import BreastCancerDataset
from torch.utils.data import DataLoader
import pandas as pd
from .preprocess_csv_data import preprocess_csv_data
from sklearn.model_selection import train_test_split
from .transform import get_transforms

class BreastCancerDataloader(DataLoader):
    def __init__(self, data_dir, df_file_path, batch_size, shuffle, validation_split, num_workers, pin_memory=True, train=True,alpha=None, balance=None, transforms=get_transforms(aug=False)):
        self.CATEGORY_AUX_TARGETS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']

        self.df = pd.read_csv(df_file_path)
        self.df = preprocess_csv_data(self.df, self.CATEGORY_AUX_TARGETS, 5)
        self.data_dir = data_dir
        self.transforms, self.shuffle = transforms, shuffle
        # BUG: Fix: remove next 2 lines
        # current_patient_id = [10215, 10391, 10483, 10122, 10124, 10412, 10324, 10151, 10385, 10366, 10322, 10329, 10051, 10223, 10486, 10355, 10243, 10328, 10514, 10282, 10342, 10438, 10289, 1028, 10508, 10025, 1014, 10198, 10394, 10011, 10097, 1025, 10130, 10116, 10179, 10445, 10006, 10132, 10494, 10219, 10401, 1026, 10153, 10404, 10234, 10208, 10038, 10434, 10335, 10224, 10302, 10489, 10506, 10468, 10285, 10257, 10478, 10442, 10182, 10383, 10439, 1036, 1045, 10086, 10413, 10095, 10050, 105, 10309, 10144, 10317, 10353, 10363, 10152, 10126, 10106, 10314, 10487, 10200, 10175, 10426, 10119, 10315, 10429, 10399, 10232, 10136, 10406, 10048, 10185, 10273, 10183, 10407, 10424, 10226, 10102, 10432, 10388, 10042, 1015, 10267, 10512, 10188, 10308, 10509, 10428, 10217, 10240, 10049, 10359]
        # current_patient_id = [10317, 10353, 10363, 10152, 10126, 10106, 10314, 10487, 10200, 10175, 10426, 10119]
        
        # self.df = self.df[self.df['patient_id'].isin(current_patient_id)]
        self.train_df, self.val_df = train_test_split(self.df, test_size=validation_split, random_state=42)
        train_dataset = BreastCancerDataset(self.train_df, self.data_dir, train, self.CATEGORY_AUX_TARGETS, self.transforms)
        super().__init__(train_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=pin_memory)
    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            val_dataset = BreastCancerDataset(self.val_df, self.data_dir, True, self.CATEGORY_AUX_TARGETS, self.transforms)
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        # def __init__(self, df, img_dir, train, categories_classes, transform=None):
