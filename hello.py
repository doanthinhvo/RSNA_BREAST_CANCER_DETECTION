# # Test dataset.py 
# from data_loader.dataset import BreastCancerDataset
# from data_loader.preprocess_csv_data import preprocess_csv_data
# import pandas as pd
# from data_loader.transform import get_transforms
# IMAGE_DIR = data/RSNA-cut-off-empty-space
# DATA_DIR = 'data/RSNA-Screening-mamography-breast-cancer-detection'
CATEGORY_AUX_TARGETS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
# import matplotlib.pyplot as plt
# # Transform
# tfm = get_transforms(aug=True)
# df_train = pd.read_csv(f'{DATA_DIR}/train.csv')
# df_train = preprocess_csv_data(df_train, CATEGORY_AUX_TARGETS, 5)

# ds_train = BreastCancerDataset(df_train, IMAGE_DIR, 'train', CATEGORY_AUX_TARGETS, tfm)
# X, y_cancer, y_aux = ds_train[42]
# print(X.shape)
# plt.figure(figsize=(20, 20))
# for i in range(8):
#     v = X.permute(1, 2, 0)
#     v -= v.min()
#     v /= v.max()
#     # plt.imshow(v)
#     # break
#     plt.subplot(2, 4, i + 1).imshow(v)
#     # 
# plt.show()
# plt.tight_layout()


# Test data_loader.py
from data_loader.data_loaders import BreastCancerDataloader
from data_loader.transform import get_transforms
from data_loader.preprocess_csv_data import preprocess_csv_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

IMAGE_DIR = 'data/RSNA-cut-off-empty-space'
df_file_path = 'data/RSNA-Screening-mamography-breast-cancer-detection/train.csv'
# dataloader = BreastCancerDataloader(get_transforms(aug=True), IMAGE_DIR, df_file_path, 4, True, 0.2, 4, train=True)
# test dataloader
# print(os.listdir(IMAGE_DIR))
df = pd.read_csv(df_file_path)
df = preprocess_csv_data(df, CATEGORY_AUX_TARGETS, 5)
# get df that has value of patient_id in list of patient_id in RNSA-cut-off-empty-space
current_patient_id = [10215, 10391, 10483, 10122, 10124, 10412, 10324, 10151, 10385, 10366, 10322, 10329, 10051, 10223, 10486, 10355, 10243, 10328, 10514, 10282, 10342, 10438, 10289, 1028, 10508, 10025, 1014, 10198, 10394, 10011, 10097, 1025, 10130, 10116, 10179, 10445, 10006, 10132, 10494, 10219, 10401, 1026, 10153, 10404, 10234, 10208, 10038, 10434, 10335, 10224, 10302, 10489, 10506, 10468, 10285, 10257, 10478, 10442, 10182, 10383, 10439, 1036, 1045, 10086, 10413, 10095, 10050, 105, 10309, 10144, 10317, 10353, 10363, 10152, 10126, 10106, 10314, 10487, 10200, 10175, 10426, 10119, 10315, 10429, 10399, 10232, 10136, 10406, 10048, 10185, 10273, 10183, 10407, 10424, 10226, 10102, 10432, 10388, 10042, 1015, 10267, 10512, 10188, 10308, 10509, 10428, 10217, 10240, 10049, 10359]
df = df[df['patient_id'].isin(current_patient_id)]
AUX_TARGET_NCLASSES = df[CATEGORY_AUX_TARGETS].max() + 1
print(AUX_TARGET_NCLASSES)
# print aux_target_nclasses as a list 
print(AUX_TARGET_NCLASSES.tolist())
# dataloader = BreastCancerDataloader(get_transforms(aug=True), IMAGE_DIR, df_file_path, 4, True, 0.2, 4, train=True)
# for i, (X, y_cancer, y_aux) in enumerate(dataloader):
#     print(X.shape)
#     print(y_cancer.shape)
#     print(y_aux.shape)
#     print(y_cancer)
#     print(y_aux)
#     break
