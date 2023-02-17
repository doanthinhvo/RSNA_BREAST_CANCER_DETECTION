# Test dataset.py 
from data_loader.dataset import BreastCancerDataset
from data_loader.preprocess_csv_data import preprocess_csv_data
import pandas as pd
from data_loader.transform import get_transforms
IMAGE_DIR = 'data/RSNA-cut-off-empty-space'
DATA_DIR = 'data/RSNA-Screening-mamography-breast-cancer-detection'
CATEGORY_AUX_TARGETS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
import matplotlib.pyplot as plt
# Transform
tfm = get_transforms(aug=True)
df_train = pd.read_csv(f'{DATA_DIR}/train.csv')
df_train = preprocess_csv_data(df_train, CATEGORY_AUX_TARGETS, 5)

ds_train = BreastCancerDataset(df_train, IMAGE_DIR, 'train', CATEGORY_AUX_TARGETS, tfm)
X, y_cancer, y_aux = ds_train[42]
print(X.shape)
plt.figure(figsize=(20, 20))
for i in range(8):
    v = X.permute(1, 2, 0)
    v -= v.min()
    v /= v.max()
    # plt.imshow(v)
    # break
    plt.subplot(2, 4, i + 1).imshow(v)
    # 
plt.show()
plt.tight_layout()





