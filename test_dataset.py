from data_loader.dataset import BreastCancerDataset
from data_loader.transform import get_transforms
import pandas as pd
from data_loader.preprocess_csv_data import preprocess_csv_data

IMG_DIR = 'data/archive'
CATEGORY_AUX_TARGETS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
df = pd.read_csv('data/MetaData/train.csv')
df = preprocess_csv_data(df, CATEGORY_AUX_TARGETS, n_folds=5)
dataset = BreastCancerDataset(df=df, img_dir=IMG_DIR, train=True, categories_classes=CATEGORY_AUX_TARGETS, transform=get_transforms(aug=True))
for i in range(1, 10): 
    print(dataset.__getitem__(i))
