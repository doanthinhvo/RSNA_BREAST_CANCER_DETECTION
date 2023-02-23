# '''
# Test inf loop
# '''
# from utils import *

# from torch.utils.data import Dataset, DataLoader

# class simpleDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#     def __getitem__(self, index):
#         return self.data[index]
#     def __len__(self):
#         return len(self.data)
    
# data = [1,2,3,4,5,6,7,8,9,10]
# dataset = simpleDataset(data)
# dataloader_base = DataLoader(dataset, batch_size=3, shuffle=False)
# dataloader = inf_loop(dataloader_base)
# print(dataloader)

# print("======dataloader_base=======")
# for batch_idx, data in enumerate(dataloader_base):
    
#     print(data)
# print("=====dataloader=====")
# for batch_idx, data in enumerate(dataloader):
#     print(batch_idx)
#     print(data)
#     if batch_idx == 3:
#         break

# for batch_idx, data in enumerate(dataloader):
#     print(batch_idx)
#     print(data)
#     if batch_idx == 4:
#         break


'''
test MetricTracker
'''

from utils import *

print(MetricTracker)
# #CHECKPOINT: test MetricTrainer + Github khorlund
train_metrics = MetricTracker('loss', 'acc')
for i in range(1, 3):
    train_metrics.update('loss', i)
    train_metrics.update('acc', 2*i)
print(train_metrics.result())