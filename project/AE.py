import glob
import torch
import zipfile
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
from torch import nn
import time

class MyDataset(Dataset):

    def __init__(self, data_path: str, split: str, len_train:float, **kwargs):
        self.data_dir = Path(data_path)
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.dat']) 
        self.imgs = imgs[:int(len(imgs) * len_train)] if split == "train" else imgs[int(len(imgs) * len_train):] 

    def __len__(self):
        return len(self.imgs) 

    def __getitem__(self, idx):
        img = np.fromfile(self.imgs[idx], dtype=np.float32) # (2048, )
        img = np.expand_dims(img, axis=0) # (1,2048)
        img = torch.from_numpy(img) # tensor,torch.Size([1,2048])
        return img, 0.0  # dummy datat to prevent breaking

class AutoEncoder(nn.Module):
    def __init__(self, bytes_rate):
        super(AutoEncoder, self).__init__()
        if bytes_rate == 64:
            self.encoder = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
                # nn.Linear(32, 16),
                # nn.ReLU()
            )

            self.decoder = nn.Sequential(

                # nn.Linear(16, 32),
                # nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.Sigmoid()
            )

        elif bytes_rate == 128:
            self.encoder = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
                # nn.Linear(64, 32),
                # nn.ReLU()
            )

            self.decoder = nn.Sequential(

                # nn.Linear(32, 64),
                # nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.Sigmoid()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
                # nn.Linear(128, 64),
                # nn.ReLU()
            )

            self.decoder = nn.Sequential(
                # nn.Linear(64, 128),
                # nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.Sigmoid()
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':
    starttime = time.time()
    torch.manual_seed(1) 
    EPOCH = 10
    BATCH_SIZE = 64
    LR = 0.0001
    N_TEST_IMG = 5
    NUM_WORKS = 1
    DATA_PATH = r'C:\\Users\\yling\\Desktop\\dataset\\train_feature\\'
    # DATA_PATH = 'C:\\Users\\yling\\Desktop\\test_B\\gallery_feature_B\\'
    # DATA_PATH = 'C:\\Users\\cheng\\Desktop\\datasets\\NAIC2021Reid\\gallery_feature_A\\'
    

    Coder = AutoEncoder()  # 实例化
    print(Coder)

    # print(Coder.parameters())
    optimizer = torch.optim.Adam(Coder.parameters(),lr=LR) 
    loss_func = nn.MSELoss() 
    train_data = MyDataset(DATA_PATH, split='train',len_train=0.85, ) # torch.Size([1,2048])
    loader = DataLoader(dataset=train_data,  
                        batch_size=BATCH_SIZE, 
                        num_workers=NUM_WORKS ,  
                        shuffle=True)
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(loader):  
            # print('step:',step,'|'+'(x,y)',x.shape) # torch.size([64,1,2048])
            encoded , decoded = Coder(x)
            # print('step:',step,'|' + 'encoded:',encoded.size()) # torch.Size([64, 1, 16])
            # print('step:',step,'|' + 'dncoded:',decoded.size()) # torch.Size([64, 1, 2048])
            loss = loss_func(decoded,x)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()  

            if step%50 == 0:  
                print('Epoch :', epoch,'|','train_loss:%.4f'%loss.data)

        # torch.save(Coder, 'AutoEncoder'+'_epoch'+str(epoch)+'.pkl') # 保存模型
    torch.save(Coder,'AutoEncoder_256_f32.pkl')
    print('________________________________________')
    print('finish training')

    endtime = time.time()
    print('训练耗时：',(endtime - starttime))


    #Coder = AutoEncoder()
    #Coder = torch.load('AutoEncoder.pkl')