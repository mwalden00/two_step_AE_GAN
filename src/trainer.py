import torch as t
from tqdm import trange
from GAN import WGAN
import pandas as pd
import numpy as np
import seaborn as sns

class trainer():

    def __init__(self, model: WGAN, training_loader, testing_loader):
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.model = model

    def train_epochs(self, N):
        losses = pd.DataFrame(data=np.zeros(100), columns=['Generator loss', 'Critic loss'], index=range(100))
        with trange(100,desc='epoch:') as pbar:
            for i in pbar:
                loss_G, loss_D = self.model.train_one_epoch()
                pbar.desc = f'G Loss:{loss_G}, D Loss:{loss_D}, epoch:'
                losses.loc[i,'Generator loss'] = loss_G
                losses.loc[i,'Critic loss'] = loss_D
        sns.lineplot(losses, x='Epoch', y='Loss')
        return losses
        

    


