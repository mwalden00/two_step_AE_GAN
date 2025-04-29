from torch import nn
import torch.functional as F
import torch as t
from tqdm import tqdm

class basic_G(nn.module):
    '''
    Basic G implementation. Embedding layer followed by 4-layer LSTM.
    '''
    def __init__(self, in_dim, h_dim, device=t.device('cpu')):
        super(self,basic_G).__init__()
        self.embedding = nn.Linear(in_dim*2, h_dim)
        self.lstm = nn.LSTM(h_dim,h_dim,num_layers=4)
        self.out = nn.Linear(h_dim,in_dim)
    
    def forward(self, x):
        return self.out(self.lstm(self.embedding(x)))
    
class basic_D(nn.module):
    '''
    Basic D implementation. 4-layer LSTM, followed by linear + RELU activation.
    '''
    def __init__(self, in_dim, h_dim):
        super(self,basic_D).__init__()
        self.LSTM_block = nn.LSTM(in_dim, h_dim, num_layers=4,bidirectional=True)
        self.linear = nn.Linear(h_dim, in_dim)

    def forward(self, x):
        return F.sigmoid(self.linear(self.lstm(x)))
    
class WGAN():

    def __init__(self, in_dim):
        self._G = basic_G(in_dim, in_dim)
        self._D = basic_D(in_dim, in_dim)
        self._optim_G = t.optim.Adam(self.G.parameters(), lr=0.001, betas=(0.5,0.999))
        self._optim_D = t.optim.Adam(self.D.parameters(), lr=0.001, betas=(0.5,0.999))

    @property
    def D(self):
        return self._D

    @property
    def G(self):
        return self._G
    
    @property
    def optimizers(self):
        '''
        Generator optim, Discriminator optim
        '''
        return self._optim_G, self._optim_D

    def loss_G(self, x, m, z):
        x_z = x + z
        x_tilde = self.G(t.cat([x_z,m],1))

        L_ort = nn.MSELoss(x * m - x_tilde * m) # Observation  reconstruction
        x_hat = x * m + x_tilde * (1-m)
        y = self.D(x_hat)
        L_G = t.sum(y * (1-m))/t.sum(1-m) # Critique loss

        return L_ort + L_G
    
    def loss_D(self, x, m, z, l = 10):
        x_z = x + z
        x_tilde = self.G(t.cat([x_z,m],1))
        x_hat = x * m + x_tilde * (1-m)
        y = self.D(x_hat)
        L_D = t.sum(y * m)/t.sum(m) - t.sum(y * (1-m))/t.sum(1-m) # Wasserstein estimate
        L_reg = l * self._grad_penalty(x,self.G(t.cat([z,t.zeros(z.shape)],1)),m) # Grad penalty

        return L_D + L_reg
    
    def _grad_penalty(self, x_real, x_fake, m):
        epsilon = t.rand(x_real.shape)
        x_interp = (x_real * epsilon + x_fake * (1-epsilon)).required_grad_(True)
        y_interp = self.D(x_interp) * m
        fake_weights = t.autograd.Variable(
            t.Tensor(x_real.shape[0], 1).fill_(1.0), requires_grad=False)
        gradients = t.autograd.grad(
            outputs=y_interp,
            inputs=x_interp,
            grad_outputs=fake_weights,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (((gradients**2).sum()**0.5 - 1)**2).mean()
        return gradient_penalty
    
    def train_one_epoch(self, training_loader: t.utils.data.DataLoader):
        optim_G, optim_D = self.optimizers
        loss_G_avg = 0
        for i, (x,m) in training_loader:
            z = t.normal(0,1,size=x.shape)
            optim_G.zero_grad()
            loss_G = self.loss_G(x,m,z)
            loss_G.backwards()
            loss_G_avg = loss_G_avg+loss_G
        loss_D_avg = 0
        for n in range(5):
            for j, (x,m) in training_loader:
                z = t.normal(0,1,size=x.shape)
                optim_D.zero_grad()
                loss_D = self.loss_D(x,m,z)
                loss_D.backward()
                loss_D_avg = loss_D_avg+loss_D
        return loss_G_avg / i, loss_D_avg / (j*5)
        



    
    
