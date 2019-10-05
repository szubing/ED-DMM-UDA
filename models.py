"""Discriminator model for ADDA."""

from torch import nn
import torch

 
def get_model(name, n_outputs, input_channels=None, patch_size=5, n_classes=None):

    if name == 'EmbeddingNetHyperX':
        model = EmbeddingNetHyperX(input_channels, n_outputs=n_outputs, patch_size=patch_size, n_classes=n_classes).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005)
        scheduler = None

        return model.cuda(), opt, scheduler
    
    if name == 'DiscNetHyperX':
        model = DiscriminatorHyperX(input_dims=n_outputs,
                                      hidden_dims=64,
                                      output_dims=2)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005)
        scheduler = None
        
        return model.cuda(), opt, scheduler

######################## hyperspectral
class DiscriminatorHyperX(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(DiscriminatorHyperX, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )
        
    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
####################
###################################################################
class EmbeddingNetHyperX(nn.Module):
    def __init__(self, input_channels, n_outputs=128, patch_size=5, n_classes=None):
        super(EmbeddingNetHyperX, self).__init__()
        self.dim = 200 ## 200 pavia is good;
        self.convnet = nn.Sequential(
            # 1st conv layer
            # input [input_channels x patch_size x patch_size]
            nn.Conv2d(input_channels,self.dim,kernel_size=1,padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(self.dim,self.dim,kernel_size=1,padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(self.dim,self.dim,kernel_size=1,padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(self.dim,self.dim,kernel_size=1,padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(patch_size,stride=1)
        )
        self.n_classes = n_classes
        self.n_outputs = n_outputs
        self.fc = nn.Sequential(nn.Linear(self.dim, self.n_outputs))
        
    def extract_features(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)
        return output

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
##################################################