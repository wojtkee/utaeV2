import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):

    def __init__(self, in_channels,out_channels):

        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )





    def forward(self, x):
        return self.double_conv(x)
    
class DoubleConv_2d(nn.Module):

    def __init__(self, in_channels,out_channels):

        self.double_conv_2d = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        return self.double_conv_2d(x)
    


class Down(nn.Module):

    def __init__(self,in_channels,out_channels):

        super().__init__()
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels,out_channels)
        )


    def forward(self, x):

        return self.maxpool_conv(x)



class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up=nn.ConvTranspose2d(in_channels, (in_channels//2), kernel_size=2, stride=2)
        self.conv_2d=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.norm_2d=nn.BatchNorm2d(out_channels)
        self.relu_2d=nn.ReLU(inplace=True)
    

    def forward(self,x1,x2):
        
        # print('xx1', x1.shape)
        x1=self.up(x1)
        # print('xxx1', x1.shape)
        # print('xxx2', x2.shape)

        # diffY=x2.size()[2]-x1.size()[2]
        # diffX=x2.size()[3]-x1.size()[3]

        # x1=F.pad()

        x=torch.cat([x2,x1],dim=1)
        # print('final x shape: ',x.shape)

        rethelp=self.conv_2d(x)
        rethelp_norm=self.norm_2d(rethelp)
        retReLU=self.relu_2d(rethelp_norm)
        # print('retshape', retReLU.shape)
        return retReLU
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv,self).__init__()
        
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)


    def forward(self, x):
        return self.conv(x)
