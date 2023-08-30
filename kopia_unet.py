import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class UNET_v1(nn.Module):
    def __init__ (self, in_channels, out_channels, num_time_stamps):
        super(UNET_v1,self).__init__()

        self.conv1=nn.Conv3d(in_channels,64,kernel_size=(3,3,3),padding=1)
        self.conv2=nn.Conv3d(64,128,kernel_size=(3,3,3),padding=1)
        self.conv3=nn.Conv3d(128,256,kernel_size=(3,3,3),padding=1)
        self.conv4=nn.Conv3d(256,512,kernel_size=(3,3,3),padding=1)
        self.pool=nn.MaxPool3d(kernel_size=2,stride=2)


        self.conv_trans=nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))

        self.up_conv3=nn.ConvTranspose3d(1024,256,kernel_size=2,stride=2,padding=0)
        self.conv5=nn.Conv3d(512,256,kernel_size=(3,3,3),padding=1)
        self.up_conv2=nn.ConvTranspose3d(256,128,kernel_size=2,stride=2,padding=0)
        self.conv6=nn.Conv3d(256,128,kernel_size=(3,3,3),padding=1)
        self.up_conv1=nn.ConvTranspose3d(128,64,kernel_size=2,stride=2,padding=0)
        self.conv7=nn.Conv3d(128,64,kernel_size=(3,3,3),padding=1)
        self.conv8=nn.Conv3d(64,out_channels,kernel_size=1)

    def forward(self,x):

        # przeplyw down


        x1 = nn.functional.relu(self.conv1(x))
        p1 = self.pool(x1)
        x2 = nn.functional.relu(self.conv2(p1))
        p2 = self.pool(x2)
        x3 = nn.functional.relu(self.conv3(p2))
        p3 = self.pool(x3)
        x4 = nn.functional.relu(self.conv4(p3))

   

        # przeplyw up

        x = nn.functional.relu(self.conv_trans(x4))

        x = torch.cat([x, x4], dim=1)



        x = nn.functional.relu(self.up_conv3(x))

        x = torch.cat([x, x3], dim=1)

        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.up_conv2(x))

        x = torch.cat([x, x2], dim=1)

        x = nn.functional.relu(self.conv6(x))
        x = nn.functional.relu(self.up_conv1(x))

        x = torch.cat([x, x1], dim=1)


        x = nn.functional.relu(self.conv7(x))

        out = self.conv8(x)
 
            
        return(out)