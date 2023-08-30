import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class UNET_v1(nn.Module):
    def __init__ (self, in_channels, out_channels, num_time_stamps):
        super(UNET_v1,self).__init__()

        self.conv1=nn.Conv3d(in_channels,64,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv2=nn.Conv3d(64,128,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv3=nn.Conv3d(128,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv4=nn.Conv3d(256,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool=nn.MaxPool3d(kernel_size=2)
        # self.clone_conv = nn.Conv3d(64, 64, kernel_size=(1, 1, 1))


        self.conv_trans=nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))

        self.up_conv3=nn.ConvTranspose3d(1024,256,kernel_size=2,stride=2,padding=0)
        self.conv5=nn.Conv3d(512,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.up_conv2=nn.ConvTranspose3d(256,128,kernel_size=2,stride=2,padding=0)
        self.conv6=nn.Conv3d(256,128,kernel_size=(3,3,3),padding=(1,1,1))
        self.up_conv1=nn.ConvTranspose3d(128,64,kernel_size=2,stride=2,padding=0)
        self.conv7=nn.Conv3d(128,64,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv8=nn.Conv3d(64,out_channels,kernel_size=1)



    # def clone_last_time_step(self, x):
    #     x_shape = x.shape
    #     if x_shape[2] % 2 != 0:  # Sprawdzamy, czy drugi wymiar (czas) jest nieparzysty
    #         x_last = x[:, :, -1, :, :].unsqueeze(2)  # Wybierz ostatni krok czasowy i dodaj nowy wymiar
    #         x = torch.cat([x_last, x], dim=2)  # Dodaj ostatni krok czasowy na początku
    #     return x
    


    def forward(self,x):

        # przeplyw down

        
        x1 = nn.functional.relu(self.conv1(x))

        x2 = nn.functional.relu(self.conv2(x1))

        x3 = nn.functional.relu(self.conv3(x2))
        # print("down x3",x3.shape)
        x4 = nn.functional.relu(self.conv4(x3))
        # print("end down",x4.shape)
        


   

        # przeplyw up

        x = nn.functional.relu(self.conv_trans(x4))
        x=torch.mean(x,dim=2)
        x4=torch.mean(x4,dim=2)
        # print('po sredniej: ',x.shape)
        
        x = torch.cat([x, x4], dim=1)
        
        x = nn.functional.relu(self.up_conv3(x))
        x=torch.mean(x,dim=2)
        # print('po 1: ',x.shape)

        x3=torch.mean(x3,dim=2)
        # print('po 2: ',x3.shape)
        x = torch.cat([x, x3], dim=1)
        # print('po sredniej: ',x.shape)




        x = nn.functional.relu(self.conv5(x))
        x=torch.mean(x,dim=2)
        x = nn.functional.relu(self.up_conv2(x))
        x=torch.mean(x,dim=2)
        x2=torch.mean(x2,dim=2)
        x = torch.cat([x, x2], dim=1)
        # print('po sredniej: ',x.shape)




        x = nn.functional.relu(self.conv6(x))
        x=torch.mean(x,dim=2,keepdim=True)
        x = nn.functional.relu(self.up_conv1(x))
        x=torch.mean(x,dim=2,keepdim=True)
        x1=torch.mean(x1,dim=2,keepdim=True)
        x = torch.cat([x, x1], dim=1)

        # print('po sredniej: ',x.shape)
        x = nn.functional.softmax(self.conv7(x),dim=1)
        _ , predicted_class=torch.max(x,dim=1)
        # predicted_class=predicted_class.long()
        # print('shape:',x.shape)
        to_cut = self.conv8(x)
        # print('TC shape:',to_cut.shape)
        # out = torch.squeeze(to_cut,dim=1)
        return(to_cut,predicted_class)