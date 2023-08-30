from unet_nn_parts import *

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UNet,self).__init__()
        
        self.n_channels = n_channels
        self.n_classes=n_classes

        self.inc=(DoubleConv(n_channels,64))

        self.down1=(Down(64, 128))
        self.down2=(Down(128, 256))
        self.down3=(Down(256, 512))
        self.down4=(Down(512, 1024))

        self.up1=(Up(1024, 512))
        self.up2=(Up(512, 256))
        self.up3=(Up(256, 128))
        self.up4=(Up(128, 64))
        
        self.outc=(OutConv(64,n_classes))

    
    def forward(self,x):


        # print("x shape:", x.shape)
        x1 = self.inc(x)
        # print("x1 shape:", x1.shape)

        x2 = self.down1(x1)
        # print("x2 shape:", x2.shape)
        x3 = self.down2(x2)
        # print("x3 shape:", x3.shape)
        x4 = self.down3(x3)
        # print("x4 shape:", x4.shape)
        x5 = self.down4(x4)
        # print("x5 shape:", x5.shape)

        # print("one more time x shape:", x.shape)

        x5mean=torch.mean(x5,dim=2)
        # print('x5mean: ', x5mean.shape)
        x4mean=torch.mean(x4,dim=2)
        # print('x4mean: ', x4mean.shape)
        x = self.up1(x5mean,x4mean )
        # print("cat1 x shape:", x.shape)
        
        x = self.up2(x, torch.mean(x3,dim=2))
        # print("cat2 x shape:", x.shape)
        x = self.up3(x, torch.mean(x2,dim=2))
        # print("cat3 x shape:", x.shape)
        x = self.up4(x, torch.mean(x1,dim=2))
        # print("cat4 x shape:", x.shape)

        log = self.outc(x)

        return log




