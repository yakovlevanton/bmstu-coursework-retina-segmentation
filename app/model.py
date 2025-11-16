import torch
import torch.nn as nn
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNetPP(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Encoding path
        self.conv0_0 = conv_block(in_channels, 32)
        self.pool0 = nn.MaxPool2d(2)

        self.conv1_0 = conv_block(32, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_0 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_0 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_0 = conv_block(256, 512)

        self.up1_0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.conv0_1 = conv_block(32 + 32, 32)      # 64
        self.conv1_1 = conv_block(64 + 64, 64)      # 128
        self.conv2_1 = conv_block(128 + 128, 128)   # 256
        self.conv3_1 = conv_block(256 + 256, 256)   # 512

        self.conv0_2 = conv_block(32 + 32 + 32, 32)       # 96
        self.conv1_2 = conv_block(64 + 64 + 64, 64)       # 192
        self.conv2_2 = conv_block(128 + 128 + 128, 128)   # 384

        self.conv0_3 = conv_block(32 + 32 + 32 + 32, 32)  # 128
        self.conv1_3 = conv_block(64 + 64 + 64 + 64, 64)  # 256

        self.conv0_4 = conv_block(32 + 32 + 32 + 32 + 32, 32)  # 160

        self.final1 = nn.Conv2d(32, 1, kernel_size=1)
        self.final2 = nn.Conv2d(32, 1, kernel_size=1)
        self.final3 = nn.Conv2d(32, 1, kernel_size=1)
        self.final4 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, inputs, deep_supervision=False):
        # Encoder
        x0_0 = self.conv0_0(inputs)
        down0 = self.pool0(x0_0)

        x1_0 = self.conv1_0(down0)
        down1 = self.pool1(x1_0)

        x2_0 = self.conv2_0(down1)
        down2 = self.pool2(x2_0)

        x3_0 = self.conv3_0(down2)
        down3 = self.pool3(x3_0)

        x4_0 = self.conv4_0(down3)

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_0(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_0(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_0(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)], dim=1))

        output1 = torch.sigmoid(self.final1(x0_1))
        output2 = torch.sigmoid(self.final2(x0_2))
        output3 = torch.sigmoid(self.final3(x0_3))
        output4 = torch.sigmoid(self.final4(x0_4))

        if deep_supervision:
            return [output1, output2, output3, output4]
        else:
            return output4