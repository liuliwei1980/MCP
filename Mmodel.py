import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from GAT import GAT



class Mmodel(torch.nn.Module):
    def __init__(self):
        super(Mmodel, self).__init__()

        self.model0 = Sequential(
            # 0
            # 输入3通道、输出64通道、卷积核大小、步长、补零、
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()
        self.aap = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten(start_dim=1)
        self.fc = Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()
        self.fp1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )
        self.fp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

        self.l4 = Linear(1024, 128)
        self.R12 = ReLU()
        self.l5 = Linear(128, 32)
        self.R13 = ReLU()
        self.l6 = Linear(4, 1)
        self.R14 = ReLU()
        self.sig2 = nn.Sigmoid()
        self.GraphA = GAT(78, 78, 78, 0.2, 0.2, 4)
        self.BN30 = nn.BatchNorm1d(78)
        self.relu30 = nn.ReLU()
        self.lgat1 = Linear(2340, 256)
        self.lgat2 = Linear(256, 1)
        self.dropout = nn.Dropout(0.2)
    def forward(self,data,ecfp,hash1,list_of_node,list_of_edge):
        x_gat = torch.zeros_like(list_of_node)
        for i in range(list_of_edge.size()[0]):
            a = list_of_edge[i]
            e = list_of_node[i]
            x_gat_1 = self.GraphA(e, a)
            x_gat_1 = self.BN30(x_gat_1)
            x_gat_1 = self.relu30(x_gat_1)
            x_gat[i] = x_gat_1#128.30.78
        x_gat = x_gat.view(x_gat.size(0), -1)#128.2340
        x_gat = self.lgat1(x_gat)
        x_gat = self.lgat2(x_gat)
        # x_gat = self.sigmoid1(x_gat)
        ecfp = self.fp1(ecfp)
        ecfp = self.dropout(ecfp)
        hash1 = self.fp2(hash1)
        hash1 = self.dropout(hash1)
        x = self.model0(data)
        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        # x = self.sigmoid2(x)
        # x_out =torch.cat((x, x_gat,hash1,ecfp), dim=1)
        # x_out = self.l6(x_out)
        x_z =  (hash1 +ecfp)*0.1
        x_out = x + x_gat+x_z
        # x_out = x + x_gat
        
        x_out = self.sigmoid3(x_out)
        x_out = x_out.squeeze()
        return x_out
