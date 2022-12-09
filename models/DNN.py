import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F

class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(1048,128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,64)
        self.linear4 = nn.Linear(64,16)
        self.classifier = nn.Linear(16,2)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # print(x.shape)
        feature1 = F.relu(self.linear1(x))
        feature1_dropout = self.dropout(feature1)

        feature2 = F.relu(self.linear2(feature1_dropout))
        feature2_dropout = self.dropout(feature2)

        feature3 = F.relu(self.linear3(feature2_dropout))
        feature3_dropout = self.dropout(feature3)

        feature4 = F.relu(self.linear4(feature3_dropout))
        feature4_dropout = self.dropout(feature4)

        out = self.classifier(feature4_dropout)


        return out, feature4, [feature1,feature2,feature3,feature4]

class CNNModel_merge(nn.Module):

    def __init__(self):
        super(CNNModel_merge, self).__init__()
        self.feature1 = nn.Sequential(
            # Conv - 1
            nn.Conv1d(3, 8, 3, stride=2),
            nn.AvgPool1d(2),
            nn.ReLU(),

            # Conv - 2
            nn.Conv1d(8, 16, 3, stride=2),
            nn.AvgPool1d(2),
            nn.ReLU(),
        )

        self.feature2 = nn.Sequential(
            nn.Linear(2399,512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Linear(512,2)

    def forward(self, x1, x2, x3):
        out = self.feature1(x2)
        out = out.view(-1,16*127)
        feature_1 = torch.cat([x1,out,x3],dim=1)
        feature_2 = self.feature2(feature_1)

        out = self.classifier(feature_2)


        return out, feature_2, [feature_1, feature_2]


# 测试网络的正确性
if __name__ == '__main__':
    model = LinearModel()
    input = torch.ones([64,1048])
    # print(input.shape)
    output,_ = model(input)
    print(output.shape)

# summary(LinearModel().to('cuda'), input_size= (1,166), batch_size=64, device='cuda')

# # model = LinearModel()
# # print(model)