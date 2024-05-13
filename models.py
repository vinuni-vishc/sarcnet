import torchvision.models as models
import torch
import torch.nn as nn

class SarcNet(nn.Module):
    def __init__(self, num_classes=1, num_features=11):
        super(SarcNet, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        res_modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*res_modules)
        
        self.ln1 = nn.Linear(num_features, 128)
        self.ln2 = nn.Linear(128, 256)
        self.ln3 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
        
        self.fc = self.fcClassifier(512 * 2, num_classes)

    def forward(self, feature, image):
        x_image = self.resnet18(image)
        x_image = x_image.view(x_image.shape[0], -1)
        
        feature = feature.to(torch.float32)
        
        x_feature = self.ln1(feature)
        x_feature = self.relu(x_feature)
        x_feature = self.ln2(x_feature)
        x_feature = self.relu(x_feature)
        x_feature = self.ln3(x_feature)
        x_feature = self.relu(x_feature)
        x_feature = torch.squeeze(x_feature)
        
        x = torch.cat((x_image, x_feature), dim=1)
        
        x = self.fc(x)
        
        return x

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, numClasses),
        )

        return fc_classifier