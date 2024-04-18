import torch
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)

feature = torch.nn.Sequential(*list(model.children())[:])
features = list(model.children())[:-2]#去掉池化层及全连接层
#print(list(model.children())[:-2])
modelout=nn.Sequential(*features).to(device)
print(features)
print(modelout)

print(model)
print(feature)