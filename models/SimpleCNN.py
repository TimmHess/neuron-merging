import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SimpleCNN']

defaultcfg = [8, "M", 16, "M", 32]

class SimpleCNN(nn.Module):
    def __init__(self, out_classes=10, init_weights=True, cfg=None):
        super(SimpleCNN, self).__init__()
        if cfg is None:
            cfg = defaultcfg

        self.cfg = cfg

        self.feature = self.make_layers(cfg)
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1]*3*3, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, out_classes)
        )

        self._initialize_weights_kaiming()

        
    def make_layers(self, cfg):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        #print("features:", x.size())
        x = x.view(x.size(0), -1)
        
        y = self.classifier(x)
        #y = y.view(y.size(0),-1)
        return y


    def _initialize_weights_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    
    def extract_feature(self, x, preReLU=False):
        x = x.cuda()
        feat1 = self.feature(x)
        if not preReLU:
            feat1 = F.relu(feat1)
        return [feat1]

    def get_cfg(self):
        return self.cfg