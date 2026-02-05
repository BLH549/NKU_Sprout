
import jittor as jt
from jittor import init
from jittor import nn

class jt_AODNet(nn.Module):

    def __init__(self):
        super(jt_AODNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv(3, 3, 1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv(3, 3, 3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv(6, 3, 5, stride=1, padding=2, bias=True)
        self.conv4 = nn.Conv(6, 3, 7, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv(12, 3, 3, stride=1, padding=1, bias=True)

    def execute(self, x):
        x1 = nn.relu(self.conv1(x))
        x2 = nn.relu(self.conv2(x1))
        concat1 = jt.contrib.concat((x1, x2), dim=1)
        x3 = nn.relu(self.conv3(concat1))
        concat2 = jt.contrib.concat((x2, x3), dim=1)
        x4 = nn.relu(self.conv4(concat2))
        concat3 = jt.contrib.concat((x1, x2, x3, x4), dim=1)
        x5 = nn.relu(self.conv5(concat3))
        clean_image = nn.relu((((x5 * x) - x5) + 1))
        return clean_image
