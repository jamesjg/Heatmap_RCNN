from torchvision.models import ResNet101_Weights, ResNet50_Weights, ResNet34_Weights, ResNet18_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone,BackboneWithFPN
from torch import nn
import torch

def get_backbone_info(config):

    model_w = {
        'resnet18' :ResNet18_Weights.DEFAULT, 
        'resnet34' :ResNet34_Weights.DEFAULT, 
        'resnet50' :ResNet50_Weights.DEFAULT, 
        'resnet101' :ResNet101_Weights.DEFAULT
    }
    
    return model_w[config.model.backbone]



class ResNet_FC(nn.Module):
    def __init__(self,config):
        super(ResNet_FC, self).__init__()
        self.config = config
        self.num_heads = len(config.model.output_stages)
        self.output_stages = config.model.output_stages
        self.output_resolutions = [int(64 / (2 ** (r - 1))) for r in config.model.output_stages] # output_stages start from 1, represent 64x64

        self.backbone = resnet_fpn_backbone(config.model.backbone, weights=get_backbone_info(config), trainable_layers=5)
        self.backbone_heads = self._make_head()
        self.roi_fc = ROI_FC(config)

        # print(self.backbone)
        # print(self.backbone_heads)
        for name,p in self.named_parameters():
            print(name, p.requires_grad)
        # exit()


    def _make_head(self):
        backbone_heads = []
        # resnetfpn output channels is 256
        resnet_planes = 256
        for head_i in range(self.num_heads):
            head = nn.Sequential(
                nn.Conv2d(resnet_planes, resnet_planes, 1, 1),
                nn.BatchNorm2d(resnet_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(resnet_planes, self.config.data.num_landmarks, 1, 1)
            )
            backbone_heads.append(head)

        return nn.ModuleList(backbone_heads)


    def forward(self,x):

        fpn_feature = self.backbone(x) # fpn feature dict start from "0"
        outputs = []
        for i in range(len(self.output_stages)):
            stage = self.output_stages[i] # start from 1
            head = self.backbone_heads[i]
            output = head(fpn_feature.get(str(stage - 1)))
            outputs.append(output)

        # for i in range(self.num_heads):
        #     outputs.append(self.backbone_heads[i](fpn_feature.get(str(i))))
        # print([(k,v.shape) for k,v in fpn_feature.items()])
        return outputs



        

        # print(self.parameters())
        # # compute the output
        # x = torch.rand(1,3,256,256)
        # output = self.backbone(x)
        # print([(k, v.shape) for k, v in output.items()])

        # net = resnet18()
        # return_layers = {'layer1':1, 'layer2':2}
        # in_channels_list = [64,128]
        # out_channels = 256
        # diy_backbone = BackboneWithFPN(net, return_layers, in_channels_list, out_channels)
        # output = diy_backbone(x)
        # print([(k, v.shape) for k, v in output.items()])

        # create ROI_FC 
        # self.roi_fc = 


class ROI_FC(nn.Module):
    # only depend on the output heatmap
    def __init__(self, config):
        super(ROI_FC, self).__init__()
        self.config = config
        self.num_landmarks = config.data.num_landmarks

        self.flat_stage_sizes = [7,5,3] # 64, 32, 16
        self.fc1 = nn.Linear(49,256)

    def forward(self, x):
        
        return nn.Sequential()(x)
        




if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from lib.utils.parser import parser_args
    import mmcv
    from lib.utils.config_tool import merge_args_into_config
    args, unparsed = parser_args()
    config = mmcv.Config.fromfile(args.config_file)
    merge_args_into_config(args, unparsed, config)
    model = ResNet_FC(config)


        
