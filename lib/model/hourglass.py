import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import collections
import sys
import math
sys.path.append('.')
from .roi import Heatmap_Direct_Regression_Module


class ConvBNReLu(nn.Module):
    def __init__(self,inp,oup,kernel=3,stride=1,bn=True,relu=True):
        super(ConvBNReLu,self).__init__()
        
        self.conv = nn.Conv2d(inp,oup,kernel,stride,padding=(kernel-1)//2,bias=False)
        self.bn = nn.BatchNorm2d(oup) if bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Sequential()

    def forward(self,x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Residual(nn.Module):
    def __init__(self,inp,oup):
        super(Residual,self).__init__()
        self.conv1 = ConvBNReLu(inp,int(oup/2),1)
        self.conv2 = ConvBNReLu(int(oup/2),int(oup/2),3)
        self.conv3 = ConvBNReLu(int(oup/2),oup,1,relu=False)
        self.skip_layer = ConvBNReLu(inp,oup,1,relu=False)
        self.need_skip = False if inp == oup else True
        self.last_relu = nn.ReLU(inplace=True)

    def forward(self,x):
        
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.last_relu(out)

        return out

class Down_Block(nn.Module):
    def __init__(self,inp,oup,stride=2):
        super(Down_Block,self).__init__()  
        self.conv1 = ConvBNReLu(inp,oup,3,stride=stride)
        self.conv2 = ConvBNReLu(oup,oup,3,relu=False)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            ConvBNReLu(inp,oup,1,stride=stride,relu=False)
        ) if stride!=1 or inp!=oup else nn.Sequential()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.downsample(x)
        out = self.relu(out)
        return out



class Hourglass(nn.Module):
    def __init__(self,config):
        super(Hourglass,self).__init__()
        self._n = config.backbone.num_layer
        self._f = config.backbone.num_feature
        self.config = config
        self._init_layers(self._n,self._f)

    def _init_layers(self,n,f):
        setattr(self, 'res' + str(n) + '_1', Residual(f, f))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(f, f))
        if n > 1:
            self._init_layers(n - 1, f)
        else:
            self.res_center = Residual(f, f)
        setattr(self, 'res' + str(n) + '_3', Residual(f, f))

    def _forward(self, x, n, f):
        up1 = eval('self.res' + str(n) + '_1')(x)

        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)

        return up1 + up2

    def forward(self,x):
        return  self._forward(x,self._n,self._f)

class Multi_Outputs_Hourglass(Hourglass):
    """
        The Last Hourglass is with multiple resolution feature maps compared with standrad hourglass
    """
    def __init__(self,config):
        super(Multi_Outputs_Hourglass,self).__init__(config)
        self.output_resolutions = [int(64 / 2 ** (r-1)) for r in config.model.output_stages]

    def _forward(self, x, n, f, multi_features):
        up1 = eval('self.res' + str(n) + '_1')(x)

        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f, multi_features)
        else:
            low2 = self.res_center(low1)
        multi_features.append(low2)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)

        return up1 + up2
    
    def forward(self,x):
        multi_features = []
        # Note multi features is in reversed  (4x4->8x8->...->64x64) 
        multi_features.append(self._forward(x,self._n,self._f, multi_features))
        # # so reverse to (64->32->...->4)
        multi_features.reverse()

        output_features = [feat for feat in multi_features if feat.shape[-1] in self.output_resolutions]

        return output_features

class StackedHourGlass(nn.Module):
    def __init__(self,config):
        super(StackedHourGlass,self).__init__()
        self.config = config
        self.stride = config.data.input_size // config.heatmap.heatmap_size
        self.num_feature = config.backbone.num_feature

        self.pre_conv = nn.Sequential(
            ConvBNReLu(3,self.num_feature//4,7,2),
            Residual(self.num_feature // 4, self.num_feature // 2),
            nn.MaxPool2d(2,2),
            Residual(self.num_feature // 2, self.num_feature // 2),
            Residual(self.num_feature // 2, self.num_feature))

        self._init_stacked_hourglass(config)

        if hasattr(config.model, "use_roi") and config.model.use_roi:
            self.roi_module = Heatmap_Direct_Regression_Module(config)

        self.init_weights(config.model.ckpt if config.model.ckpt is not None else '')


    # init hourglass
    def _init_stacked_hourglass(self,config):
        for i in range(config.backbone.num_stack):
            setattr(self, 'hg' + str(i), Hourglass(config))
            setattr(self, 'hg' + str(i) + '_res1',
                    Residual(self.num_feature, self.num_feature))
            setattr(self, 'hg' + str(i) + '_lin1',
                    ConvBNReLu(self.num_feature, self.num_feature,1))
            setattr(self, 'hg' + str(i) + '_conv_pred',
                    nn.Conv2d(self.num_feature, config.data.num_landmarks, 1))
            # if self.offset_func is not None:
            #     setattr(self, 'offset_' + str(i),
            #         self.offset_func(config)) 
            if i < self.config.backbone.num_stack - 1:
                setattr(self, 'hg' + str(i) + '_conv1',
                        ConvBNReLu(self.num_feature, self.num_feature, 1))
                setattr(self, 'hg' + str(i) + '_conv2',
                        ConvBNReLu(config.data.num_landmarks, self.num_feature, 1))

    def forward(self,x):
        x = self.pre_conv(x)

        out_preds = []

        for i in range(self.config.backbone.num_stack):
            hg = eval('self.hg'+str(i))(x)
            ll = eval('self.hg'+str(i)+'_res1')(hg)
            feature = eval('self.hg'+str(i)+'_lin1')(ll)
            preds = eval('self.hg'+str(i)+'_conv_pred')(feature)
            out_preds.append(preds)

            if i < self.config.backbone.num_stack - 1:
                merge_feature = eval('self.hg'+str(i)+'_conv1')(feature)
                merge_preds = eval('self.hg'+ str(i)+'_conv2')(preds)
                x = x+merge_feature+merge_preds

        if self.training:

            out_preds = torch.stack(out_preds,dim=1)
            # out_preds size: n_batch,n_stack,n_landmark,heatmap_size,heatmap_size

            # filter select indices to supervised
            # out_preds = out_preds[:,self.config.backbone.sel_indices,:,:]
            
            return out_preds

        else:
            return out_preds[-1]

    def forward_roi(self,x, **kwargs):
        
        offset = self.roi_module(x, **kwargs)

        return offset


    def init_weights(self,pretrained=''):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        print("Finish init hourglass weights")

        if os.path.isfile(pretrained) or os.path.islink(pretrained):
            
            pretrained_dict = torch.load(pretrained,map_location=(f'cuda:{self.config.train.gpu_id}'))
            if not isinstance(pretrained_dict,collections.OrderedDict):    
                # suppose state_dict in pretrained_dict 
                if isinstance(pretrained_dict['model_state_dict'],collections.OrderedDict):
                    pretrained_dict = pretrained_dict['model_state_dict']
                else :
                    raise ValueError("cannot find the state_dict in {}".format(pretrained))

            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class StackedHourGlass_MultiOutputs(StackedHourGlass):
    def __init__(self, config):
        super(StackedHourGlass_MultiOutputs, self).__init__(config)
        self._init_stacked_hourglass(config)
        self._init_last_stack(config)
        
        self.output_resolutions = [int(64 / (2 ** (r - 1))) for r in config.model.output_stages] # output_stages start from 1, represent 64x64
        self.init_weights(config.model.ckpt if config.model.ckpt is not None else '')

    # init hourglass
    def _init_stacked_hourglass(self,config):
        for i in range(config.backbone.num_stack - 1):
            setattr(self, 'hg' + str(i), Hourglass(config))
            setattr(self, 'hg' + str(i) + '_res1',
                    Residual(self.num_feature, self.num_feature))
            setattr(self, 'hg' + str(i) + '_lin1',
                    ConvBNReLu(self.num_feature, self.num_feature,1))
            setattr(self, 'hg' + str(i) + '_conv_pred',
                    nn.Conv2d(self.num_feature, config.data.num_landmarks, 1))

            setattr(self, 'hg' + str(i) + '_conv1',
                    ConvBNReLu(self.num_feature, self.num_feature, 1))
            setattr(self, 'hg' + str(i) + '_conv2',
                    ConvBNReLu(config.data.num_landmarks, self.num_feature, 1))

    def _init_last_stack(self, config):
        
        setattr(self,'hg'+str(config.backbone.num_stack-1), 
                    Multi_Outputs_Hourglass(config))

        for i in range(config.model.use_roi):

            setattr(self, 'hg_roi' + str(i) + '_res1',
                    Residual(self.num_feature, self.num_feature))
            setattr(self, 'hg_roi' + str(i) + '_lin1',
                    ConvBNReLu(self.num_feature, self.num_feature,1))
            setattr(self, 'hg_roi' + str(i) + '_conv_pred',
                    nn.Conv2d(self.num_feature, config.data.num_landmarks, 1))


    def forward(self,x):
        x = self.pre_conv(x)

        out_preds = []

        for i in range(self.config.backbone.num_stack - 1):
            hg = eval('self.hg'+str(i))(x)
            ll = eval('self.hg'+str(i)+'_res1')(hg)
            feature = eval('self.hg'+str(i)+'_lin1')(ll)
            preds = eval('self.hg'+str(i)+'_conv_pred')(feature)
            out_preds.append(preds)

            
            merge_feature = eval('self.hg'+str(i)+'_conv1')(feature)
            merge_preds = eval('self.hg'+ str(i)+'_conv2')(preds)
            x = x+merge_feature+merge_preds

        # process last stack
        last_idx = self.config.backbone.num_stack - 1
        multi_features = eval('self.hg'+str(last_idx))(x)  # the index is keep same with the stage

        multi_stage_preds = []
        
        for i in range(len(multi_features)):
            stage_feat = multi_features[i]
            ll = eval('self.hg_roi'+str(i)+'_res1')(stage_feat)
            feature = eval('self.hg_roi'+str(i)+'_lin1')(ll)
            stage_preds = eval('self.hg_roi'+str(i)+'_conv_pred')(feature)
            multi_stage_preds.append(stage_preds)

        if self.training:

            out_preds = torch.stack(out_preds,dim=1) if self.config.backbone.num_stack > 1 else torch.tensor([])
            # out_preds size: n_batch,n_stack-1,n_landmark,heatmap_size,heatmap_size

            # filter select indices to supervised
            # out_preds = out_preds[:,self.config.backbone.sel_indices,:,:]
            
            return out_preds, multi_stage_preds

        else:
            return multi_stage_preds

if __name__ == "__main__":
    from lib.utils.parser import train_parser_args as parser_args
    import mmcv
    from lib.utils.config_tool import merge_args_into_config
    args, unparsed = parser_args()
    config = mmcv.Config.fromfile(args.config_file)
    merge_args_into_config(args, unparsed, config)
    model = StackedHourGlass(config)

    input = torch.randn(64,3,256,256)
    output = model(input)

    """
    import time
    start = time.time()
    for i in range(15):
        output = model(input)
    print("per image process time : {:.4f}".format((time.time() - start)/15))
    # print(output.size())
    # exit()
    """

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    exit()

    from thop import profile,clever_format
    input = torch.randn(1,3,256,256)
    flops, params = profile(model,inputs=(input, ), verbose=False)   
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: {} Params: {}".format(flops,params))
