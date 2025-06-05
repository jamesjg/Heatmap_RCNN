import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Direct Regression (only use heatmap)
"""

class Heatmap_Direct_Regression_Module(nn.Module):
    def __init__(self,config) -> None:
        super(Heatmap_Direct_Regression_Module, self).__init__()
        
        self.config = config
        self.embed_offset = config.model.embed_offset if hasattr(config.model, "embed_offset") else False
        self.mid_layer_nums = config.model.roi_layers - 2 if hasattr(config.model, "roi_layers") else 0

        self.num_landmarks = config.data.num_landmarks
        flat_size = self.get_flat_size()

        extra_input_c = 0
        if self.embed_offset:
            self.embed_fn, input_ch = get_embedder(3)
            extra_input_c = input_ch * (config.model.use_roi - 1)
        
        self.act_0 = nn.Sequential()

        self.input_x_c = extra_input_c+flat_size
        self.fc_inp = nn.Linear(self.input_x_c ,256)
        self.act_1 = nn.LeakyReLU(inplace=True)

        mid_layers = [nn.Sequential(nn.Linear(256,256),nn.LeakyReLU(inplace=True)) for _ in range(self.mid_layer_nums)]
        self.mid_layers = nn.ModuleList(mid_layers)

        self.fc_oup = nn.Linear(256, 2)
        

    def get_flat_size(self):
        model_info = self.config.model

        if hasattr(model_info, "roi_size") and model_info.roi_size is not None:
            return model_info.roi_size ** 2
        
        if hasattr(model_info, "roi_sizes") and model_info.roi_sizes is not None:
            flat_size = 0
            for roi_size in model_info.roi_sizes:
                flat_size += roi_size ** 2
            return flat_size

    def forward_module(self, x):
        """
            Args:
                x : input heatmap feature (batch, num_landmarks, 1 , roi_size, roi_size)
            Return:
                out: (num_landmarks, 2) the offset based on the center roi
        """
        x = x.view(-1, self.input_x_c)
        
        x = self.act_0(x)

        x = self.fc_inp(x)
        x = self.act_1(x)

        for fc_layer in self.mid_layers:
            x = fc_layer(x)

        x = self.fc_oup(x)

        x = x.view(-1, self.num_landmarks, 2)
        
        return x
    
    def forward(self,x,**kwargs):
        if len(kwargs) !=0 and kwargs["resolution_landmark_offsets"] !=0:
            res_land_offs = kwargs["resolution_landmark_offsets"]
            embed_offs = []
            for off in res_land_offs:
                embed_offs.append(self.embed_fn(off))

            x = torch.cat([x]+embed_offs, -1)

        return self.forward_module(x)
    
    def init_weights(self,pretrained=''):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        print("Finish init roi module weights")


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2, # score value is only 1 number
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
