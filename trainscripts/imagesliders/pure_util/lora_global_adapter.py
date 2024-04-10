## LoRA with global control
ACTUAL_SCALE = 1.0
from einops import rearrange

from torch import nn
import torch
import numpy as np 
import torch.nn.functional as F

from inspect import isfunction

try:
    from lora import LoRANetwork
except:
    #from trainscripts.imagesliders.lora import LoRANetwork
    from trainscripts.textsliders.lora import LoRANetwork

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class GlobalAdapter(nn.Module):
    def __init__(self, in_dim, channel_mult=[2, 4]):
        super().__init__()
        dim_out1, mult1 = in_dim*channel_mult[0], channel_mult[0]*2
        dim_out2, mult2 = in_dim*channel_mult[1], channel_mult[1]*2//channel_mult[0]
        self.in_dim = in_dim
        self.channel_mult = channel_mult
        
        self.ff1 = FeedForward(in_dim, dim_out=dim_out1, mult=mult1, glu=True, dropout=0.1)
        self.ff2 = FeedForward(dim_out1, dim_out=dim_out2, mult=mult2, glu=True, dropout=0.3)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_out1)

    def forward(self, x):
        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        x = rearrange(x, 'b (n d) -> b n d', n=self.channel_mult[-1], d=self.in_dim).contiguous()
        return x

class LoRADisabledScaleMetwork(LoRANetwork):
    
    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0.0

class LoRAGlobalSingleScaleAdapter(LoRADisabledScaleMetwork):
    def __init__(self, *args, **kwargs):
        if 'global_dim' in kwargs:
            global_dim = kwargs['global_dim']
            kwargs.pop('global_dim')
        else:
            global_dim = 768
        if 'global_mult' in kwargs:
            global_mult = kwargs['global_mult']
            kwargs.pop('global_mult')
        else:
            global_mult = [2, 4]
        if 'converter_dim' in kwargs:
            converter_dim = kwargs['converter_dim']
            kwargs.pop('converter_dim')
        else:
            converter_dim = 768

        super().__init__(*args, **kwargs)
        self.global_adapter = GlobalAdapter(global_dim, global_mult)
        self.global_dim_converter = nn.Linear(converter_dim, global_dim)

    def forward(self, x):
        
        ones = torch.ones(x.size(0), 768, device=x.device) #[B,768] as an input
        global_token = self.global_adapter(ones * self.scale)
        lora_input = torch.cat([x, global_token], dim=1) # need to verify dimension
        return (
            self.org_forward(x) + self.lora_up(self.lora_down(lora_input)) * self.multiplier * ACTUAL_SCALE
        )
    
   

class LoRAGlobalMultiScaleAdapter(LoRADisabledScaleMetwork):
    def __init__(self, *args, **kwargs):
        if 'global_input_dim' in kwargs:
            global_input_dim = kwargs['global_input_dim']
            kwargs.pop('global_input_dim')
        else:
            global_input_dim = 2

        if 'global_dim' in kwargs:
            global_dim = kwargs['global_dim']
            kwargs.pop('global_dim')
        else:
            global_dim = 768
        if 'global_mult' in kwargs:
            global_mult = kwargs['global_mult']
            kwargs.pop('global_mult')
        else:
            global_mult = [2, 4]
        if 'converter_dim' in kwargs:
            converter_dim = kwargs['converter_dim']
            kwargs.pop('converter_dim')
        else:
            converter_dim = 768

        super().__init__(*args, **kwargs)
        self.global_input = nn.Sequential(
            nn.Linear(global_input_dim, global_dim),
            nn.Tanh() #Why TANH? because we previous experiment dataset with scale [-1,1] and it completly work just fine
        )
        self.global_adapter = GlobalAdapter(global_dim, global_mult)
        self.global_dim_converter = nn.Linear(converter_dim, global_dim)

    def forward(self, x):
        
        token_input = self.global_input(self.scale.to(self.global_input.device))
        global_token = self.global_adapter(token_input)
        lora_input = torch.cat([x, global_token], dim=1) # need to verify dimension
        return (
            self.org_forward(x) + self.lora_up(self.lora_down(lora_input)) * self.multiplier * ACTUAL_SCALE
        )

class LoRAMappingNetwork(LoRADisabledScaleMetwork):
    def __init__(self, *args, **kwargs):
        if 'global_input_dim' in kwargs:
            global_input_dim = kwargs['global_input_dim']
            kwargs.pop('global_input_dim')
        else:
            global_input_dim = 2

        if 'global_dim' in kwargs:
            global_dim = kwargs['global_dim']
            kwargs.pop('global_dim')
        else:
            global_dim = 768
        if 'global_mult' in kwargs:
            global_mult = kwargs['global_mult']
            kwargs.pop('global_mult')
        else:
            global_mult = [2, 4]
        if 'converter_dim' in kwargs:
            converter_dim = kwargs['converter_dim']
            kwargs.pop('converter_dim')
        else:
            converter_dim = 768
        
        if 'learnable_matrix' in kwargs:
            learnable_matrix = kwargs['learnable_matrix']
            kwargs.pop('learnable_matrix')
        else:
            learnable_matrix = None

        super().__init__(*args, **kwargs)
        
        if learnable_matrix is None:
            self.global_input = nn.Sequential(
                nn.Linear(global_input_dim, global_dim),
                #nn.Tanh() #Why TANH? because we previous experiment dataset with scale [-1,1] and it completly work just fine
             )
            self.global_adapter = GlobalAdapter(global_dim, global_mult)
            self.global_dim_converter = nn.Linear(converter_dim, global_dim)
            self.learnable_matrix = None
        else:
            generator = torch.Generator().manual_seed(0)
            self.learnable_matrix = nn.Parameter(torch.normal(0, 1, size=(learnable_matrix, 4, 768), generator=generator), requires_grad=True)

    def get_global_token(self, x):
        if self.learnable_matrix is None:
            token_input = self.global_input(x)  #[1,768]
            global_token = self.global_adapter(token_input) #[4,768]
        else:
            global_token = self.learnable_matrix[x]
        return global_token
    
    def prepare_optimizer_params(self):
        all_params = []

        all_params = all_params + super().prepare_optimizer_params()

        if self.learnable_matrix is not None:
            all_params.append({"params": self.learnable_matrix})
        else:
            all_params.append({"params":self.global_input.parameters()})
            all_params.append({"params":self.global_adapter})
            all_params.append({"params":self.global_dim_converter})


        return all_params

    


# class LoRAGlobalAdapter(LoRAModule):
#     def __init__(self, org_module, lora_dim, alpha, multiplier, train_method, channel_mult=[2, 4]):
#         super().__init__(org_module, lora_dim, alpha, multiplier, train_method)
#         self.global_adapter = GlobalAdapter(lora_dim, channel_mult)
        
#     def forward(self, x):

#         if self.multipler > 0: 
#             global_token = self.global_adapter(x)
#             lora_input = torch.cat([x, global_token], dim=1) # need to verify dimension
#         else:
#             lora_input = x
#         return (
#             self.org_forward(x)
#             + self.lora_up(self.lora_down(lora_input)) * self.multiplier * self.scale
#         )
    