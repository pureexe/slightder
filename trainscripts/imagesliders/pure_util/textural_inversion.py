# TEST with at least 5 images to make sure code is being able to do a textureal inversion 

import torch
import torch.nn as nn

import os 
from typing import Optional, List, Type, Set, Literal
from safetensors.torch import save_file


class TexturalInversionNetwork(nn.Module):

    def __init__(
        self,
        learnable_matrix = 1,
        learnable_size=4,
        *args, 
        **kwargs
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.learnable_matrix = self.learnable_matrix = nn.Parameter(torch.normal(0, 1, size=(learnable_matrix, learnable_size, 768), generator=generator), requires_grad=True)

    def prepare_optimizer_params(self):
        return self.parameters() 

    def get_global_token(self, x):
        return self.learnable_matrix[x]


    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file) 
    
    def set_lora_slider(self, scale):
        """ DO NOTHING, KEEP AS COMPATIBILITY WITH LoRANetwork """
        pass

    def __enter__(self):
        """ DO NOTHING, KEEP AS COMPATIBILITY WITH LoRANetwork """
        pass

    def __exit__(self, exc_type, exc_value, tb):
        """ DO NOTHING, KEEP AS COMPATIBILITY WITH LoRANetwork """
        pass