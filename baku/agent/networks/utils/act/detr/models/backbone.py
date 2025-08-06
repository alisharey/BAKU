# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import List

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import os, sys
baku_agent_path = os.path.dirname(os.path.abspath(__file__))
baku_path = os.path.dirname(baku_agent_path)
vggt_parent_path = "/l/users/ali.abouzeid/BAKU/baku/vggt"
print(f"Adding VGGT path: {vggt_parent_path}")  # Debugging line
if vggt_parent_path not in sys.path:
    sys.path.insert(0, vggt_parent_path)

from vggt.models.vggt import VGGT # Corrected import path

    
import IPython

e = IPython.embed

class VGGT_BackBone(nn.Module):
    def __init__(self, use_cache=False, intermediate_layer_idx=[4, 11, 17, 23]):
        super().__init__()
        self.use_cache = use_cache
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intermediate_layer_idx = intermediate_layer_idx
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        for param in self.vggt.parameters():
            param.requires_grad = False
        self.feature_projector = VGGTProjector(
            use_cache=use_cache,
            intermediate_layer_idx=intermediate_layer_idx,
            input_dim=2048,  # VGGT output dimension
            output_dim=512,  # Desired output dimension
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, num_views, seq_len, hidden_dim]
        
        Returns:
            List of feature tensors from specified layers.
        """
        with torch.no_grad():
            vggt_features_list, ps_idx = self.vggt.aggregator(x)
        features = self.feature_projector(vggt_features_list)
        
        return features

class VGGTProjector(nn.Module):
    def __init__(self, use_cache=False, intermediate_layer_idx=[4, 11, 17, 23], input_dim=2048, output_dim=512):
        super().__init__()
        self.intermediate_layer_idx = intermediate_layer_idx
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_cache = use_cache
        
        # Conv layers to process each intermediate layer - remove adaptive pooling
        self.layer_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 1024, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(1024, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                # Remove AdaptiveAvgPool1d to preserve spatial dimensions
            ) for _ in range(len(intermediate_layer_idx))
        ])
        
        # Final projection to reshape to 4x4 spatial grid
        self.final_proj = nn.Sequential(
            nn.Linear(512 * len(intermediate_layer_idx), 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim * 4 * 4)  # Output 512 * 16 = 8192 values
        )
    
    def forward(self, aggregated_tokens_list):
        """
        Args:
            aggregated_tokens_list: List of length 24, we use indices [4, 11, 17, 23]
        
        Returns:
            torch.Tensor: Shape [batch_size, 2, 512, 4, 4]
        """
        # Extract the 4 intermediate layers
        if not self.use_cache:
            selected_tokens = []
            for idx in self.intermediate_layer_idx:
                selected_tokens.append(aggregated_tokens_list[idx])
            selected_tokens = torch.stack(selected_tokens, dim=1)
        else:
            selected_tokens = aggregated_tokens_list
            
        batch_size, num_layers, num_views, seq_len, hidden_dim = selected_tokens.shape
        
        # Process each view separately
        output_views = []
        for view_idx in range(num_views):
            view_tokens = selected_tokens[:, :, view_idx, :, :]  # [batch_size, 4, 86, 2048]
            
            # Process each layer
            layer_features = []
            for layer_idx in range(num_layers):
                layer_token = view_tokens[:, layer_idx, :, :]  # [batch_size, 86, 2048]
                
                # Transpose for conv1d: [batch_size, 2048, 86]
                layer_token = layer_token.transpose(1, 2)
                
                # Apply convolutions - now keeps spatial dimension
                processed = self.layer_convs[layer_idx](layer_token)  # [batch_size, 512, 86]
                
                # Average pool over sequence length to get single representation
                processed = processed.mean(dim=-1)  # [batch_size, 512]
                layer_features.append(processed)
            
            # Concatenate all layer features
            view_features = torch.cat(layer_features, dim=1)  # [batch_size, 512 * 4]
            
            # Final projection and reshape to 4x4 grid
            view_output = self.final_proj(view_features)  # [batch_size, 512 * 16]
            view_output = view_output.view(batch_size, self.output_dim, 4, 4)  # [batch_size, 512, 4, 4]
            output_views.append(view_output)
        
        # Stack views to get [batch_size, 2, 512, 4, 4]
        output = torch.stack(output_views, dim=1)
        return output




class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos
    

class VGGTJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        features = self[0](tensor_list)  # [batch, 2, 512, 4, 4]
        
        out: List[NestedTensor] = []
        pos = []
        
        # Handle each view separately
        for view_idx in range(features.shape[1]):
            view_features = features[:, view_idx]  # [batch, 512, 4, 4]
            out.append(view_features)
            # Generate position encoding for this view
            pos.append(self[1](view_features).to(view_features.dtype))

        return out, pos
    


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    
    if args.backbone == "vggt":
        backbone = VGGT_BackBone()
        model = VGGTJoiner(backbone, position_embedding)
        model.num_channels = 512  # or 1024 if you concatenate views
    else:
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
    
    return model
