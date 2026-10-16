import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ImageEncoder(nn.Module):

    def __init__(self, backbone_name: str='tiny', hidden_dim: int=256, pretrained: bool=True, freeze_backbone: bool=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone_name
        self.backbone, self.backbone_dim = self._load_backbone(backbone_name, pretrained)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.channel_projection = nn.Conv2d(in_channels=self.backbone_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.channel_projection.weight)
        nn.init.zeros_(self.channel_projection.bias)
        self.spatial_h = 8
        self.spatial_w = 8
        self.sequence_length = self.spatial_h * self.spatial_w

    def _load_custom_weights(self, model: nn.Module, weights_enum):
        package_root = Path(__file__).resolve().parent.parent
        save_dir = package_root / 'checkpoints' / 'copyright'
        save_dir.mkdir(parents=True, exist_ok=True)
        url = weights_enum.url
        filename = 'image_encoder_final.pth'
        file_path = save_dir / filename
        if not file_path.exists():
            print(f'Downloading pretrained weights to: {file_path}')
            torch.hub.download_url_to_file(url, str(file_path))
        else:
            print(f'Loading local pretrained weights: {file_path}')
        try:
            state_dict = torch.load(file_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]
                if name.startswith('backbone.'):
                    name = name[9:]
                new_state_dict[name] = v
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f'Weights loaded. (Missing: {len(missing)}, Unexpected: {len(unexpected)})')
        except Exception as e:
            print(f'[ERROR] Weight loading failed: {e}')
            print('Falling back to random initialization.')

    def _load_backbone(self, backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        backbone_name = backbone_name.lower()
        model = None
        backbone_dim = 0
        weights_enum = None
        if backbone_name == 'tiny':
            model = models.convnext_tiny(weights=None)
            weights_enum = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            backbone_dim = 768
        elif backbone_name == 'small':
            model = models.convnext_small(weights=None)
            weights_enum = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            backbone_dim = 768
        elif backbone_name == 'base':
            model = models.convnext_base(weights=None)
            weights_enum = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            backbone_dim = 1024
        elif backbone_name == 'large':
            model = models.convnext_large(weights=None)
            weights_enum = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
            backbone_dim = 1536
        else:
            raise ValueError(f'Unknown backbone: {backbone_name}')
        if pretrained:
            try:
                self._load_custom_weights(model, weights_enum)
            except Exception as e:
                print(f'Warning: weight loading failed ({e}). Using random initialization.')
        backbone = model.features
        return (backbone, backbone_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        if images.dim() != 4:
            raise ValueError(f'Expected 4D input, got {images.dim()}D.')
        visual_features = self.backbone(images)
        projected_features = self.channel_projection(visual_features)
        projected_features = F.interpolate(projected_features, size=(8, 8), mode='bilinear', align_corners=False)
        flattened = projected_features.flatten(2)
        image_sequence = flattened.transpose(1, 2)
        return image_sequence

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def get_sequence_length(self) -> int:
        return self.sequence_length

class ImageEncoderFactory:

    @staticmethod
    def create(config: Optional[object]=None, backbone_name: str='tiny', hidden_dim: int=256, pretrained: bool=True, freeze_backbone: bool=False) -> ImageEncoder:
        if config is not None:
            backbone_name = getattr(config, 'IMAGE_BACKBONE', backbone_name)
            hidden_dim = getattr(config, 'HIDDEN_DIM', hidden_dim)
            pretrained = getattr(config, 'USE_PRETRAINED', pretrained)
            freeze_backbone = getattr(config, 'FREEZE_BACKBONE', freeze_backbone)
        return ImageEncoder(backbone_name=backbone_name, hidden_dim=hidden_dim, pretrained=pretrained, freeze_backbone=freeze_backbone)
