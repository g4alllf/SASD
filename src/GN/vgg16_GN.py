from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

prob = 0.006

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.origin_weights = {} # GhostNet
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # GhostNet
    def reset_GN(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                module.weight = nn.Parameter(self.origin_weights[name].clone().detach().to(module.weight.device))
                module.weight.requires_grad_(False)
                F.dropout(module.weight, p=prob, inplace=True)
                module.weight.requires_grad_(True)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}

def ghost_vgg(cfg: str, batch_norm: bool, weights, progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            kwargs["num_classes"] = len(weights.meta["categories"])
    else:
        raise ValueError("weights is None")
    
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), dropout=0.5, **kwargs)
    model.load_state_dict(weights.get_state_dict(progress=progress))
    # GhostNet
    model.origin_weights = {name : module.weight.clone().detach() for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.requires_grad_(False)
            F.dropout(module.weight, p=prob, inplace=True)
            module.weight.requires_grad_(True)
    return model

def ghost_VGG16(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:

    return ghost_vgg("D", False, weights, progress, **kwargs)

def ghost_VGG16_BN(*, weights = None, progress: bool = True, **kwargs: Any) -> VGG:

    return ghost_vgg("D", True, weights, progress, **kwargs)
