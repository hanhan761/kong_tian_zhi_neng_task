"""Published segmentation architectures with explicit, auditable initialization."""
from pathlib import Path
import importlib.util
import re
import sys
import torch
import segmentation_models_pytorch as smp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'course-source'))
from network import modeling as course_models

WEIGHTS = Path('/data/aerospace-intelligence/cache/pretrained')
FILES = {
    'mobile': 'mobilenet_v2-b0353104.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet34': 'resnet34-b627a593.pth',
    'resnet18': 'resnet18-f37072fd.pth',
    'mit_b0': 'mit_b0.pth',
    'mit_b2': 'mit_b2.pth',
}


def load_backbone(module, key, course_mobile=False):
    state = torch.load(WEIGHTS / FILES[key], map_location='cpu', weights_only=True)
    if isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
        state = state['state_dict']
    if course_mobile:
        remapped = {}
        for name, value in state.items():
            if not name.startswith('features.'):
                continue
            fields = name.split('.')
            idx = int(fields[1])
            if idx < 4:
                remapped['low_level_features.' + '.'.join(fields[1:])] = value
            elif idx < 18:
                remapped['high_level_features.' + '.'.join(fields[1:])] = value
        state = remapped
    target = module.state_dict()
    matching = {k: v for k,v in state.items() if k in target and hasattr(v, 'shape') and v.shape == target[k].shape}
    total = sum(v.numel() for v in target.values())
    loaded = sum(target[k].numel() for k in matching)
    coverage = loaded / total if total else 0
    if coverage < 0.95:
        raise RuntimeError('insufficient pretrained coverage for %s: %.4f' % (key, coverage))
    torch.nn.Module.load_state_dict(module, matching, strict=False)
    return coverage


def load_torchvision_convnext(module):
    """Map official torchvision ConvNeXt-Tiny backbone to timm's feature encoder."""
    state = torch.load(WEIGHTS / 'convnext_tiny-983f1562.pth', map_location='cpu', weights_only=True)
    target = module.state_dict()
    mapped = {}
    for key, value in state.items():
        parts = key.split('.')
        if len(parts) < 4 or parts[0] != 'features':
            continue
        stage = int(parts[1])
        if stage == 0:
            name = 'model.stem_' + parts[2] + '.' + '.'.join(parts[3:])
        elif stage in (2, 4, 6):
            name = 'model.stages_' + str(stage // 2) + '.downsample.' + '.'.join(parts[2:])
        elif stage in (1, 3, 5, 7):
            prefix = 'model.stages_' + str(stage // 2) + '.blocks.' + parts[2] + '.'
            if parts[3] == 'layer_scale':
                name = prefix + 'gamma'
            elif parts[3] == 'block' and len(parts) >= 6:
                layer = {'0': 'conv_dw', '2': 'norm', '3': 'mlp.fc1', '5': 'mlp.fc2'}.get(parts[4])
                if layer is None:
                    continue
                name = prefix + layer + '.' + '.'.join(parts[5:])
            else:
                continue
        else:
            continue
        if name in target and target[name].numel() == value.numel():
            mapped[name] = value.reshape(target[name].shape)
    total = sum(v.numel() for v in target.values())
    coverage = sum(target[k].numel() for k in mapped) / total if total else 0.
    if coverage < .99:
        raise RuntimeError('insufficient ConvNeXt backbone weight coverage: %.5f' % coverage)
    torch.nn.Module.load_state_dict(module, mapped, strict=False)
    return coverage


class DinoSmallSeg(torch.nn.Module):
    """DINOv2 ViT-S/14 with four intermediate feature projections."""
    def __init__(self, pretrained=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model('vit_small_patch14_dinov2', pretrained=False, img_size=518, num_classes=0)
        for parameter in self.backbone.parameters(): parameter.requires_grad_(False)
        for block in self.backbone.blocks[-2:]:
            for parameter in block.parameters(): parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters(): parameter.requires_grad_(True)
        self.projections = torch.nn.ModuleList([torch.nn.Conv2d(384, 128, 1) for _ in range(4)])
        self.head = torch.nn.Sequential(torch.nn.Conv2d(512, 128, 3, padding=1), torch.nn.BatchNorm2d(128),
                                        torch.nn.GELU(), torch.nn.Conv2d(128, 8, 1))
        self.coverage = None
        if pretrained:
            state = torch.load(WEIGHTS / 'dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
            target = self.backbone.state_dict()
            match = {k: v for k,v in state.items() if k in target and v.shape == target[k].shape}
            self.coverage = sum(target[k].numel() for k in match) / sum(v.numel() for v in target.values())
            if self.coverage < .99: raise RuntimeError('DINOv2 backbone coverage insufficient')
            self.backbone.load_state_dict(match, strict=True)
    def forward(self, x):
        if x.shape[-2:] != (518, 518): raise ValueError('DINOv2 expects 518 x 518 native tiles')
        features = self.backbone.forward_intermediates(x, indices=[2,5,8,11], norm=True, intermediates_only=True)
        y = torch.cat([layer(feature) for layer,feature in zip(self.projections, features)], dim=1)
        return torch.nn.functional.interpolate(self.head(y), size=x.shape[-2:], mode='bilinear', align_corners=False)


class DinoBaseSeg(torch.nn.Module):
    """DINOv2 ViT-B/14 with four intermediate feature projections."""
    def __init__(self, pretrained=False):
        super().__init__()
        import timm
        self.backbone=timm.create_model('vit_base_patch14_dinov2',pretrained=False,img_size=518,num_classes=0)
        for parameter in self.backbone.parameters(): parameter.requires_grad_(False)
        for block in self.backbone.blocks[-2:]:
            for parameter in block.parameters(): parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters(): parameter.requires_grad_(True)
        self.projections=torch.nn.ModuleList([torch.nn.Conv2d(768,128,1) for _ in range(4)])
        self.head=torch.nn.Sequential(torch.nn.Conv2d(512,128,3,padding=1),torch.nn.BatchNorm2d(128),torch.nn.GELU(),torch.nn.Conv2d(128,8,1))
        self.coverage=None
        if pretrained:
            state=torch.load(WEIGHTS/'dinov2_vitb14_pretrain.pth',map_location='cpu',weights_only=True)
            target=self.backbone.state_dict();matched={k:v for k,v in state.items() if k in target and v.shape==target[k].shape}
            self.coverage=sum(target[k].numel() for k in matched)/sum(v.numel() for v in target.values())
            if self.coverage<.99:raise RuntimeError('DINOv2 B backbone coverage insufficient')
            self.backbone.load_state_dict(matched,strict=True)
    def forward(self,x):
        if x.shape[-2:]!=(518,518):raise ValueError('DINOv2 expects 518 x 518 native tiles')
        features=self.backbone.forward_intermediates(x,indices=[2,5,8,11],norm=True,intermediates_only=True)
        y=torch.cat([layer(feature) for layer,feature in zip(self.projections,features)],dim=1)
        return torch.nn.functional.interpolate(self.head(y),size=x.shape[-2:],mode='bilinear',align_corners=False)


class DinoLargeSeg(torch.nn.Module):
    """DINOv2 ViT-L/14, four intermediate projections and a compact head."""
    def __init__(self,pretrained=False):
        super().__init__()
        import timm
        self.backbone=timm.create_model('vit_large_patch14_dinov2',pretrained=False,img_size=518,num_classes=0)
        for parameter in self.backbone.parameters():parameter.requires_grad_(False)
        for block in self.backbone.blocks[-2:]:
            for parameter in block.parameters():parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters():parameter.requires_grad_(True)
        self.projections=torch.nn.ModuleList([torch.nn.Conv2d(1024,128,1) for _ in range(4)])
        self.head=torch.nn.Sequential(torch.nn.Conv2d(512,128,3,padding=1),torch.nn.BatchNorm2d(128),torch.nn.GELU(),torch.nn.Conv2d(128,8,1))
        self.coverage=None
        if pretrained:
            state=torch.load(WEIGHTS/'dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True)
            target=self.backbone.state_dict();matched={k:v for k,v in state.items() if k in target and v.shape==target[k].shape}
            self.coverage=sum(target[k].numel() for k in matched)/sum(v.numel() for v in target.values())
            if self.coverage<.99:raise RuntimeError('DINOv2 L backbone coverage insufficient')
            self.backbone.load_state_dict(matched,strict=True)
    def forward(self,x):
        if x.shape[-2:]!=(518,518):raise ValueError('DINOv2 L expects native 518 tiles')
        features=self.backbone.forward_intermediates(x,indices=[5,11,17,23],norm=True,intermediates_only=True)
        y=torch.cat([layer(feature) for layer,feature in zip(self.projections,features)],dim=1)
        return torch.nn.functional.interpolate(self.head(y),size=x.shape[-2:],mode='bilinear',align_corners=False)


class MaxVitSeg(torch.nn.Module):
    """U-Net with timm's ImageNet MaxViT-S encoder at native 512 resolution."""
    def __init__(self,pretrained=False):
        super().__init__()
        self.net=smp.Unet(encoder_name='tu-maxvit_small_tf_512',encoder_weights=None,classes=8)
        self.coverage=None
        if pretrained:
            state=torch.load(WEIGHTS/'maxvit_small_tf_512.pth',map_location='cpu',weights_only=True)
            if isinstance(state.get('state_dict'),dict):state=state['state_dict']
            target=self.net.encoder.state_dict();matched={}
            for k,v in state.items():
                name='model.'+re.sub(r'^stages\.(\d+)\.',r'stages_\1.',k)
                if name in target and v.shape==target[name].shape:matched[name]=v
            self.coverage=sum(target[k].numel() for k in matched)/sum(v.numel() for v in target.values())
            if self.coverage<.95:raise RuntimeError('MaxViT backbone coverage insufficient: %.5f'%self.coverage)
            self.net.encoder.load_state_dict(matched,strict=False)
    def forward(self,x):
        if x.shape[-2:]!=(512,512):raise ValueError('MaxViT expects native 512 tiles')
        return self.net(x)


def create_model(name, pretrained=True):
    if name == 'dinov2_large_seg':
        model=DinoLargeSeg(pretrained=pretrained)
        return model,model.coverage
    if name == 'unet_maxvit_small':
        model=MaxVitSeg(pretrained=pretrained)
        return model,model.coverage
    if name == 'dinov2_base_seg':
        model = DinoBaseSeg(pretrained=pretrained)
        return model, model.coverage
    if name == 'segformer_mitb2':
        model = smp.Segformer(encoder_name='mit_b2', encoder_weights=None, classes=8)
        return model, load_backbone(model.encoder, 'mit_b2') if pretrained else None
    if name == 'dinov2_small_seg':
        model = DinoSmallSeg(pretrained=pretrained)
        return model, model.coverage
    if name == 'course_mobile':
        model = course_models.deeplabv3plus_mobilenet(num_classes=8, output_stride=16, pretrained_backbone=False)
        backbone, key, remap = model.backbone, 'mobile', True
    elif name == 'course_resnet50':
        model = course_models.deeplabv3plus_resnet50(num_classes=8, output_stride=16, pretrained_backbone=False)
        backbone, key, remap = model.backbone, 'resnet50', False
    elif name == 'unet_resnet34':
        model = smp.Unet(encoder_name='resnet34', encoder_weights=None, classes=8)
        backbone, key, remap = model.encoder, 'resnet34', False
    elif name == 'fpn_resnet34':
        model = smp.FPN(encoder_name='resnet34', encoder_weights=None, classes=8)
        backbone, key, remap = model.encoder, 'resnet34', False
    elif name == 'segformer_mitb0':
        model = smp.Segformer(encoder_name='mit_b0', encoder_weights=None, classes=8)
        backbone, key, remap = model.encoder, 'mit_b0', False
    elif name == 'unetformer_resnet18':
        source = HERE / 'vendor/geoseg/UNetFormer.py'
        spec = importlib.util.spec_from_file_location('uavid_geoseg_unetformer', source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.UNetFormer(backbone_name='resnet18', pretrained=False, num_classes=8)
        backbone, key, remap = model.backbone, 'resnet18', False
    elif name == 'expert_mobile_unet':
        model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights=None, classes=8,
                         decoder_channels=(128,64,32,16,8))
        backbone, key, remap = model.encoder, 'mobile', False
    elif name == 'expert_person':
        model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights=None, classes=2,
                         decoder_channels=(128,64,32,16,8))
        backbone, key, remap = model.encoder, 'mobile', False
    elif name == 'sfanet_resnet18':
        source = HERE / 'vendor/sfanet/SFANet.py'
        spec = importlib.util.spec_from_file_location('uavid_sfanet', source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.SFANet(backbone_name='resnet18', pretrained=False, num_classes=8)
        backbone, key, remap = model.backbone, 'resnet18', False
    elif name == 'unet_convnext_tiny':
        model = smp.Unet(encoder_name='tu-convnext_tiny', encoder_weights=None, classes=8)
        coverage = load_torchvision_convnext(model.encoder) if pretrained else None
        return model, coverage
    else:
        raise ValueError('unknown model: ' + name)
    coverage = load_backbone(backbone, key, remap) if pretrained else None
    return model, coverage


def forward(model, x):
    output = model(x)
    return output if isinstance(output, tuple) else (output, None)
