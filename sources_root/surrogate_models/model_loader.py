import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as tvmodels
import timm

__all__ = ['load_surrogate_model', 'add_normalize_layer', 'add_resize_layer']


def _get_weights_param(pretrained):
    """Convert pretrained boolean to weights parameter for torchvision models."""
    if pretrained:
        return 'DEFAULT'
    else:
        return None


IMAGENET_MODEL_NAMES = [
    'resnet18', 'resnet34', 'resnet50', 'resnet152',
    'vgg11_bn', 'vgg19', 'vgg19_bn',
    'inception_v3',
    'densenet121',
    'mobilenet_v2', 'mobilenet_v3',
    'senet154',
    'resnext101',
    'wrn50', 'wrn101',
    'pnasnet', 'mnasnet',
    'convnext_b', 'convnext_l', 'convnext_t',
    'swin_b', 'swin_s', 'swin_t',
    'vit_b_16', 'vit_b_32', 'vit_l_16'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_normalize_layer(model, mean, std):
    return nn.Sequential(
        transforms.Normalize(mean=mean, std=std),
        model
    )


def add_resize_layer(model, size, **kwargs):
    return nn.Sequential(
        transforms.Resize(size=size, **kwargs),
        model
    )


def load_resnet_model(model_name, pretrained=True, **kwargs):
    model_map = {
        'resnet18': tvmodels.resnet18,
        'resnet34': tvmodels.resnet34,
        'resnet50': tvmodels.resnet50,
        'resnet152': tvmodels.resnet152,
    }
    if model_name not in model_map:
        raise ValueError(f'Unsupported ResNet model: {model_name}')
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return model_map[model_name](weights=weights, **kwargs)


def load_vgg_model(model_name, pretrained=True, **kwargs):
    model_map = {
        'vgg11_bn': tvmodels.vgg11_bn,
        'vgg19': tvmodels.vgg19,
        'vgg19_bn': tvmodels.vgg19_bn,
    }
    if model_name not in model_map:
        raise ValueError(f'Unsupported VGG model: {model_name}')
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return model_map[model_name](weights=weights, **kwargs)


def load_inception_v3(pretrained=True, **kwargs):
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return tvmodels.inception_v3(weights=weights, aux_logits=True, transform_input=False, init_weights=False, **kwargs)


def load_densenet_model(model_name, pretrained=True, **kwargs):
    model_map = {
        'densenet121': tvmodels.densenet121,
    }
    if model_name not in model_map:
        raise ValueError(f'Unsupported DenseNet model: {model_name}')
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return model_map[model_name](weights=weights, **kwargs)


def load_mobilenet_model(model_name, pretrained=True, **kwargs):
    model_map = {
        'mobilenet_v2': tvmodels.mobilenet_v2,
        'mobilenet_v3': tvmodels.mobilenet_v3_small,
    }
    if model_name not in model_map:
        raise ValueError(f'Unsupported MobileNet model: {model_name}')
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return model_map[model_name](weights=weights, **kwargs)


def load_resnext101(pretrained=True, **kwargs):
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return tvmodels.resnext101_32x8d(weights=weights, **kwargs)


def load_wrn_model(model_name, pretrained=True, **kwargs):
    model_map = {
        'wrn50': tvmodels.wide_resnet50_2,
        'wrn101': tvmodels.wide_resnet101_2,
    }
    if model_name not in model_map:
        raise ValueError(f'Unsupported Wide ResNet model: {model_name}')
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return model_map[model_name](weights=weights, **kwargs)


def load_mnasnet(pretrained=True, **kwargs):
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return tvmodels.mnasnet1_0(weights=weights, **kwargs)


def load_senet154(pretrained=True, **kwargs):
    try:
        return timm.create_model('senet154', pretrained=pretrained, **kwargs)
    except Exception as e:
        error_msg = str(e)
        if 'huggingface_hub' in error_msg.lower() or 'charset_normalizer' in error_msg.lower():
            raise ValueError(
                f'Failed to load SENet-154 model: huggingface_hub dependency issue. Possibly charset_normalizer version conflict. Try running: pip install --upgrade charset-normalizer huggingface_hub')
        else:
            raise ValueError(
                f'Failed to load SENet-154 model: {e}. Please ensure timm and huggingface_hub libraries are installed.')


def load_pnasnet(pretrained=True, **kwargs):
    try:
        return timm.create_model('pnasnet5large', pretrained=pretrained, **kwargs)
    except Exception as e:
        error_msg = str(e)
        if 'huggingface_hub' in error_msg.lower() or 'charset_normalizer' in error_msg.lower():
            raise ValueError(
                f'Failed to load PNASNet model: huggingface_hub dependency issue. Possibly charset_normalizer version conflict. Try running: pip install --upgrade charset-normalizer huggingface_hub')
        else:
            raise ValueError(
                f'Failed to load PNASNet model: {e}. Please ensure timm and huggingface_hub libraries are installed.')


def load_convnext_model(model_name, pretrained=True, **kwargs):
    if model_name == 'convnext_b':
        try:
            return timm.create_model('convnext_base', pretrained=pretrained, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if 'huggingface_hub' in error_msg.lower() or 'charset_normalizer' in error_msg.lower():
                raise ValueError(
                    f'Failed to load ConvNeXt-B model: huggingface_hub dependency issue. Possibly charset_normalizer version conflict. Try running: pip install --upgrade charset-normalizer huggingface_hub')
            else:
                raise ValueError(
                    f'Failed to load ConvNeXt-B model: {e}. Please ensure timm and huggingface_hub libraries are installed.')
    elif model_name == 'convnext_l':
        try:
            return timm.create_model('convnext_large', pretrained=pretrained, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if 'huggingface_hub' in error_msg.lower() or 'charset_normalizer' in error_msg.lower():
                raise ValueError(
                    f'Failed to load ConvNeXt-L model: huggingface_hub dependency issue. Possibly charset_normalizer version conflict. Try running: pip install --upgrade charset-normalizer huggingface_hub')
            else:
                raise ValueError(
                    f'Failed to load ConvNeXt-L model: {e}. Please ensure timm and huggingface_hub libraries are installed.')
    elif model_name == 'convnext_t':
        weights = _get_weights_param(pretrained)
        if 'pretrained' in kwargs:
            kwargs.pop('pretrained')
        return tvmodels.convnext_small(weights=weights, **kwargs)
    else:
        raise ValueError(f'Unsupported ConvNeXt model: {model_name}')


def load_swin_model(model_name, pretrained=True, **kwargs):
    model_map = {
        'swin_b': tvmodels.swin_b,
        'swin_s': tvmodels.swin_s,
        'swin_t': tvmodels.swin_t,
    }
    if model_name not in model_map:
        raise ValueError(f'Unsupported Swin model: {model_name}')
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return model_map[model_name](weights=weights, **kwargs)


def load_vit_model(model_name, pretrained=True, **kwargs):
    model_map = {
        'vit_b_16': tvmodels.vit_b_16,
        'vit_b_32': tvmodels.vit_b_32,
        'vit_l_16': tvmodels.vit_l_16,
    }
    if model_name not in model_map:
        raise ValueError(f'Unsupported ViT model: {model_name}')
    weights = _get_weights_param(pretrained)
    if 'pretrained' in kwargs:
        kwargs.pop('pretrained')
    return model_map[model_name](weights=weights, **kwargs)


def load_surrogate_model(model_name, pretrained=True, norm_layer=True, parallel=True,
                         require_grad=False, device=None, use_NIPS17=True):
    if device is None:
        device = DEVICE

    if use_NIPS17:
        from .models_blackboxbench.load_model import load_model
        model = load_model(model_name, pretrained=pretrained)
    elif model_name.startswith('resnet'):
        model = load_resnet_model(model_name, pretrained=pretrained)
    elif model_name.startswith('vgg'):
        model = load_vgg_model(model_name, pretrained=pretrained)
    elif model_name == 'inception_v3':
        model = load_inception_v3(pretrained=pretrained)
    elif model_name.startswith('densenet'):
        model = load_densenet_model(model_name, pretrained=pretrained)
    elif model_name.startswith('mobilenet'):
        model = load_mobilenet_model(model_name, pretrained=pretrained)
    elif model_name == 'resnext101':
        model = load_resnext101(pretrained=pretrained)
    elif model_name.startswith('wrn'):
        model = load_wrn_model(model_name, pretrained=pretrained)
    elif model_name == 'mnasnet':
        model = load_mnasnet(pretrained=pretrained)
    elif model_name.startswith('convnext'):
        model = load_convnext_model(model_name, pretrained=pretrained)
    elif model_name.startswith('swin'):
        model = load_swin_model(model_name, pretrained=pretrained)
    elif model_name.startswith('vit'):
        model = load_vit_model(model_name, pretrained=pretrained)
    elif model_name == 'senet154':
        model = load_senet154(pretrained=pretrained)
    elif model_name == 'pnasnet':
        model = load_pnasnet(pretrained=pretrained)
    else:
        raise ValueError(f'Unsupported model: {model_name}. Supported models: {IMAGENET_MODEL_NAMES}')

    if norm_layer:
        if 'inception' in model_name:
            model = add_normalize_layer(model=model, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            model = add_resize_layer(model=model, size=(299, 299))
        elif 'pnasnet' in model_name:
            model = add_normalize_layer(model=model, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            model = add_resize_layer(model=model, size=(331, 331))
        else:
            model = add_normalize_layer(model=model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            model = add_resize_layer(model=model, size=(224, 224))

    model.to(device)

    if parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()

    if not require_grad:
        for param in model.parameters():
            param.requires_grad = False

    return model


def load_surrogate_models(model_names, pretrained=True, norm_layer=True, parallel=True,
                          require_grad=False, device=None, skip_errors=False):
    """
    Batch load multiple surrogate models

    Args:
        model_names: List of model names
        pretrained: Whether to load pre-trained weights
        norm_layer: Whether to add normalization layer
        parallel: Whether to use DataParallel
        require_grad: Whether gradients are needed
        device: Device to use
        skip_errors: If True, skip models that fail to load and continue loading others

    Returns:
        If skip_errors=True, returns (models, loaded_model_names) tuple
        Otherwise returns models list
    """
    models = []
    loaded_model_names = []
    failed_models = []
    for model_name in model_names:
        try:
            model = load_surrogate_model(
                model_name=model_name,
                pretrained=pretrained,
                norm_layer=norm_layer,
                parallel=parallel,
                require_grad=require_grad,
                device=device
            )
            models.append(model)
            loaded_model_names.append(model_name)
            print(f"Model {model_name} loaded successfully (downloaded pre-trained weights from internet)")
        except Exception as e:
            if skip_errors:
                print(f"Warning: Failed to load model {model_name}: {e}")
                failed_models.append(model_name)
            else:
                raise
    if skip_errors and failed_models:
        print(f"\nSkipped {len(failed_models)} models that failed to load: {', '.join(failed_models)}")

    if skip_errors:
        return models, loaded_model_names
    else:
        return models


if __name__ == '__main__':
    print("Testing surrogate model loading...")
    # model_names = ['convnext_b', 'senet154']
    # models = load_surrogate_models(model_names, pretrained=True)
    # print(f"Loaded {len(models)} models in total")

    # Load all available models, skipping those that cannot be loaded
    model_names = IMAGENET_MODEL_NAMES
    models, loaded_names = load_surrogate_models(model_names, pretrained=True, skip_errors=True)
    print(f"\nSuccessfully loaded {len(models)} models in total")
    for model_name, model in zip(loaded_names, models):
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{model_name} parameter count: {param_count:,}")
