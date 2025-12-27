"""
Unified model loader function, integrates all models defined in MODEL_URLS.
Loads pretrained weights from the internet using URL.
"""
import torch
import torchvision.models as tvmodels
import torchvision.models as models
import re
import sys
import os

try:
    from .url import MODEL_URLS
    from .resnet import ResNet, Bottleneck
    from .inception_v3 import Inception3
    from .senet import SENet, SEBottleneck, pretrained_settings as senet_pretrained_settings
    from .pnasnet import PNASNet5Large, pretrained_settings as pnasnet_pretrained_settings
except ImportError:
    # If relative import fails (when running script directly), add current dir to path and use absolute import
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add current directory to sys.path to enable importing modules in same directory
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    # Now we can import the modules in the same directory
    from url import MODEL_URLS
    from resnet import ResNet, Bottleneck
    from inception_v3 import Inception3
    from senet import SENet, SEBottleneck, pretrained_settings as senet_pretrained_settings
    from pnasnet import PNASNet5Large, pretrained_settings as pnasnet_pretrained_settings

try:
    import timm
except ImportError:
    timm = None


def safe_load_state_dict_from_url(url, model_name=None, map_location='cpu', max_retries=3):
    """
    Safely load state_dict from URL, with error handling, retry mechanism, and progress display.
    If already downloaded locally, load from cache instead of downloading again.
    Can detect if local weight file is corrupted.

    Args:
        url: Model weights URL
        model_name: Model name (for error messages)
        map_location: Location to load the model
        max_retries: Maximum number of retries

    Returns:
        state_dict
    """
    import ssl
    import urllib.request
    from torch.hub import get_dir

    hub_dir = get_dir()
    filename = os.path.basename(url.split('?')[0])
    cache_file = os.path.join(hub_dir, "checkpoints", filename) if filename else None

    # Prefer local cache
    if cache_file and os.path.exists(cache_file):
        try:
            print(f"[Local Load] Loading model weights from cache: {cache_file}")
            state_dict = torch.load(cache_file, map_location=map_location)
            # Simple consistency check: must be a dict similar to state_dict or weights
            if not isinstance(state_dict, dict) or not any(k in state_dict for k in ('state_dict', 'model', 'model_state', 'module', 'weight', 'weights')):
                # Many common model weights are dicts, with param names and tensors
                num_tensor = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
                if num_tensor < 1:
                    raise RuntimeError("File is not a valid state_dict format")
            print(f"[Local Load] Successfully loaded model weights from cache: {cache_file}")
            return state_dict
        except Exception as e:
            print(f"[Warning] Failed loading from local cache, file may be corrupted: {e}, attempting cleanup and redownload.")
            try:
                os.remove(cache_file)
                print(f"[Cleanup] Removed corrupted cache file: {cache_file}")
            except Exception:
                pass
            # Continue to download from network

    # Download from network
    for attempt in range(max_retries):
        try:
            print(f"[Network Download] Attempting to download model weights: {url}")
            # torch.hub.load_state_dict_from_url does sha256 check and automatically clears corrupted cache files
            state_dict = torch.hub.load_state_dict_from_url(
                url,
                model_dir=os.path.join(hub_dir, "checkpoints"),
                map_location=map_location,
                weights_only=False
            )
            print(f"[Network Download] Model download and load complete!")
            return state_dict
        except Exception as e:
            error_msg = str(e)
            # SSL certificate issue
            if 'ssl' in error_msg.lower() or 'certificate' in error_msg.lower() or 'CERTIFICATE_VERIFY_FAILED' in error_msg:
                print(f"[Warning] SSL certificate verification failed (Attempt {attempt + 1}/{max_retries})")
                if model_name:
                    print(f"[Warning] Model: {model_name}, URL: {url}")
                print(f"[Warning] Trying download with SSL verification disabled (for expired certificate workaround only)...")

                try:
                    ssl_context = ssl._create_unverified_context()
                    import tempfile
                    import shutil
                    from tqdm import tqdm

                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_path = tmp_file.name

                    try:
                        req = urllib.request.Request(url)
                        with urllib.request.urlopen(req, context=ssl_context) as response:
                            total = response.length or response.headers.get("content-length")
                            if total is not None:
                                total = int(total)
                            else:
                                total = None
                            with open(tmp_path, 'wb') as f, tqdm(
                                total=total,
                                unit='B',
                                unit_scale=True,
                                desc=f"Download {model_name or os.path.basename(url)} (SSL Bypassed)",
                                ncols=80,
                                ascii=True
                            ) as t:
                                chunk_size = 8192
                                while True:
                                    chunk = response.read(chunk_size)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    t.update(len(chunk))
                        # Copy to cache_file
                        if cache_file:
                            shutil.copyfile(tmp_path, cache_file)
                        print(f"[Network Download-SSL Disabled] Model weights downloaded: {cache_file or tmp_path}")

                        # Check if newly downloaded file is corrupted
                        try:
                            state_dict = torch.load(tmp_path, map_location=map_location)
                            # Check content
                            if not isinstance(state_dict, dict) or not any(k in state_dict for k in ('state_dict', 'model', 'model_state', 'module', 'weight', 'weights')):
                                num_tensor = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
                                if num_tensor < 1:
                                    raise RuntimeError("File is not a valid state_dict format")
                        except Exception as err:
                            print(f"[Warning] Newly downloaded model weights file corrupted: {err}")
                            raise err

                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        print(f"[Success] Fallback method succeeded in loading model weights")
                        return state_dict
                    except Exception as download_error:
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                        raise download_error
                except Exception as ssl_fallback_error:
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Failed to load model weights (SSL certificate verification failed, tried {max_retries} times): {e}\n"
                            f"Fallback download method also failed: {ssl_fallback_error}\n"
                            f"Please check your network or download manually: {url}"
                        )
                    continue
            # Corrupt file errors
            elif 'unpickling' in error_msg.lower() or 'stack underflow' in error_msg.lower() or 'corrupt' in error_msg.lower():
                print(f"[Warning] Corrupted cache file detected (Attempt {attempt + 1}/{max_retries})")
                if model_name:
                    print(f"[Warning] Model: {model_name}, URL: {url}")
                try:
                    # Delete corrupted cache file
                    if cache_file and os.path.exists(cache_file):
                        print(f"[Cleanup] Removing corrupted cache file: {cache_file}")
                        try:
                            os.remove(cache_file)
                        except Exception:
                            pass
                    # Iterate cache directory for possible alias files
                    cache_dir = os.path.join(hub_dir, "checkpoints")
                    if os.path.exists(cache_dir):
                        for f in os.listdir(cache_dir):
                            if model_name and model_name in f:
                                cache_file2 = os.path.join(cache_dir, f)
                                print(f"[Cleanup] Removing corrupted cache file: {cache_file2}")
                                try:
                                    os.remove(cache_file2)
                                except Exception:
                                    pass
                except Exception as cleanup_error:
                    print(f"[Warning] Error encountered during cache cleanup: {cleanup_error}")
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load model weights (tried {max_retries} times): {e}\n"
                        f"Please check network connection or manually clean cache directory: {hub_dir}/checkpoints"
                    )
                continue
            else:
                # Other unknown errors, possibly due to corrupt cache files also
                if cache_file and os.path.exists(cache_file):
                    print(f"[Exception] Cache file may be corrupted, deleting and retrying: {cache_file}")
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass
                    continue
                raise
    raise RuntimeError(f"Failed to load model weights: {url}")



def load_resnet50(pretrained=False, **kwargs):
    """Load ResNet50 model"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['resnet50'], model_name='resnet50', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_resnet152(pretrained=False, **kwargs):
    """Load ResNet152 model"""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['resnet152'], model_name='resnet152', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_vgg19_bn(pretrained=False, **kwargs):
    """Load VGG19_BN model"""
    model = tvmodels.vgg19_bn(**kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['vgg19_bn'], model_name='vgg19_bn', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_inception_v3(pretrained=False, **kwargs):
    """Load Inception V3 model"""
    model = Inception3(**kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['inception_v3'], model_name='inception_v3', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_densenet121(pretrained=False, **kwargs):
    """Load DenseNet121 model"""
    model = tvmodels.densenet121(**kwargs)
    if pretrained:
        # DenseNet needs to specially handle the keys in state_dict
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['densenet121'], model_name='densenet121', map_location='cpu'
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def load_vit_b_16(pretrained=False, **kwargs):
    """Load ViT-B/16 model"""
    model = tvmodels.vit_b_16(**kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['vit_b_16'], model_name='vit_b_16', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_mobilenet_v2(pretrained=False, **kwargs):
    """Load MobileNet V2 model"""
    model = tvmodels.mobilenet_v2(**kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['mobilenet_v2'], model_name='mobilenet_v2', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_senet154(pretrained=True, num_classes=1000, dataset='imagenet', **kwargs):
    """Load SENet154 model"""
    def initialize_pretrained_model_from_url(ckpt_url, model, num_classes, settings):
        assert num_classes == settings['num_classes'], \
            'num_classes should be {}, but is {}'.format(
                settings['num_classes'], num_classes)
        model.load_state_dict(safe_load_state_dict_from_url(
            ckpt_url, model_name='senet154', map_location=torch.device('cpu')
        ))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    ckpt_url = MODEL_URLS['senet154']
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained and dataset is not None:
        settings = senet_pretrained_settings['senet154'][dataset]
        initialize_pretrained_model_from_url(ckpt_url, model, num_classes, settings)
    return model


def load_resnext101(pretrained=False, **kwargs):
    """Load ResNeXt101 model"""
    model = tvmodels.resnext101_32x8d(**kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['resnext101'], model_name='resnext101', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_wrn101(pretrained=False, **kwargs):
    """Load Wide ResNet101 model"""
    model = tvmodels.wide_resnet101_2(**kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['wrn101'], model_name='wrn101', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_pnasnet(pretrained=False, num_classes=1000, dataset='imagenet', **kwargs):
    """Load PNASNet model"""
    if dataset and pretrained:
        ckpt_url = MODEL_URLS['pnasnet']
        settings = pnasnet_pretrained_settings['pnasnet5large'][dataset]
        assert num_classes == settings[
            'num_classes'], 'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)

        # both 'imagenet' & 'imagenet+background' are loaded from same parameters
        model = PNASNet5Large(num_classes=1001)
        model.load_state_dict(safe_load_state_dict_from_url(
            ckpt_url, model_name='pnasnet', map_location='cpu'
        ))

        if dataset == 'imagenet':
            from torch import nn
            new_last_linear = nn.Linear(model.last_linear.in_features, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = PNASNet5Large(num_classes=num_classes)
    return model


def load_mnasnet(pretrained=False, **kwargs):
    """Load MNASNet model"""
    model = tvmodels.mnasnet1_0(**kwargs)
    if pretrained:
        state_dict = safe_load_state_dict_from_url(
            MODEL_URLS['mnasnet'], model_name='mnasnet', map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model


def load_swin_b(pretrained=False, **kwargs):
    """Load Swin-B model"""
    if timm is None:
        raise ImportError("timm is required for Swin-B model. Please install it with: pip install timm")
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    return model


def load_convnext_s(pretrained=False, **kwargs):
    """Load ConvNeXt-S model"""
    model = models.convnext_small(pretrained=False)
    if pretrained:
        checkpoint = safe_load_state_dict_from_url(
            MODEL_URLS['convnext_s'], model_name='convnext_s', map_location='cpu'
        )
        model.load_state_dict(checkpoint)
    return model


def load_convnext_b(pretrained=False, **kwargs):
    """Load ConvNeXt-B model"""
    if timm is None:
        raise ImportError("timm is required for ConvNeXt-B model. Please install it with: pip install timm")
    model = timm.create_model('convnext_base', pretrained=pretrained)
    return model

def load_convnext_t(pretrained=False, **kwargs):
    """Load ConvNeXt-B model"""
    if timm is None:
        raise ImportError("timm is required for ConvNeXt-B model. Please install it with: pip install timm")
    model = timm.create_model('convnext_base', pretrained=pretrained)
    return model



def load_adv_resnet50(pretrained=True, **kwargs):
    """Load adversarially trained ResNet50 model"""
    if not pretrained:
        raise ValueError("adv_resnet50 only supports pretrained=True")
    try:
        from robustbench.utils import load_model as robustbench_load_model
    except ImportError:
        raise ImportError("robustbench is required for adv_resnet50. Please install it with: pip install robustbench")
    return robustbench_load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model='Linf').model


def load_adv_wrn50(pretrained=True, **kwargs):
    """Load adversarially trained Wide ResNet50 model"""
    if not pretrained:
        raise ValueError("adv_wrn50 only supports pretrained=True")
    try:
        from robustbench.utils import load_model as robustbench_load_model
    except ImportError:
        raise ImportError("robustbench is required for adv_wrn50. Please install it with: pip install robustbench")
    return robustbench_load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf').model


def load_adv_swin_b(pretrained=True, **kwargs):
    """Load adversarially trained Swin-B model"""
    if not pretrained:
        raise ValueError("adv_swin_b only supports pretrained=True")
    try:
        from robustbench.utils import load_model as robustbench_load_model
    except ImportError:
        raise ImportError("robustbench is required for adv_swin_b. Please install it with: pip install robustbench")
    return robustbench_load_model(model_name='Liu2023Comprehensive_Swin-B', dataset='imagenet', threat_model='Linf').model


def load_adv_convnext_b(pretrained=True, **kwargs):
    """Load adversarially trained ConvNeXt-B model"""
    if not pretrained:
        raise ValueError("adv_convnext_b only supports pretrained=True")
    try:
        from robustbench.utils import load_model as robustbench_load_model
    except ImportError:
        raise ImportError("robustbench is required for adv_convnext_b. Please install it with: pip install robustbench")
    return robustbench_load_model(model_name='Liu2023Comprehensive_ConvNeXt-B', dataset='imagenet', threat_model='Linf').model


# Mapping from model name to loader function (only models in MODEL_URLS are included)
MODEL_LOADERS = {
    'resnet50': load_resnet50,
    'vgg19_bn': load_vgg19_bn,
    'inception_v3': load_inception_v3,
    'densenet121': load_densenet121,
    'vit_b_16': load_vit_b_16,
    'resnet152': load_resnet152,
    'mobilenet_v2': load_mobilenet_v2,
    'senet154': load_senet154,
    'resnext101': load_resnext101,
    'wrn101': load_wrn101,
    'pnasnet': load_pnasnet,
    'mnasnet': load_mnasnet,
    'swin_b': load_swin_b,
    'convnext_s': load_convnext_s,
    'convnext_b': load_convnext_b,
    # Adversarially trained models (require robustbench)
    # 'adv_resnet50': load_adv_resnet50,
    # 'adv_wrn50': load_adv_wrn50,
    # 'adv_swin_b': load_adv_swin_b,
    # 'adv_convnext_b': load_adv_convnext_b,
}


def load_model(model_name, pretrained=True, **kwargs):
    """
    Unified model loader function

    Args:
        model_name: Model name, must be a key in MODEL_URLS
        pretrained: Whether to load pretrained weights
        **kwargs: Other model-specific parameters (e.g., num_classes, dataset, etc.)

    Returns:
        Loaded model

    Example:
        >>> model = load_model('resnet50', pretrained=True)
        >>> model = load_model('vgg19_bn', pretrained=True)
        >>> model = load_model('senet154', pretrained=True, num_classes=1000, dataset='imagenet')
    """
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unknown model name: {model_name}. "
                         f"Available models: {list(MODEL_LOADERS.keys())}")

    loader = MODEL_LOADERS[model_name]
    model = loader(pretrained=pretrained, **kwargs)
    return model

def load_models(model_names, pretrained=True, **kwargs):
    """
    Batch load models

    Args:
        model_names: A list of model names
        pretrained: Whether to load pretrained weights
        **kwargs: Other model-specific parameters

    Returns:
        A dictionary mapping model name to loaded model
    """
    models = {}
    for model_name in model_names:
        models[model_name] = load_model(model_name, pretrained=pretrained, **kwargs)
    return models

__all__ = [
    'load_model',
    'load_resnet50', 'load_resnet152',
    'load_vgg19_bn',
    'load_inception_v3',
    'load_densenet121',
    'load_vit_b_16',
    'load_mobilenet_v2',
    'load_senet154',
    'load_resnext101',
    'load_wrn101',
    'load_pnasnet',
    'load_mnasnet',
    'load_swin_b',
    'load_convnext_s', 'load_convnext_b',
    'load_adv_resnet50', 'load_adv_wrn50', 'load_adv_swin_b', 'load_adv_convnext_b',
    'MODEL_LOADERS',
]

if __name__ == "__main__":
    # Test all models
    test_model_names = list(MODEL_LOADERS.keys())
    models_dict = load_models(test_model_names, pretrained=True)

    # Set input size for each model (only common ones listed; add specific ones if needed)
    model_inputs = {
        'resnet50': torch.randn(1, 3, 224, 224),
        'resnet152': torch.randn(1, 3, 224, 224),
        'vgg19_bn': torch.randn(1, 3, 224, 224),
        'inception_v3': torch.randn(1, 3, 299, 299),  # inception_v3 needs 299 input
        'densenet121': torch.randn(1, 3, 224, 224),
        'vit_b_16': torch.randn(1, 3, 224, 224),
        'mobilenet_v2': torch.randn(1, 3, 224, 224),
        'senet154': torch.randn(1, 3, 224, 224),
        'resnext101': torch.randn(1, 3, 224, 224),
        'wrn101': torch.randn(1, 3, 224, 224),
        'pnasnet': torch.randn(1, 3, 331, 331),  # some pnasnet implementations use 331
        'mnasnet': torch.randn(1, 3, 224, 224),
        'swin_b': torch.randn(1, 3, 224, 224),
        'convnext_s': torch.randn(1, 3, 224, 224),
        'convnext_b': torch.randn(1, 3, 224, 224),
        'adv_resnet50': torch.randn(1, 3, 224, 224),
        'adv_wrn50': torch.randn(1, 3, 224, 224),
        'adv_swin_b': torch.randn(1, 3, 224, 224),
        'adv_convnext_b': torch.randn(1, 3, 224, 224),
    }

    for name, model in models_dict.items():
        model.eval()
        input_data = model_inputs.get(name, torch.randn(1, 3, 224, 224))
        try:
            with torch.no_grad():
                output = model(input_data)
            print(f"Model '{name}' loaded: {type(model)}, output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        except Exception as e:
            print(f"Model '{name}' failed forward pass: {e}")
