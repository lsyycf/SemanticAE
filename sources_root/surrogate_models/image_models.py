

import torch
from omegaconf import OmegaConf
from torch.nn import DataParallel

from .input_randomization import InputRandomizationKernel, DeCowA, BSR, Identity

from surrogate_models import model_loader
__path_extended__ = False


class LogitsVoting(torch.nn.ModuleList):
    def __init__(self, type, *models):
        super().__init__(*[models])
        self.type = type

    def forward(self, x):
        if self.type == "mean":
            logits = torch.stack([model(x) for model in self], dim=1)
            return logits.mean(dim=1)
        else:
            # TODO
            logits = torch.stack([model(x) for model in self], dim=1)
            return logits.mean(dim=1)





def get_target_model_blackbox(target_model_config):
    target = model_loader.load_surrogate_models(target_model_config.models)
    target = [t.module if isinstance(t, DataParallel) else t for t in target]
    if hasattr(target_model_config, "fuse"):
        target = LogitsVoting("mean", *target)

    return target



def get_surrogate_model_blackbox(surrogate_config):



    target = model_loader.load_surrogate_models(surrogate_config.models)
    target = [t.module if isinstance(t, DataParallel) else t for t in target]
    if hasattr(surrogate_config, "fuse"):
        target = LogitsVoting("mean", *target)
    target.eval()
    target.requires_grad_(False)
    return target



def get_transformation(type, **kwargs) -> InputRandomizationKernel:
    if type == "decowa":
        return DeCowA(**kwargs)
    elif type == "bsr":
        return BSR(**kwargs)
    elif type == "none":
        return Identity()




import torch.nn.functional as F
def forward_template_targeted(image, vic_model, target_label, trans_kernel = None):
    loss_func = lambda logits_: F.log_softmax(logits_, dim=-1)[range(len(logits_)), target_label].sum()
    loss = trans_kernel(image, vic_model, loss_func) if trans_kernel is not None else loss_func(vic_model(image))

    return loss


def get_target_label_stamo(logits, label): # second-like label for attack

    rates, indices = logits.sort(1, descending=True)
    #rates, indices = rates.squeeze(0), indices.squeeze(0)

    tar_label = torch.zeros_like(torch.tensor(label)).to(logits.device)

    for i in range(logits.shape[0]):
        if label[i] == indices[i][0]:  # classify is correct
            tar_label[i] = indices[i][1]
        else:
            tar_label[i] = indices[i][0]

    return tar_label

def forward_template_autoselect_target(image, vic_model, evasion_label, trans_kernel = None):
    logits = vic_model(image)
    target_label = get_target_label_stamo(logits, evasion_label)
    loss_func = lambda logits_: F.log_softmax(logits_, dim=-1)[range(len(logits_)), target_label].sum()
    loss = trans_kernel(image, vic_model, loss_func) if trans_kernel is not None else loss_func(vic_model(image))
    return loss


def test_model_selection_blackbox():
    surrogate_config = OmegaConf.create({
        "source_model_path": ["resnet50"],
        "fuse": True # TODO: add fuse type
    })
    target_config = OmegaConf.create({
        "target_model_path": ["resnet50"],
        "fuse": True # TODO: add fuse type
    })
    input_transformation = OmegaConf.create({
        "type": "decowa",
        "num_warping": 3
    })


    victim_model = get_surrogate_model_blackbox(surrogate_config) # default is cuda
    target_model = get_target_model_blackbox(target_config)  # default is cuda

    # For Training / Optimizing Adversarial Examples
    image = torch.randn(2, 3, 224, 224).cuda()
    image.requires_grad = True
    input_transformation = get_transformation(**input_transformation)
    loss = forward_template_autoselect_target(image, victim_model, [1, 2], input_transformation)


    print(loss)
    loss.backward()
    assert image.grad.sum().abs() > 0.0001

    # For Evaluation
    print(target_model(image).shape)


