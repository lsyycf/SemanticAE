import math
import random
from typing import Tuple, Callable

import numpy.random
import torch
import torchvision
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from resadv_ddim.model import StableDiffDDIMGeneratorDescriptor, resadv_ddim


def get_target_label(logits, label):  # seond-like label for attack
    rates, indices = logits.sort(1, descending=True)
    rates, indices = rates.squeeze(0), indices.squeeze(0)

    tar_label = torch.zeros_like(label).to(label.device)

    for i in range(label.shape[0]):
        if label[i] == indices[i][0]:  # classify is correct
            tar_label[i] = indices[i][1]
        else:
            tar_label[i] = indices[i][0]

    return tar_label


def get_target_label_stamo(logits, label):  # second-like label for attack
    if label.shape[1] > 1:
        return get_target_label_stamo_multilabel(logits, label)
    logits = logits.clone().detach()
    logits[torch.arange(label.shape[0], device=label.device), label.reshape(-1)] = - torch.inf
    rates, indices = logits.sort(1, descending=True)
    return indices[..., 0]


def get_target_label_stamo_multilabel(logits, label: torch.Tensor):  # seond-like label for attack
    logits = logits.clone().detach()
    logits[label] = - torch.inf
    rates, indices = logits.sort(1, descending=True)
    return indices[..., 0]


class MaskedStableDiffDDIMGeneratorDescriptor(StableDiffDDIMGeneratorDescriptor):
    def __init__(self, pipeline: StableDiffusionPipeline, guidance_scale=3.0, eta=0.0, guidance_scale_2=0.3,
                 edge_ratio=1 / 32, total_sampling_steps=100):
        super().__init__(pipeline, guidance_scale, eta, total_sampling_steps=total_sampling_steps)
        self.guidance_scale_2 = guidance_scale_2
        self.edge_ratio = edge_ratio

    def diffusion_step(self, latents, context, t, extra_step_kwargs):
        latent_input = torch.cat([latents] * 2)
        s: DDIMScheduler = self.pipeline.scheduler
        latent_input = s.scale_model_input(latent_input, t)

        noise_pred = self.pipeline.unet(latent_input, t, encoder_hidden_states=context, return_dict=False)[0]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        D = (noise_prediction_text - noise_pred_uncond)

        d_edge = math.ceil(latents.shape[-1] * self.edge_ratio)
        edit_mask = torch.zeros_like(latents, dtype=torch.bool)
        edit_mask[:, :, d_edge:-d_edge, d_edge:-d_edge] = 1

        noise_pred = noise_pred_uncond + torch.where(edit_mask, self.guidance_scale * D, self.guidance_scale_2 * D)
        latents, latents_0 = s.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)

        prev_step = t - s.config.num_train_timesteps // s.num_inference_steps
        # return (latents_0, prev_step) if prev_step <= 0 else (latents, prev_step)
        return latents, prev_step


from resadv_ddim.model import VictimDescriptor, StableDiffDDIMGeneratorDescriptor


class ClassificationDescriptorUntargeted(VictimDescriptor):
    def __init__(self, model, trans_kernel: Callable = torchvision.transforms.v2.Identity(), loss_kernel=None,
                 target_moving_avg=0.3):
        self.model = model
        self.trans_kernel = trans_kernel
        self.target_moving_avg = target_moving_avg
        self.loss_kernel = loss_kernel

    def get_target_label_stamo(self, logits, label):  # second-like label for attack
        if label.shape[1] > 1:
            return get_target_label_stamo_multilabel(logits, label)
        logits = logits.clone().detach()
        logits[torch.arange(label.shape[0], device=label.device), label.reshape(-1)] = - torch.inf
        rates, indices = logits.sort(1, descending=True)
        return indices[..., 0]

    def get_target_label_stamo_multilabel(self, logits, label: torch.Tensor):  # seond-like label for attack
        logits = logits.clone().detach()
        logits[label] = - torch.inf
        rates, indices = logits.sort(1, descending=True)
        return indices[..., 0]

    def get_results_loss(self, model_input, context: dict, result_differentiable=False, grad_differentiable=False,
                         update_context=False) -> Tuple[torch.Tensor, torch.Tensor]:
        label = context["target_label"]
        input_transformed = self.trans_kernel(model_input)
        logits = self.model(input_transformed)
        # logits_avg = context.get("second_like_label", logits.detach())
        #
        # target_label = self.get_target_label_stamo(logits_avg.detach(), label) # 2dn like target label
        # # if update_context:
        # logits_avg = logits_avg * (1 - self.target_moving_avg) + logits.detach() * self.target_moving_avg
        # context["second_like_label"] = logits_avg

        target_label = getattr(context,  "second_like_label", None)
        if target_label is None or update_context:
            target_label = get_target_label_stamo(logits, label)
            context["second_like_label"] = target_label
        if label.shape[1] > 1:
            loss_func = lambda logits_: F.log_softmax(logits_, dim=-1)[range(len(logits_)), target_label].mean() - \
                                        (F.log_softmax(logits_, dim=-1) * label.float() /
                                         label.float().sum(dim=-1, keepdim=True)).sum(dim=-1).mean()
        else:
            loss_func = lambda logits_: F.log_softmax(logits_, dim=-1)[range(len(logits_)), target_label].mean()  - \
                                        F.log_softmax(logits_, dim=-1)[
                                            torch.arange(len(logits_), device=logits_.device), label.reshape(-1)
                                        ].float().mean()
        loss = loss_func(logits) if self.loss_kernel is None else self.loss_kernel(input_transformed, self.model,
                                                                                   loss_func)
        return logits, loss

    def get_result(self, model_input):
        return self.model(model_input)

    def attack_judge(self, logits, context) -> torch.Tensor:
        label = context["target_label"]
        prob_cls = F.softmax(logits, dim=-1)
        orig_prob = (prob_cls * label).max() if label.shape[1] > 1 else prob_cls[..., label.reshape(-1)].mean()

        return orig_prob


def demo():

    model_id = "bguisard/stable-diffusion-nano-2-1"
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler/",
                                              local_files_only=True, # Turn on this if your network is not stable
                                              )

    diffusion = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to("cuda")
    scheduler.set_timesteps(100)
    # set all param in diffusion not requires grad.
    for param in diffusion.unet.parameters():
        param.requires_grad = False
    for param in diffusion.vae.parameters():
        param.requires_grad = False

    # build resnet50 from torchvision with input color norm
    model = torchvision.models.resnet50(pretrained=True).cuda()
    model.eval()  # Set the model to evaluation mode
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalize color channels
    ])

    # set rnd seed
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)
    latents_shape = (
        1,
        diffusion.unet.config.in_channels,
        128 // diffusion.vae_scale_factor,
        128 // diffusion.vae_scale_factor,
    )
    latents = torch.randn(
        latents_shape, device=diffusion.device)
    latents = latents * scheduler.init_noise_sigma

    # return latents
    def generate_context(model, text_label):
        max_length = 77
        uncond_input = model.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

        text_input = model.tokenizer(
            [text_label],
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

        context = [uncond_embeddings, text_embeddings]
        context = torch.cat(context)
        return context

    from surrogate_models.input_randomization import get_transformation
    loss_kernel = get_transformation("decowa")
    trans_kernel = lambda x: transform(x)
    victim_descriptor = ClassificationDescriptorUntargeted(model, trans_kernel=trans_kernel, loss_kernel=loss_kernel)
    # generation_descriptor = StableDiffDDIMGeneratorDescriptor(diffusion)
    generation_descriptor = MaskedStableDiffDDIMGeneratorDescriptor(diffusion, guidance_scale=3.0, eta=0.0,
                                                                    guidance_scale_2=0.3, edge_ratio=1 / 16)
    latents_adv, latents_exemplar = resadv_ddim(latents, generate_context(diffusion, "Jellyfish"),
                                                torch.tensor([[107]]), generation_descriptor, victim_descriptor,
                                                residual_approx_step = 3,
                                                step_size = 0.7, budget_all=2.5, verbose=True)

    image = generation_descriptor.decode(latents_adv)
    image_clean = generation_descriptor.decode(latents_exemplar)
    torch.cuda.empty_cache()
    # save torch tensor image
    torchvision.utils.save_image(image, "test.png")
    torchvision.utils.save_image(image_clean, "test_clean.png")
    context = {"target_label": torch.tensor([[107]])}
    print(f"clean confidence:{float(victim_descriptor.attack_judge(victim_descriptor.get_results_loss(image_clean, context)[0], context))}")
    print(f"adv. confidence:{float(victim_descriptor.attack_judge(victim_descriptor.get_results_loss(image, context)[0], context))}")




if __name__ == '__main__':
    demo()