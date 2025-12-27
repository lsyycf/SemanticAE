import math
from abc import ABC, abstractmethod
from typing import Tuple, Callable

import torchvision.transforms.v2
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
from tqdm import tqdm

import torch
import torch.nn.functional as F


def diffusion_step_masked(model, latents, context, t, guidance_scale, guidance_scale_2, edit_mask, extra_step_kwargs):
    latent_input = torch.cat([latents] * 2)
    latent_input = model.scheduler.scale_model_input(latent_input, t)

    noise_pred = model.unet(latent_input, t, encoder_hidden_states=context, return_dict=False)[0]

    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    D = (noise_prediction_text - noise_pred_uncond)
    noise_pred = noise_pred_uncond + torch.where(edit_mask, guidance_scale * D, guidance_scale_2 * D)
    latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    return latents



class VictimDescriptor(ABC):
    @abstractmethod
    def get_results_loss(self, model_input, context: dict, result_differentiable=False, loss_differentiable=False,
                         update_context=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return result (model output) and loss (Typical Attack Loss)
        context: dictionary data describing the historical direction
        @param result_differentiable: whether result is differentiable on model_input
        @param grad_differentiable: whether loss is differentiable on model_input
        """
        pass

    @abstractmethod
    def get_result(self, model_input):
        """
        return results
        """
        pass

    def __call__(self, data):
        return self.get_result(data)

    @abstractmethod
    def attack_judge(self, results, context) -> torch.Tensor:
        """
        return the attack success judge of the results.
        return: ASR, float tensor [B, ] \in [0.0, 1.0]
        """
        pass




class GeneratorDescriptor(ABC):
    @abstractmethod
    def original_step(self, latents, cur_step, guidance_context, mutable_contexts=None) -> Tuple[torch.Tensor, int]:
        """
        @param: from N to 0
        @returns the tuple (next step latents, remaining step num)
        """
        pass

    @abstractmethod
    def sketch_step(self, latents, cur_step, jump_count, guidance_context, mutable_contexts=None,
                    differentiable=True) -> Tuple[torch.Tensor, int]:
        """
        could also be implemented as speculative decoding
        cur_step: from N to 0
        @param jump_count
        @returns the tuple (next step latents, remaining step num)
        """
        pass

    @abstractmethod
    def decode(self, latents, differentiable=True) -> torch.Tensor:
        pass

    @abstractmethod
    def get_total_training_step(self):
        pass


class StableDiffDDIMGeneratorDescriptor(GeneratorDescriptor):
    def __init__(self, pipeline: StableDiffusionPipeline, guidance_scale=3.0, eta=0.0, total_sampling_steps = 100):
        self.pipeline = pipeline
        self.guidance_scale = guidance_scale
        self.pipeline.scheduler.set_timesteps(total_sampling_steps)
        self.extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(None, eta)

    def diffusion_step(self, latents, context, t,
                       extra_step_kwargs):
        latent_input = torch.cat([latents] * 2)
        s: DDIMScheduler = self.pipeline.scheduler
        latent_input = s.scale_model_input(latent_input, t)
        noise_pred = self.pipeline.unet(latent_input, t, encoder_hidden_states=context, return_dict=False)[0]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents, latents_0 = s.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)
        prev_step = t - s.config.num_train_timesteps // s.num_inference_steps
        # return (latents_0, prev_step) if prev_step <= 0 else (latents, prev_step)
        return latents, prev_step

    def original_step(self, latents, cur_step, guidance_context, mutable_contexts=None) -> Tuple[torch.Tensor, int]:
        latents, prev_step = self.diffusion_step(latents, guidance_context, cur_step, self.extra_step_kwargs)
        return latents, prev_step

    def sketch_step(self, latents, cur_step, jump_count, guidance_context, mutable_contexts=None,
                    differentiable=True) -> Tuple[torch.Tensor, int]:
        s: DDIMScheduler = self.pipeline.scheduler
        # hack DDIM Scheduler
        a, b = s.config.num_train_timesteps, s.num_inference_steps
        s.config.num_train_timesteps, s.num_inference_steps = jump_count * 2, 2
        latents, prev_step = self.diffusion_step(latents, guidance_context, cur_step, self.extra_step_kwargs)
        s.config.num_train_timesteps, s.num_inference_steps = a, b
        # print(f"sketch step: {cur_step},  {prev_step}")

        return latents, prev_step

    def decode(self, latents, differentiable=True) -> torch.Tensor:
        # differentiable version of self.pipeline.decode_latents(latents)
        latents = latents / self.pipeline.vae.config.scaling_factor
        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        image = self.pipeline.image_processor.postprocess(image, output_type='pt')
        return image

    def get_total_training_step(self):
        s: DDIMScheduler = self.pipeline.scheduler
        # assert s.num_inference_steps is not None, "num_inference_steps must be set with num_inference_steps"
        return s.config.num_train_timesteps




def resadv_ddim(latents, guidance_context, evasion_labbel, generation_descriptor: GeneratorDescriptor,
                victim_descriptor: VictimDescriptor,
                budget_all=3.0, residual_approx_step=4, timestep_add_perturb=0.75,
                optim_iters_intense = 10, optim_iters_normal = 3,
                step_size: float = 1.0, beta=0.5,
                threshold_adv_optim_conf_xi_1=0.1, threshold_adv_optim_conf_xi_2=0.01,
                verbose=True):

    next_intense_optim_step = math.inf
    total_steps = generation_descriptor.get_total_training_step()
    remaining_steps = total_steps - 1
    EPS = budget_all
    asr_judge = 1.0
    latents = latents.clone()
    latents_adv = latents.clone()
    v_inner = None

    first_step = True
    context_victim = {"target_label": evasion_labbel}

    while remaining_steps > 0:
        # Jump interval for sketch step computation. modify this for continuous t
        jump_step = math.ceil(remaining_steps / math.ceil(1.0 * residual_approx_step * remaining_steps / total_steps))
        index = remaining_steps
        # print(index)
        n_iter = 0
        # Adaptive Optimization Mechanism
        optim_iters = optim_iters_normal
        if index < next_intense_optim_step and index <= total_steps * timestep_add_perturb:
            next_intense_optim_step = total_steps * 0.05
            optim_iters = optim_iters_intense
        threshold_adv_optim_conf = threshold_adv_optim_conf_xi_2 if index < total_steps * 0.05 else threshold_adv_optim_conf_xi_1

        while (index >= total_steps * 0 and index <= total_steps * timestep_add_perturb) and optim_iters > 0:
            optim_iters -= 1

            with torch.enable_grad():
                latents_n = latents_adv.detach().requires_grad_(True)
                prev_timestep = int(remaining_steps) # modify this for continuous t (e.g., Euler Sampler)
                latents_n_0 = latents_n

                # predict a sketched x_0
                # print(prev_timestep)
                while residual_approx_step > 0 and prev_timestep >= 0:
                    latents_n, prev_timestep = generation_descriptor.sketch_step(latents_n, prev_timestep, jump_step, guidance_context)
                    # print(prev_timestep, jump_step)
                if prev_timestep > 0:
                    latents_n, prev_timestep = generation_descriptor.sketch_step(latents_n, prev_timestep, prev_timestep, guidance_context)
                if threshold_adv_optim_conf == threshold_adv_optim_conf_xi_2:  # refined sketch: considering latter clipping step
                    latents_n, _ = clip_latents_adv(EPS, generation_descriptor.sketch_step(latents, remaining_steps, remaining_steps, guidance_context)[0], latents_n)


                image = generation_descriptor.decode(latents_n)
                results, loss = victim_descriptor.get_results_loss(image, context_victim, result_differentiable=False,
                                                                   update_context=not first_step)

                asr_judge = victim_descriptor.attack_judge(results.detach(), context_victim)
                failed = asr_judge > threshold_adv_optim_conf
                if verbose: print("ASR: ", 1 - asr_judge, " Loss: ", loss.item(), " Failed: ", failed, "n_iter:", n_iter)

                gradient = torch.autograd.grad(loss, latents_n_0, create_graph=False, retain_graph=False)[0].detach()
                if v_inner is None:
                    v_inner = gradient
                v_inner = beta * v_inner + (1.00 - beta) * gradient

                del loss, results, latents_n, latents_n_0, image, gradient # For cuda memory gc after enable_grad
            # torch.cuda.empty_cache()

            if v_inner is not None:
                latents_adv = latents_adv.detach() + step_size * v_inner.detach().to(latents_adv)
                # if n_iter > optim_iters_normal: latents_adv, _ = clip_latents_adv(EPS, latents, latents_adv)


            n_iter += 1
            if not failed:
                break

        with torch.no_grad():
            if n_iter != 0 and first_step:
                latents, _ = clip_latents_adv(EPS * 2, latents, latents_adv) # protect from OOD
                first_step = False
            # original diffusion step
            latents_adv, remaining_steps_0 = generation_descriptor.original_step(latents_adv, index, guidance_context)
            latents, remaining_steps = generation_descriptor.original_step(latents, index, guidance_context)
            assert remaining_steps == remaining_steps_0
            # overall constraint
            latents_adv, delta = clip_latents_adv(EPS, latents, latents_adv)
            # adaptive step size
            if index < total_steps * timestep_add_perturb * 0.66 and delta < EPS * .5:
                step_size *= 1.5

            if verbose:
                # print(f"Index: {index}, Remaining steps: {remaining_steps}")
                # print(f"L2 norm of latents_adv: {torch.norm(latents_adv - latents, p=2)}")
                print(f"L2 norm of Delta: {delta}, EPS:{EPS}, index{index}, ASR: {1 - asr_judge}")

    # latents_exemplar = latents
    # Adv-related Edge Mask Write-Back For 2D Adv. Image Generation
    latents_exemplar = (latents+latents_adv) / 2
    latents_exemplar[:,: ,1:-1, 1:-1] = latents[:,: ,1:-1, 1:-1]

    return latents_adv, latents_exemplar


def clip_latents_adv(EPS, latents, latents_adv):
    total_changed = latents_adv - latents
    delta = float(torch.norm(total_changed, p = 2))
    if delta > EPS:
        total_changed1 = total_changed * (EPS / delta)
        latents_adv = latents + total_changed1
    return latents_adv, delta





