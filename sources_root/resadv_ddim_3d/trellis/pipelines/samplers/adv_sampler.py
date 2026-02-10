from typing import Optional, Any

import numpy as np
import torch
from tqdm import tqdm

from . import FlowEulerSampler, Sampler


from easydict import EasyDict as edict

from ...modules.sparse import SparseTensor


# class AdvGuidanceImageModel(Sampler):






class AdvSampler(Sampler):
    def __init__(
        self,
        origin_sampler: FlowEulerSampler,
        eps = 12.5,
        grad_acc = 5
    ):
        self.origin_sampler = origin_sampler
        self.eps = eps
        self.grad_acc = grad_acc
        self.approx_k = 4


    def sample(
        self,
        model,
        noise,
        adv_guidance,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        sample = noise
        sample_adv = noise
        first_step = True
        momentum = None
        print(self.approx_k)

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((i, t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [], "samples_clean": None})
        for index_t, t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            with torch.enable_grad():
                gradient = None
                if index_t > steps * 0.25:
                    for k in range(30):
                        sample_adv: SparseTensor
                        latents = SparseTensor(feats=sample_adv.feats.detach().clone(), coords=sample_adv.coords.detach().clone())
                        latents.feats.requires_grad_(True)
                        latents_1 = latents
                        next_index = index_t
                        while next_index < steps and self.approx_k > 0:
                            tc =  float(t_seq[next_index])
                            next_index = min(steps, next_index + steps // self.approx_k)
                            tp = float(t_seq[next_index])
                            out = self.origin_sampler.sample_once(model, latents_1, tc, tp, cond, **kwargs)
                            latents_1 = out.pred_x_prev

                        prob, loss = adv_guidance(latents_1)
                        print(loss, prob)
                        if prob.mean() > 0.01:
                            grad = torch.autograd.grad(loss.sum(), latents.feats)[0]

                            if (k + 1) % self.grad_acc == 0:
                                gradient = gradient + grad if gradient is not None else grad
                                momentum = momentum * 0.5 + gradient * 0.5 if momentum is not None else gradient
                                gradient = None
                                sample_adv = SparseTensor(feats=(sample_adv.feats + momentum * 0.5).detach(), coords=sample_adv.coords.detach().clone())

                                diff = torch.norm(sample_adv.feats - sample.feats, p=2)

                                print(diff)
                                if diff > self.eps * 1.1:
                                    sample_adv = SparseTensor(
                                        feats=(sample.feats + (self.eps / diff) * (sample_adv.feats - sample.feats)).detach(),
                                        coords=sample.coords.detach().clone())


                                if first_step and diff > self.eps / 2:
                                    first_step = False
                                    sample = sample_adv
                            else:
                                gradient = gradient + grad if gradient is not None else grad
                        else: break

            with torch.no_grad():
                out_adv = self.origin_sampler.sample_once(model, sample_adv, t, t_prev, cond, **kwargs)
                out = self.origin_sampler.sample_once(model, sample, t, t_prev, cond, **kwargs)
            # clip ||out_adv, out|| to max = eps

            sample = out.pred_x_prev
            sample_adv = out_adv.pred_x_prev

            # ret.pred_x_t.append(out_adv.pred_x_t)
            # ret.pred_x_0.append(out_adv.pred_x_0)

            diff = torch.norm(sample_adv.feats - sample.feats, p = 2)

            print(diff)
            if diff > self.eps:
                sample_adv = SparseTensor(feats=(sample.feats + (self.eps / diff) * (sample_adv.feats - sample.feats)).detach(), coords=sample.coords.detach().clone())


        ret.samples = sample_adv
        ret.samples_clean = sample
        return ret

