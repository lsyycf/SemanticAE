import sys
import os
import random
import argparse

import torchvision

from resadv_ddim.untargeted_attack import resadv_ddim
from surrogate_models.image_models import get_target_model_blackbox, get_surrogate_model_blackbox, get_transformation
from workflow import init_standardization
from workflow.standarization import get_args, GlobalSettings, args_collect_standardization

# sys.path.append(".")
import imagenet_label

import torch
from tqdm import tqdm

from diffusers import DDIMScheduler, StableDiffusionPipeline
from torch.backends import cudnn

import numpy as np

from torchvision.transforms.functional import to_pil_image

parser = argparse.ArgumentParser()

init_standardization("insur", parser)
args_collect_standardization()
args = get_args()

# label:  default= [107, 99, 113, 130, 207, 309]


def generate_latents(model, n_samples_per_class,img_size):
    latents_shape = (
                n_samples_per_class,
                model.unet.config.in_channels,
                img_size // model.vae_scale_factor,
                img_size // model.vae_scale_factor,
            )
    latents = torch.randn(
        latents_shape, device=model.device)
    latents = latents * model.scheduler.init_noise_sigma

    return latents

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





def main():

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.task.seed)
    torch.cuda.manual_seed(args.task.seed)
    np.random.seed(args.task.seed)
    random.seed(args.task.seed)

    # os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
        "bguisard/stable-diffusion-nano-2-1",
        torch_dtype=torch.float32,
        # local_files_only=True,
    )
    diffusion_pipeline.scheduler = DDIMScheduler.from_config(diffusion_pipeline.scheduler.config,
                                                             # local_files_only=True,
                                                             )
    diffusion_pipeline.to(device)
    #diffusion_pipeline.enable_xformers_memory_efficient_attention()
    diffusion_pipeline.enable_vae_slicing()
    
    vic_model = get_surrogate_model_blackbox(args.task.surrogate_model)
    print(args.task.surrogate_model.input_transformation)
    from surrogate_models.input_randomization import get_transformation
    loss_kernel = get_transformation(**args.task.surrogate_model.input_transformation)

    vic_model.to(device)
    vic_model.requires_grad_(False)
    vic_model.eval()



    if args.task.dev_run:
        #labels = np.arange(200)
        labels = [3]
    else:
        labels = np.arange(1000)


    n_samples_per_class = args.task.n_samples_per_class


    # target_label = args.target_label
    timesteps = args.attack.timesteps
    diffusion_pipeline.scheduler.num_inference_steps = timesteps
    timestep_add_perturb = args.attack.timestep_add_perturb
    scale = args.attack.scale   # for unconditional guidance
    scale_edge = args.attack.scale_edge   # for unconditional guidance
    budget_all = args.attack.budget_all   # for unconditional guidance


    save_path = GlobalSettings.get_path(GlobalSettings.PathType.LOG, create=True)
    test_image_path = os.path.join(save_path, "out_images")
    test_image_path_exemplar = os.path.join(save_path, "out_images_exemplar")
    test_label_path = os.path.join(save_path, "labels.txt")

    # delete test_label_path
    if os.path.exists(test_label_path) and args.task.rerun:
        os.remove(test_label_path)
    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(test_image_path_exemplar, exist_ok=True)

    with torch.no_grad():
        for i, class_label in enumerate(labels):
            text_label = imagenet_label.refined_Label[class_label]
            text_label = f"realistic image of {text_label}"
            # target_label = labels_targets[i]
            # target_label_text = imagenet_label.refined_Label[target_label]
            # print(f"rendering {n_samples_per_class} examples of class {class_label}: {text_label} -> {target_label} : {target_label_text} in {timesteps} steps and using s={scale:.2f}.")
            print(
                f"rendering {n_samples_per_class} non-target adv. examples of class {class_label}: {text_label} in {timesteps} steps and using s={scale:.2f}.")

            #latents, context, target_text, diffusion_pipeline, vic_model, timesteps: int, guidance_scale:float=2.5, eta:float=0.0, label=None, iterations: int=5, s:float=1.0, a:float=0.5, beta=0.5):

            label = torch.tensor([class_label]).long().unsqueeze(0).to(diffusion_pipeline.device)

            context =  generate_context(diffusion_pipeline, text_label)

            latents_batch = generate_latents(diffusion_pipeline, n_samples_per_class, args.task.image_size)

            for j in tqdm(
                range(n_samples_per_class),
                total=n_samples_per_class,
                desc="Samples",
                leave=False,
            ):

                if not args.task.rerun:
                    tt = imagenet_label.refined_Label[class_label]
                    image_name = f"{tt}_{j}"
                    image_name = image_name.replace(" ", "_")
                    filename = os.path.join(test_image_path, f"{image_name}.png")
                    if os.path.exists(filename):
                        print(f"{filename} already exists, skipping...")
                        continue


                latents_adv, latents_exemplar = resadv_ddim(
                    latents=latents_batch[[j]],
                    context=context,
                    model=diffusion_pipeline,
                    vic_model=vic_model,
                    trans_kernel=loss_kernel,
                    timesteps = timesteps,
                    timestep_add_perturb=timestep_add_perturb,
                    gudiance_scale_edge = scale_edge,
                    guidance_scale = scale,
                    budget_all = budget_all,
                    budget_onestep=0.1,
                    eta = args.attack.ddim_eta,
                    label=label.to(diffusion_pipeline.device),
                    iterations=args.attack.K,
                    s=args.attack.s,
                    beta=args.attack.beta
                )


                samples = latents_exemplar / diffusion_pipeline.vae.config.scaling_factor
                images = diffusion_pipeline.vae.decode(samples, return_dict=False)[0]
                images = diffusion_pipeline.image_processor.postprocess(images, output_type="pt")

                samples_adv = latents_adv / diffusion_pipeline.vae.config.scaling_factor
                images_adv = diffusion_pipeline.vae.decode(samples_adv, return_dict=False)[0]
                images_adv = diffusion_pipeline.image_processor.postprocess(images_adv, output_type="pt")
                torch.cuda.empty_cache()


                text_label = imagenet_label.refined_Label[class_label]
                image_name = f"{text_label}_{j}"
                image_name = image_name.replace(" ", "_")
                to_pil_image(images_adv[0].cpu()).save(os.path.join(test_image_path, f"{image_name}.png"))
                to_pil_image(images[0].cpu()).save(os.path.join(test_image_path_exemplar, f"{image_name}.png"))
                # save the test image
                # export the target_label to log dir/label.txt

                with open(test_label_path, "a") as f:
                    f.write(f"{image_name}.png, {class_label}\n")
                        # parse: eval(line) (may not secure, use in trusted codes)

if __name__ == '__main__':
    main()




