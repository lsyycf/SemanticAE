import os
from functools import partial
from typing import *
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.utils import save_image
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
import torch.nn.functional as F
import utils3d

from omegaconf import OmegaConf
from .base import Pipeline
from . import samplers
from .samplers.adv_sampler import AdvSampler
from ..modules import sparse as sp

from ..renderers import gaussian_render
from ..renderers.gaussian_render import GaussianRenderer
from ..utils.render_utils import get_renderer

# class GradPrecomputeKernel(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, times, randomized_forward_fn):
#         loss = randomized_forward_fn
#         return loss
#
#     @staticmethod
#     def backward(ctx, *grad_output):
#         grad, = ctx.saved_tensors
#         grad_input = grad * grad_output[0] # grad_output is Delta loss, normally is one.
#         return grad_input, None, None
#


class TrellisTextTo3DPipelineAdv(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
        resolution=(224, 224),
        eps=12.5,
        grad_acc=5,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

        self.adv_sampler_slat = AdvSampler(
            self.slat_sampler, eps=eps, grad_acc=grad_acc
        )
        self.resolution = resolution

        rendering_options = {"near": 0.8, "far": 1.6, "ssaa": 1, "bg_color": (0, 0, 0)}
        # rendering_options = {"bg_color": (0,0,0)}
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.rendering_options.resolution = self.resolution[0]

        self.renderer.pipe.kernel_size = (
            0.1  #  self.models['slat_decoder_gs'].rep_config['2d_filter_kernel_size']
        )

        # self.evasion_label = torch.zeros(1000, device=self.device, dtype=torch.bool)
        # self.evasion_label[123] = True
        self.target_model = torchvision.models.resnet50()
        self.target_model.load_state_dict(torch.load("./resnet50.pth"))
        self.target_model.eval().cuda()

    def sample_and_render(self, rep):

        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2] * 1
        # yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + np.random.uniform(-np.pi / 4, np.pi / 4) for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(len(yaws))]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = (
                torch.tensor(
                    [
                        np.sin(yaw) * np.cos(pitch),
                        np.cos(yaw) * np.cos(pitch),
                        np.sin(pitch),
                    ]
                )
                .float()
                .cuda()
                * 2
            )
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = self.renderer
        images = []
        # import pdb
        # pdb.set_trace()
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            img = renderer.render(rep, ext, intr)["color"].clip(0, 1)
            # resize image to self.resolution
            # print(img.max(), img.min())
            img = torchvision.transforms.Resize(self.resolution)(img)
            images.append(img)
        images = torch.stack(images)

        # save images to temp_vis/123
        # save_dir = "temp_vis/124"
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # 保存每张图像
        #
        # for i, img in enumerate(images):
        #     save_path = os.path.join(save_dir, f"image_{i}.png")
        #     save_image(img, save_path)

        return images

    def get_target_label_stamo_multilabel(self, logits):  # seond-like label for attack
        logits = logits.clone().detach().mean(dim=0)
        logits[self.evasion_label] = -torch.inf
        rates, indices = logits.sort(-1, descending=True)
        print(indices[0])
        # rates, indices = rates.squeeze(0), indices.squeeze(0)

        # tar_label = torch.zeros(logits.shape[0], device=label.device, dtype=torch.long)

        return indices[0]

    def adv_feedback(self, images):
        logits = self.target_model(images)

        prob = F.softmax(logits, dim=-1)[..., self.evasion_label].max(dim=-1)[0].mean()
        loss = (
            F.log_softmax(logits, dim=-1)[
                ..., self.get_target_label_stamo_multilabel(logits)
            ].mean()
            - (
                F.log_softmax(logits, dim=-1)
                * self.evasion_label.float()
                / self.evasion_label.float().sum(dim=-1, keepdim=True)
            )
            .sum(dim=-1)
            .mean()
        )  #  / self.evasion_label.sum()
        return prob, loss

    def estimate_from_sparse(self, z_s, cond, slat_sampler_params):

        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        num_step_old = slat_sampler_params["steps"]
        slat_sampler_params["steps"] = 3
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        slat_sampler_params["steps"] = num_step_old
        return self.adv_feedback(self.estimate_from_slat(slat))

    def estimate_from_slat(self, slat):
        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean
        rep = self.models["slat_decoder_gs"](slat)[0]
        return self.adv_feedback(self.sample_and_render(rep))

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipelineAdv":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(
            TrellisTextTo3DPipelineAdv, TrellisTextTo3DPipelineAdv
        ).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipelineAdv()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]

        new_pipeline._init_text_cond_model(args["text_cond_model"])

        new_pipeline.adv_sampler_slat = AdvSampler(new_pipeline.slat_sampler)
        new_pipeline.resolution = (224, 224)

        rendering_options = {"near": 0.8, "far": 1.6, "ssaa": 1, "bg_color": (0, 0, 0)}
        new_pipeline.renderer = GaussianRenderer(rendering_options)
        new_pipeline.renderer.pipe.kernel_size = 0.1
        new_pipeline.renderer.rendering_options.resolution = new_pipeline.resolution[0]
        # attack python debugger

        new_pipeline.renderer.pipe.kernel_size = new_pipeline.models[
            "slat_decoder_gs"
        ].rep_config[
            "2d_filter_kernel_size"
        ]  # new_pipeline.evasion_label = torch.zeros(1000, device=new_pipeline.models['decoder'].device, dtype=torch.bool)
        # new_pipeline.evasion_label[123] = True
        new_pipeline.target_model = torchvision.models.resnet50()
        new_pipeline.target_model.load_state_dict(torch.load("./resnet50.pth"))
        new_pipeline.target_model.eval().cuda()

        return new_pipeline

    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            "model": model,
            "tokenizer": tokenizer,
        }
        self.text_cond_model["null_cond"] = self.encode_text([""])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(
            isinstance(t, str) for t in text
        ), "text must be a list of strings"
        encoding = self.text_cond_model["tokenizer"](
            text,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoding["input_ids"].cuda()
        embeddings = self.text_cond_model["model"](input_ids=tokens).last_hidden_state

        return embeddings

    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model["null_cond"]
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(
            self.device
        )
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if "mesh" in formats:
            ret["mesh"] = self.models["slat_decoder_mesh"](slat)
        if "gaussian" in formats:
            ret["gaussian"] = self.models["slat_decoder_gs"](slat)
        if "radiance_field" in formats:
            ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models["slat_flow_model"]
        # TODO: Optimize Coords
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        ret = self.adv_sampler_slat.sample(
            flow_model,
            noise,
            adv_guidance=self.estimate_from_slat,
            **cond,
            **sampler_params,
            verbose=True
        )
        slat = ret.samples
        slat_clean = ret.samples_clean

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean
        slat_clean = slat_clean * std + mean

        return slat, slat_clean

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        target_label: Union[int, List[int]],
        eps=12.5,
        num_samples: int = 1,
        seed: int = 42,
        grad_acc=5,
        approx_k=4,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        # if isinstance(target_label, list) and len(target_label > 1):
        # build 1000 class one hot vector
        template = torch.zeros(1000)
        template[target_label] = 1
        self.adv_sampler_slat.eps = eps
        self.adv_sampler_slat.grad_acc = grad_acc
        self.adv_sampler_slat.approx_k = approx_k

        self.evasion_label = template.bool().to(self.device).reshape(1000)
        # else:
        #     self.evasion_label = torch.tensor([target_label]).to(self.device).reshape(1, 1)

        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(
            cond, num_samples, sparse_structure_sampler_params
        )
        slat, slat_clean = self.sample_slat(cond, coords, slat_sampler_params)

        return self.decode_slat(slat, formats), self.decode_slat(slat_clean, formats)

    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1 / 64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat(
            [
                torch.arange(num_samples)
                .repeat_interleave(coords.shape[0], 0)[:, None]
                .int()
                .cuda(),
                coords.repeat(num_samples, 1),
            ],
            1,
        )
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
