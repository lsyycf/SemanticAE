import os
from typing import Dict, Type, Tuple, Any, Literal

import pyiqa
import torch
import torchvision.models
from torch import nn, Tensor
from torchmetrics import Metric, Accuracy
from torchmetrics.functional.image.lpips import _NoTrainLpips, _lpips_update
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore
from torchvision.transforms import transforms

from image_evaluation.datasets import ImageEvaluationData
from surrogate_models.image_models import get_target_model_blackbox

# Metric registry
metric_registry: Dict[str, Type[Metric]] = {}


def register_metric(name: str):
    def decorator(cls: Type[Metric]):
        metric_registry[name] = cls
        return cls

    return decorator


#
# __path_extended__ = False
# def get_target_model(target_model_config: dict):
#     from workflow import GlobalSettings
#     from workflow.standarization import temp_chdir
#     PATH_BLACKBOX = os.path.join(GlobalSettings.get_path(GlobalSettings.PathType.SOURCES), "blackboxbench/blackbox")
#     global __path_extended__
#     if not __path_extended__:
#         __path_extended__ = True
#         import sys
#         sys.path.append(PATH_BLACKBOX)
#     import surrogate_model.utils
#     import tools.load
#     surrogate_model.utils.base_dir = os.path.join(PATH_BLACKBOX, "surrogate_model")
#     with temp_chdir(PATH_BLACKBOX):
#         target = tools.load.load_target_models(config=target_model_config)
#     return target
#

def clip_image_data(data: ImageEvaluationData, key="dist_anchor", clip_value: object = True):
    use_value = [a.get(key) != clip_value for a in data.annotations]
    if not any(use_value):
        return None
    annotations_filtered = [a for i, a in enumerate(data.annotations) if use_value[i]]
    return ImageEvaluationData(image=data.image[use_value], label=data.label[use_value],
                               annotations=annotations_filtered)



@register_metric('cls_atk')
class ClsAtk(Metric):
    def __init__(self, target_model, accuracy=None, accuracy_dict=None, num_classes=1000, **kwargs):
        super().__init__()

        if accuracy is None and accuracy_dict is None:
            raise ValueError("Either accuracy or accuracy_dict must be provided.")
        # TODO: build multi accuracy metric by accuracy_dic
        self.ac_metric = Accuracy(**accuracy)
        self.ac_metric_anchor = Accuracy(**accuracy)
        self.add_state("anchor_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("adv_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("effective_data", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("success_attack", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("all_attack", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("per_class_ASR", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("per_class_total", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        target = get_target_model_blackbox(target_model)
        self.target_model_name = "target_model"
        self.target = target
        # target_model = torchvision.models.resnet18(weights='IMAGENET1K_V1').eval()
        # self.target = [target_model]
        # print(self.target)

        # [self.add_module(f"target_model_{i}", t) for i, t in enumerate(self.target)]


    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)
        if image_ev is None:
            return

        image_ev = clip_image_data(image_ev, "anchor_image", None)
        if image_ev is None:
            return

        results = self.target(image_ev.image.to(self.device)).to(self.device)  # for t in self.target]

        anchors = torch.stack([a["anchor_image"] for a in image_ev.annotations]).to(self.device)
        results_anchor = self.target(anchors)
        labels = image_ev.label.to(self.device)
        prob_confidence = torch.softmax(results_anchor, dim=-1)[torch.arange(image_ev.image.shape[0]), labels]
        # noinspection PyTypeChecker
        flag = torch.logical_and(prob_confidence > 0.8, results.argmax(dim=-1) != labels)

        adv_correct = labels == results.argmax(dim=-1)
        anchor_correct = labels == results_anchor.argmax(dim=-1)
        flag1 = torch.logical_and(torch.logical_not(adv_correct), anchor_correct)
        self.effective_data += flag.sum()
        self.success_attack += flag1.sum()
        self.adv_correct += adv_correct.sum()
        self.anchor_correct += anchor_correct.sum()

        # Update per-class correct and total
        for label, correct in zip(labels, adv_correct):
            self.per_class_total[label] += 1
            self.per_class_ASR[label] += correct

        # get the mean val of results.
        # results = torch.stack(results).mean(dim=0)

        self.ac_metric.update(results, labels)
        self.ac_metric_anchor.update(results_anchor, labels)
        self.all_attack += image_ev.image.shape[0]

    def compute(self):
        accuracy = self.ac_metric.compute()
        anchor_accuracy = self.ac_metric_anchor.compute()
        anchor_accuracy_top = float(self.anchor_correct) / float(self.all_attack)
        adv_accuracy_top = float(self.adv_correct) / float(self.all_attack)
        ASR = float(self.success_attack) / float(self.anchor_correct)
        # EDR = float(self.effective_data) / float(self.all_attack)  # Effective Data Rate

        # Calculate per-class accuracy
        per_class_ASR = self.per_class_ASR / self.per_class_total
        per_class_ASR = per_class_ASR[~torch.isnan(per_class_ASR)]

        # Calculate standard deviation of per-class accuracy
        std_per_class_ASR = per_class_ASR.std().item()

        return {
            # "accuracy": accuracy,
            # "anchor_accuracy": anchor_accuracy,
            "benign_acc": anchor_accuracy_top,
            "adv_acc": adv_accuracy_top,
            "ASR": ASR,
            # "EDR": EDR,
            # "per_class_ASR": per_class_ASR.cpu().detach().tolist(),
            "std_per_class_ASR": std_per_class_ASR
        }

    def reset(self) -> None:
        self.ac_metric.reset()
        self.ac_metric_anchor.reset()



@register_metric('cls_atk_targeted')
class ClsAtkT(ClsAtk):
    def __init__(self, target_model, accuracy=None, accuracy_dict=None, **kwargs):
        super().__init__(target_model, accuracy, accuracy_dict, **kwargs)


    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)
        if image_ev is None:
            return

        image_ev = clip_image_data(image_ev, "anchor_image", None)
        if image_ev is None:
            return

        results = self.target(image_ev.image.to(self.device)).to(self.device)  # for t in self.target]

        anchors = torch.stack([a["anchor_image"] for a in image_ev.annotations]).to(self.device)
        labels_t = torch.tensor([a["target_label"] for a in image_ev.annotations]).to(self.device)
        results_anchor = self.target(anchors)
        labels = image_ev.label.to(self.device)
        prob_confidence = torch.softmax(results_anchor, dim=-1)[torch.arange(image_ev.image.shape[0]), labels]
        # noinspection PyTypeChecker
        flag = torch.logical_and(prob_confidence > 0.8, results.argmax(dim=-1) == labels_t)

        # get the mean val of results.
        # results = torch.stack(results).mean(dim=0)

        self.ac_metric.update(results, labels_t)
        self.ac_metric_anchor.update(results_anchor, labels)
        self.success_attack += flag.sum()
        self.all_attack += image_ev.image.shape[0]


@register_metric('cls_atk_multilabel')
class ClsAtkM(Metric):
    def __init__(self, target_model, accuracy=None, num_classes = 1000, accuracy_dict=None, **kwargs):
        super().__init__()

        if accuracy is None and accuracy_dict is None:
            raise ValueError("Either accuracy or accuracy_dict must be provided.")
        # TODO: build multi accuracy metric by accuracy_dic
        self.ac_metric = Accuracy(**accuracy)
        self.ac_metric_anchor = Accuracy(**accuracy)
        self.add_state("anchor_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("adv_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("effective_data", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("success_attack", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("all_attack", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("per_class_ASR", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("per_class_total", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        target = get_target_model_blackbox(target_model)
        self.target_name = target_model
        self.target = target
        # target_model = torchvision.models.resnet18(weights='IMAGENET1K_V1').eval()
        # self.target = [target_model]
        # print(self.target)

        # [self.add_module(f"target_model_{i}", t) for i, t in enumerate(self.target)]

    def update(self, image_ev: ImageEvaluationData):
        print(f"evaluated:{self.target_name}")
        image_ev = clip_image_data(image_ev, "dist_anchor", True)
        if image_ev is None:
            return

        image_ev = clip_image_data(image_ev, "anchor_image", None)
        if image_ev is None:
            return

        results = self.target(image_ev.image.to(self.device)).to(self.device)  # for t in self.target]

        anchors = torch.stack([a["anchor_image"] for a in image_ev.annotations]).to(self.device)
        results_anchor = self.target(anchors)
        # labels =  torch.stack([a["class_id"] for a in image_ev.annotations])
        labels =  torch.zeros_like(results.to(self.device), dtype=torch.long)
        for i, a in enumerate(image_ev.annotations):
            labels[i, a["class_id"].to(self.device).long()] = 1


        prob_confidence_anchor = torch.softmax(results_anchor, dim=-1) #[torch.arange(image_ev.image.shape[0]), labels]
        prob_confidence_anchor[torch.logical_not(labels)] = 0
        # print(prob_confidence_anchor.max(dim = -1)[0])
        # noinspection PyTypeChecker

        adv_correct = labels[torch.arange(labels.shape[0]), results.argmax(dim=-1)]
        anchor_correct = labels[torch.arange(labels.shape[0]), results_anchor.argmax(dim=-1)]
        flag = torch.logical_and(prob_confidence_anchor.max(dim=-1)[0] > 0.5, torch.logical_not(adv_correct))
        flag1 = torch.logical_and(torch.logical_not(adv_correct), anchor_correct)

        for label, correct in zip(labels, flag1):
            self.per_class_total[label] += 1
            self.per_class_ASR[label] += correct
        # get the mean val of results.
        # results = torch.stack(results).mean(dim=0)

        self.ac_metric.update(results, labels)
        self.ac_metric_anchor.update(results_anchor, labels)
        self.effective_data += flag.sum()
        self.success_attack += flag1.sum()
        self.adv_correct += adv_correct.sum()
        self.anchor_correct += anchor_correct.sum()
        self.all_attack += image_ev.image.shape[0]


    def compute(self):

        # Calculate per-class accuracy
        per_class_ASR = self.per_class_ASR / self.per_class_total
        per_class_ASR = per_class_ASR[~torch.isnan(per_class_ASR)]

        # Calculate standard deviation of per-class accuracy
        std_per_class_ASR = per_class_ASR.std().item()
        return {
            # "accuracy": self.ac_metric.compute(),
            # "anchor_accuracy": self.ac_metric_anchor.compute(),
            "benign_acc": float(self.anchor_correct) / float(self.all_attack),
            "adv_acc": float(self.adv_correct) / float(self.all_attack),
            "ASR": float(self.success_attack) / float(self.anchor_correct),
            # "EDR": float(self.effective_data) / float(self.all_attack), # Effective Data Rate

            # "per_class_ASR": per_class_ASR.cpu().detach().tolist(),
            "std_per_class_ASR": std_per_class_ASR
        }

    def reset(self) -> None:
        self.ac_metric.reset()
        self.ac_metric_anchor.reset()
    def _get_name(self):
        return f"cls_atk_multilabel{self.target_name}"



from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, \
    MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


def corrected_std(sum_x2, sum_x, n):
    """
    Calculate the corrected standard deviation.

    :param sum_x2: Sum of squares of the values
    :param sum_x: Sum of the values
    :param n: Number of observations
    :return: Corrected standard deviation
    """
    if n <= 1:
        raise ValueError("Number of observations must be greater than 1")

    mean_square = (sum_x ** 2) / n
    variance = (sum_x2 - mean_square) / (n - 1)
    std_dev = variance ** .5

    return std_dev

@register_metric('LPIPS')
class LPIPS_Wrap(Metric):
    def __init__(self, net_type: Literal["vgg", "alex", "squeeze"] = "alex", **kwargs):
        super().__init__()

        # self.lpips = LearnedPerceptualImagePatchSimilarity(**kwargs)
        self.net = _NoTrainLpips(net=net_type)
        # self.add_module("lpips", self.lpips)

        self.add_state("sum_score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_score2", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self,  image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)
        if image_ev is None:
            return
        image_ev = clip_image_data(image_ev, "anchor_image", None)
        if image_ev is None:
            return
        anns = torch.stack([a["anchor_image"] for a in image_ev.annotations]).to(self.device)
        """Update internal states with lpips score."""
        loss, total = _lpips_update(image_ev.image.to(self.device), anns, net=self.net, normalize=True)
        self.sum_score += loss.sum()
        self.sum_score2 += (loss ** 2).sum()
        self.total += total
    # def update(self, image_ev: ImageEvaluationData):
    #     image_ev = clip_image_data(image_ev, "dist_anchor", True)
    #     if image_ev is None:
    #         return
    #     image_ev = clip_image_data(image_ev, "anchor_image", None)
    #     if image_ev is None:
    #         return
    #     anns = torch.stack([a["anchor_image"] for a in image_ev.annotations]).to(self.device)
    #     self.lpips.update(image_ev.image.to(self.device), anns)

    def compute(self):
        print("=====LPIPS=====")
        # output = self.lpips.compute()
        # compute std and mean
        results =  {"mean": self.sum_score / self.total,
                "std": corrected_std(self.sum_score2, self.sum_score, self.total)}
        print(results)
        return results

    # def reset(self) -> None:
    #     self.lpips.reset()


@register_metric('MSSSIM')
class MSSSIM_Wrap(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["reduction"] = "none"
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(**kwargs)
        self.add_module("msssim", self.msssim)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256))
        ])

    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)
        if image_ev is None:
            return
        image_ev = clip_image_data(image_ev, "anchor_image", None)
        if image_ev is None:
            return
        anns = torch.stack([a["anchor_image"] for a in image_ev.annotations]).to(self.device)

        self.msssim.update(self.transform(image_ev.image.to(self.device)), self.transform(anns))

    def compute(self):
        print("=====msssim=====")
        output = self.msssim.compute()

        # compute std and mean
        results = {
            "std": output.std().item(),
            "mean": output.mean().item()
        }
        print(results)
        return results

    def reset(self) -> None:
        self.msssim.reset()


@register_metric('IS')
class IS_Wrap(InceptionScore):
    def __init__(self, **kwargs):
        super().__init__()
        self.score = InceptionScore(**kwargs)
        self.add_module("score", self.score)

    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)
        # image_ev = clip_image_data(image_ev, "anchor_image", None)
        if image_ev is None:
            return
        self.score.update(image_ev.image.to(self.device))

    def compute(self):
        print("=====IS=====")
        score = self.score.compute()
        print(score)
        return {"std": score[1], "mean": score[0]}

    def reset(self) -> None:
        self.score.reset()


@register_metric('FID')
class FID_Wrap(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get("input_img_size", None) is None:
            kwargs["input_img_size"] = (3, 224, 224)

        self.fid = FrechetInceptionDistance(**kwargs)
        self.add_module("fid", self.fid)

    def update(self, image_ev: ImageEvaluationData):
        if image_ev is None:
            return
        image_ev_syn = clip_image_data(image_ev, "dist_anchor", True)
        if image_ev_syn is not None:
            self.fid.update(image_ev_syn.image.to(self.device), False)

        image_ev_ref = clip_image_data(image_ev, "dist_anchor", False)
        if image_ev_ref is not None:
            self.fid.update(image_ev_ref.image.to(self.device), True)

    def compute(self):
        print("=====fid=====")
        print(self.fid.compute())
        return self.fid.compute()

    def reset(self) -> None:
        self.fid.reset()


@register_metric('CLIPQuality')
class CLIP_IQA(Metric):
    """
    Prompt: quality, natural
        * quality: "Good photo." vs "Bad photo."
        * brightness: "Bright photo." vs "Dark photo."
        * noisiness: "Clean photo." vs "Noisy photo."
        * colorfullness: "Colorful photo." vs "Dull photo."
        * sharpness: "Sharp photo." vs "Blurry photo."
        * contrast: "High contrast photo." vs "Low contrast photo."
        * complexity: "Complex photo." vs "Simple photo."
        * natural: "Natural photo." vs "Synthetic photo."
        * happy: "Happy photo." vs "Sad photo."
        * scary: "Scary photo." vs "Peaceful photo."
        * new: "New photo." vs "Old photo."
        * warm: "Warm photo." vs "Cold photo."
        * real: "Real photo." vs "Abstract photo."
        * beautiful: "Beautiful photo." vs "Ugly photo."
        * lonely: "Lonely photo." vs "Sociable photo."
        * relaxing: "Relaxing photo." vs "Stressful photo."
    """

    def __init__(self, prompt="quality", **kwargs):
        super().__init__()

        self.model = CLIPImageQualityAssessment(prompts=(prompt,))
        self.add_module("model", self.model)

    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)

        if image_ev is None:
            return
        self.model.update(image_ev.image.to(self.device))

    def compute(self):
        print("=====CLIP_IQA=====")
        print(self.model.compute())
        return self.model.compute()

    def reset(self) -> None:
        self.model.reset()


@register_metric('CLIPQuality')
class CLIPQuality(Metric):
    """
    Prompt: quality, natural
        * quality: "Good photo." vs "Bad photo."
        * brightness: "Bright photo." vs "Dark photo."
        * noisiness: "Clean photo." vs "Noisy photo."
        * colorfullness: "Colorful photo." vs "Dull photo."
        * sharpness: "Sharp photo." vs "Blurry photo."
        * contrast: "High contrast photo." vs "Low contrast photo."
        * complexity: "Complex photo." vs "Simple photo."
        * natural: "Natural photo." vs "Synthetic photo."
        * happy: "Happy photo." vs "Sad photo."
        * scary: "Scary photo." vs "Peaceful photo."
        * new: "New photo." vs "Old photo."
        * warm: "Warm photo." vs "Cold photo."
        * real: "Real photo." vs "Abstract photo."
        * beautiful: "Beautiful photo." vs "Ugly photo."
        * lonely: "Lonely photo." vs "Sociable photo."
        * relaxing: "Relaxing photo." vs "Stressful photo."
    """

    def __init__(self, prompt="quality", **kwargs):
        super().__init__()

        self.model = CLIPImageQualityAssessment(prompts=(prompt,))
        self.add_module("model", self.model)

    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)

        if image_ev is None:
            return
        self.model.update(image_ev.image.to(self.device))

    def compute(self):
        print("=====CLIPQuality=====")
        scores = self.model.compute()
        # print(scores)

        clipscore = {"std": scores.std(), "mean": scores.mean()}
        print(clipscore)
        return clipscore

    def reset(self) -> None:
        self.model.reset()


from cas_diffusion_attack.dataset_caption import imagenet_label


@register_metric('CLIPSemanticBenignImage')
class CLIPSemanticBenignImage(Metric):
    def __init__(self, **kwargs):
        super().__init__()

        # self.model = CLIPScore(**kwargs)
        # self.add_module("model", self.model)

        self.metric_pyiqa = pyiqa.create_metric("clipscore")
        self.add_state("sum_score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_score2", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_sample", torch.tensor(0), dist_reduce_fx="sum")

        self.use_annotation = kwargs.get("use_annotation", False)

    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)

        if image_ev is None:
            return

        # anns = torch.stack().to(self.device)
        # print([imagenet_label.refined_Label[int(l)] for l in image_ev.label])
        # self.model.update(image_ev.image.to(self.device), [imagenet_label.refined_Label[int(l)] for l in image_ev.label])
        if self.use_annotation:
            captionlist = [a["semantic"] for a in image_ev.annotations]
        else:
            captionlist = [imagenet_label.refined_Label[int(l)] for l in image_ev.label]

        scores = self.metric_pyiqa(
            torch.stack([a["anchor_image"].to(self.device) for a in image_ev.annotations], dim=0),
            caption_list=captionlist)
        self.sum_score += scores.sum()
        self.n_sample += len(image_ev.image)
        self.sum_score2 += (scores ** 2).sum()

    def compute(self):
        # return self.model.compute()
        return {"mean": self.sum_score / self.n_sample,
                "std": corrected_std(self.sum_score2, self.sum_score, self.n_sample)}




@register_metric('CLIPSemanticAdvImage')
class CLIPSemanticAdvImage(Metric):
    def __init__(self, **kwargs):
        super().__init__()

        # self.model = CLIPScore(**kwargs)
        # self.add_module("model", self.model)

        self.metric_pyiqa = pyiqa.create_metric("clipscore")
        self.add_state("sum_score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_score2", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_sample", torch.tensor(0), dist_reduce_fx="sum")

        self.use_annotation = kwargs.get("use_annotation", False)

    def update(self, image_ev: ImageEvaluationData):
        image_ev = clip_image_data(image_ev, "dist_anchor", True)

        if image_ev is None:
            return

        # anns = torch.stack().to(self.device)
        # print([imagenet_label.refined_Label[int(l)] for l in image_ev.label])
        # self.model.update(image_ev.image.to(self.device), [imagenet_label.refined_Label[int(l)] for l in image_ev.label])
        if self.use_annotation:
            captionlist = [a["semantic"] for a in image_ev.annotations]
        else:
            captionlist = [imagenet_label.refined_Label[int(l)] for l in image_ev.label]
        print(captionlist)
        scores = self.metric_pyiqa(image_ev.image, caption_list=captionlist)
        self.sum_score += scores.sum()
        self.n_sample += len(image_ev.image)
        self.sum_score2 += (scores ** 2).sum()

    def compute(self):
        # return self.model.compute()
        return {"mean": self.sum_score / self.n_sample,
                "std": corrected_std(self.sum_score2, self.sum_score, self.n_sample)}


if __name__ == '__main__':
    print(metric_registry.keys())
