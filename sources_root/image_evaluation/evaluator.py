import os
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MetricCollection
from tqdm import tqdm

from workflow import GlobalSettings, init_standardization
from workflow.standarization import get_args, args_collect_standardization
from image_evaluation.datasets import dataset_registry, ImageEvaluationDataRaw, ImageEvaluationData
from image_evaluation.metrics import metric_registry


def flatten_dict(d, parent_key='', sep='/'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# Define the Lightning module
class Evaluator(pl.LightningModule):
    def __init__(self, config):
        super(Evaluator, self).__init__()
        # self.config = config
        self.hparams.update(config)
        # generate metric by config
        # metric dict define: {name_id: {type:..., **kwargs}}
        metric_dict = {}
        for k, v in tqdm(config.metrics.items(), desc="Loading Metrics"):
            # clone v and remove key "type"
            v = v.copy()
            type = v.pop("type", None)
            metric_dict[k] = metric_registry[type](**v)



        self.metrics = MetricCollection(metric_dict, prefix="evaluation/", compute_groups=False)
        self.notes = {}


    def on_test_epoch_start(self) -> None:
        # todo: add additional dataloader for FID - like evaluation.
        pass

    def test_step(self, batch, batch_idx):
        self.metrics.update(batch)

    def on_test_epoch_end(self):
        path = os.path.join(GlobalSettings.get_path(GlobalSettings.PathType.LOG, create=True), f"results/val_results.csv")
        # dir = os.path.join(getCfg().get_log_dir(), f"results/")

        if not os.path.exists(path):
            data = pd.DataFrame()
        else:
            data = pd.read_csv(path)
        # for i, metric in enumerate(self.metric):
        #     metric: Metric
        results = self.metrics.compute()
        results = flatten_dict(results)
        for k in results.keys():
            if isinstance(results[k], torch.Tensor):
                results[k] = results[k].item()
            else:
                results[k] = str(results[k])

        results["exp_id"] = GlobalSettings.experiment_id
        results["config_id"] = GlobalSettings.config
        results["project"] = GlobalSettings.project_name

        for k, v in self.notes.items():
            results[f"_note_{k}"] = v


            # results["time"] = self.tot_time / max(self.tot_time_cnt, 1)
        new_row = pd.DataFrame(results, index=[0])
        data = pd.concat([data, new_row], ignore_index=True)
        self.metrics.reset()
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)

        # self.log_dict(results)

    def configure_optimizers(self):
        # No optimizer needed since we are not training
        return None




import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
def __collate_fn__(batch: List[ImageEvaluationDataRaw]):
    images = torch.stack([data.image for data in batch])
    labels = torch.tensor([data.label for data in batch], dtype=torch.long)
    annotations = [data.annotation for data in batch]
    return ImageEvaluationData(images, labels, annotations)

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, dataset_config):
        super().__init__()
        self.dataset_config = dataset_config
        self.dataset = None

    def setup(self, stage=None):
        dataset_type = self.dataset_config.type
        dataset_params = OmegaConf.to_container(self.dataset_config, resolve=True)

        if dataset_type in dataset_registry:
            self.dataset = dataset_registry[dataset_type](**dataset_params)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.dataset_config.batch_size,
            shuffle=getattr(self.dataset_config, "shuffle", True),
            num_workers=getattr(self.dataset_config,"num_workers", 16),
            pin_memory=False,
            collate_fn=__collate_fn__
        )

    def val_dataloader(self):
        # Assuming you have a validation dataset
        return DataLoader(
            self.dataset,  # Replace with your validation dataset
            batch_size=self.dataset_config.batch_size,
            shuffle=False,
            num_workers=getattr(self.dataset_config,"num_workers", 16),
            pin_memory=False,
            collate_fn=__collate_fn__
        )

    def test_dataloader(self):
        # Assuming you have a test dataset
        return DataLoader(
            self.dataset,  # Replace with your test dataset
            batch_size=self.dataset_config.batch_size,
            shuffle=False,
            num_workers=getattr(self.dataset_config,"num_workers", 16),
            pin_memory=False,
            collate_fn=__collate_fn__
        )





def main():
    init_standardization()
    args_collect_standardization()
    global_cfg = get_args()
    log_dir = GlobalSettings.get_path(GlobalSettings.PathType.LOG, create=True)
    torch.set_float32_matmul_precision('high')
    logger = TensorBoardLogger(save_dir=GlobalSettings.log_root,
                               name=f"{GlobalSettings.project_name}/{GlobalSettings.config}", version=GlobalSettings.experiment_id, sub_dir="tensorboard")
    from pytorch_lightning.strategies import DDPStrategy
    single_gpu = global_cfg.trainer.devices == 1
    # Sync BN is not needed since model is in eval mode and BN will not be updated.
    # if not single_gpu:
    #     model = TorchSyncBatchNorm().apply(model)

    trainer = Trainer(**global_cfg.trainer,  default_root_dir=log_dir, logger=logger,
                      strategy="auto" if single_gpu else DDPStrategy(find_unused_parameters=True),
                      sync_batchnorm = False if single_gpu else True)

    evaluator = Evaluator(global_cfg.evaluation)
    evaluator.notes = global_cfg.evaluation.dataset
    trainer.test(evaluator, datamodule=ImageDataModule(global_cfg.evaluation.dataset))

    print("test done.")

if __name__ == '__main__':
    main()

