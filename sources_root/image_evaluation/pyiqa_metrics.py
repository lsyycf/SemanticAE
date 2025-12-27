import torch
from pyiqa import SSIM, PSNR
from torchmetrics import Metric
from omegaconf import OmegaConf

# Step 1: Define a wrapper class for piqa metrics
class PiqaMetricWrapper(Metric):
    def __init__(self, piqa_metric_class, **kwargs):
        """
        Wraps a piqa metric into a torchmetrics-compatible Metric.

        Args:
            piqa_metric_class: The piqa metric class (e.g., SSIM, PSNR).
            **kwargs: Configuration arguments for the piqa metric.
        """
        super().__init__()
        self.piqa_metric = piqa_metric_class(**kwargs)
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with new predictions and targets.

        Args:
            preds: Predicted images (batch of tensors).
            target: Ground truth images (batch of tensors).
        """
        batch_size = preds.shape[0]
        self.value += self.piqa_metric(preds, target).sum()
        self.total += batch_size

    def compute(self):
        """
        Compute the final metric value.
        """
        return self.value / self.total

# Step 2: Create a configuration setter using OmegaConf
def get_metric_from_config(config):
    """
    Automatically configures and returns a wrapped piqa metric based on OmegaConf.

    Args:
        config: OmegaConf object containing metric configuration.

    Returns:
        A PiqaMetricWrapper instance.
    """
    # Extract metric name and parameters from config
    metric_name = config.metric.name
    metric_params = config.metric.params

    # Map metric names to piqa classes
    piqa_metric_map = {
        "SSIM": SSIM,
        "PSNR": PSNR,
    }

    if metric_name not in piqa_metric_map:
        raise ValueError(f"Unsupported metric: {metric_name}")

    # Get the piqa metric class
    piqa_metric_class = piqa_metric_map[metric_name]

    # Return the wrapped metric
    return PiqaMetricWrapper(piqa_metric_class, **metric_params)

# Step 3: Example usage with OmegaConf
if __name__ == "__main__":
    # Define an OmegaConf configuration
    config_yaml = """
    metric:
      name: "SSIM"
      params:
        window_size: 11
        sigma: 1.5
    """
    config = OmegaConf.create(config_yaml)

    # Get the metric from the configuration
    metric = get_metric_from_config(config)

    # Example inputs (batch of images)
    preds = torch.rand(4, 3, 64, 64)  # Batch of 4 RGB images
    target = torch.rand(4, 3, 64, 64)

    # Update and compute the metric
    metric.update(preds, target)
    result = metric.compute()

    print(f"Metric Result: {result}")
