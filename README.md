# Overview

This project contains the official implementation of the NeurIPS 2025 conference paper [*Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/947b63838c90f1485188b9c673bc3a14-Abstract-Conference.html).

Project Website: https://semanticae.github.io.


# Quick Start Guide
1. Setup environment:
   ```bash
   # install torch based on your cuda version
   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 # (modify to your specific cuda version)
   
   # install other requirements
   pip install -r requirements.txt
   ```
2. Demo run:
    ```bash
   export PYTHONPATH=./:./sources_root
   python ./sources_root/resadv_ddim/masked_2d_generation.py
   ```
   Results:

   |                                  |          Exemplar Image           |      SemanticAE       |
   |:--------------------------------:|:---------------------:|:---------------------:|
   |             Content              | ![test_clean.png](test_clean.png) | ![test.png](test.png) |
   |             Resnet50             |        Jellyfish (0.7942)         |   Goldfish(0.9992)    |
   | ViT-B/16 <br/> (Transfer Attack) |        Jellyfish (0.7103)         |   Goldfish(0.4447)    |
3. Generate and evaluate ImageNet adversarial examples:
   
   ```bash
   cd sources_root/resadv_ddim
   bash run.sh
   ```
   Please refer to ```run.sh``` for details.
   In addition, we use surrogate & targe models defined in [BlackboxBench](https://github.com/cuhk-sz/blackboxbench), see [./sources_root/surrogate_models/models_blackboxbench](./sources_root/surrogate_models/models_blackboxbench) for details.


# Project Structure

This projects follows the structure of [https://github.com/hujinCN/aiworkflow/](https://github.com/hujinCN/aiworkflow/)

1. **resadv_ddim module**: Contains core adversarial attack generation algorithms
   - [model.py](sources_root/resadv_ddim/model.py) implements the basic diffusion model attack framework
   - [masked_2d_generation.py](sources_root/resadv_ddim/masked_2d_generation.py) extends the base model with mask mechanism & attack losses.
   - [evaluation](sources_root/resadv_ddim/evaluation/) directory contains test scripts

2. **image_evaluation module**: Responsible for evaluating generated adversarial examples
   - [datasets.py](sources_root/image_evaluation/datasets.py) defines dataset loaders
   - [metrics.py](sources_root/image_evaluation/metrics.py) implements various evaluation metrics
   - [evaluator.py](sources_root/image_evaluation/evaluator.py) is the main evaluation program

3. **workflow module**: Provides standardized project configuration and utilities
   - [standarization.py](workflow/standarization.py) handles configuration files and parameter parsing
   - Other auxiliary utility functions

4. **imagenet_analytics module**: Handles ImageNet labels and category information
   - Contains coarse-grained label definition files

5. **configs/semanticae module**: YAML format configuration files
   - Defines configuration parameters for different models and evaluation tasks

# Acknowledgements
Our code references the following projects:
* Diffusion Guided Adversarial Attacks: [SD-NAE](https://github.com/linyueqian/SD-NAE), [Adv-Diff](https://github.com/EricDai0/advdiff), [VENOM](https://github.com/huizhg/VENOM)
* Benchmarks: [BlackboxBench](https://github.com/SCLBD/BlackboxBench), [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack)
* The code in [sources_root/surrogate_models/models_blackboxbench](sources_root/surrogate_models/models_blackboxbench) is for evaluation purposes only and is licensed under CC BY-NC 4.0, see [LICENSE](sources_root/surrogate_models/models_blackboxbench/LICENSE).
* The code for 3D adversarial example Demo is based on the [Trellis project](https://microsoft.github.io/TRELLIS/), under MIT License.

# Citation

```
@inproceedings{
   hu2025exploring,
   title={Exploring Semantic-constrained Adversarial  Example with Instruction Uncertainty Reduction},
   author={Jin Hu and Jiakai Wang and Linna Jing and Haolin Li and Haodong Liu and Haotong Qin and Aishan Liu and Ke Xu and Xianglong Liu},
   booktitle = {Advances in Neural Information Processing Systems},
   editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
   pages = {102640--102692},
   publisher = {Curran Associates, Inc.},
   volume = {38},
   year = {2025}
}
```

# LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
