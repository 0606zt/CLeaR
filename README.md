# 🎨CLeaR: A Unified Framework for Resolving the Leakage–Degradation Dilemma in Style Transfer

![intro](docs/intro.png)

## Introduction

**CLeaR** is a **C**ontent **Lea**kage **R**esistant framework for style transfer. It targets the leakage-degradation dilemma throughout the entire style transfer process: Orthogonal Subspace Projection for feature separation, Ensemble Inversion for feature-space grounding, and Energy-Guided Calibration for conditional diffusion generation.

> **Abstract** &ensp; Style transfer aims to render target content in the style of a reference image, but existing methods often suffer from content leakage, where objects, layouts, or semantics from the style reference appear in the generated output. Although prior data-driven and training-free methods can reduce leakage, they often face a leakage-degradation dilemma: stronger content suppression may weaken style fidelity, while richer style preservation may reintroduce unwanted reference content. We identify this dilemma across the full style-transfer pipeline, including feature separation, feature-space grounding, and diffusion generation. To address these issues, we propose CLeaR, a training-free framework for content-leakage-resistant style transfer. CLeaR first uses Orthogonal Subspace Projection to define content-reduced style targets in each vision foundation model (VFM) feature space. It then performs Ensemble Inversion, which optimizes a shared pixel-space style anchor satisfying style constraints across multiple VFMs. Finally, Energy-Guided Calibration maintains style alignment during diffusion sampling by steering the denoising trajectory toward the ensemble-defined style manifold. We further provide a theoretical analysis showing that the style-anchor estimation error decreases with the number of VFMs. Experiments on StyleBench demonstrate that CLeaR improves style alignment, reduces content leakage, and achieves better LLM-as-Judge evaluation compared with existing methods.

For more details, please visit our [paper page](https://arxiv.org/abs/2609.38136).

## Quick Start

**Configuration** &ensp; Install the required packages to set up the environment:

```bash
pip install -r requirements.txt
```

**Pre-trained Models** &ensp; Download the following weights, place them in the `/model` folder, and specify their paths in `config.json`:

- **Vision Encoders**: [CLIP](https://github.com/openai/CLIP), [Robust-CLIP](https://huggingface.co/chs20/FARE4-ViT-B-16-laion2B-s34B-b88K), [CSD-CLIP](https://github.com/learn2phoenix/CSD/), [DINO](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- **Image Generators**: [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [IP-Adapter](https://huggingface.co/h94/IP-Adapter)

**Generation** &ensp; We provide two modes: extracting a style anchor only, or running the full style transfer pipeline.

Extract a style anchor:

```bash
python -m style_extract \
    --vision_encoders robust_clip csd_clip dino inception vgg \
    --style_image /docs/style_image_example.png \
    --style_content_image /docs/style_content_image_example.png \
    --regular_weight 0.05 \
    --image_size 224 \
    --augment_batch 20 \
    --lr 1e-2 \
    --epochs 500 \
    --seed -1
```

Run the full pipeline:

```bash
python -m style_transfer \
    --image_generators sdxl \
    --vision_encoders robust_clip csd_clip dino inception vgg \
    --style_image /docs/style_image_example.png \
    --style_content_prompt "Sky with celestial bodies, tall tree on left, hills in background, village with church steeple in foreground." \
    --style_trans_prompt "A cat." \
    --ip_adapter_scale "{'down': {'block_2': [0.0, 0.5]}, 'mid': 0.5, 'up': {'block_0': [0.0, 1.0, 0.0]}}" \
    --regular_weight 0.05 \
    --image_size 224 \
    --augment_batch 20 \
    --invert_lr 1e-2 \
    --invert_epochs 300 \
    --calib_scale 0.1 \
    --calib_lr 1e-2 \
    --calib_epochs 5 \
    # 2/255 ≈ 0.007843
    --eps 0.007843 \
    --num_samples 2 \
    --seed -1
```