# MambaMark

**Zero-watermarking for vector geographic data using bidirectional Mamba and host-dominant contrastive fusion.**

MambaMark learns representations directly from point, polyline and polygon layers to link geographic data with copyright information. It constructs a registered zero-watermark without modifying the original map geometry. This repository provides the core models, training routines, registration and verification interface, and robustness and identity-discrimination evaluations.

## Method overview

### Hierarchical input organisation

A vector layer contains geographic features with different numbers of vertices. Two-level equidistant indexing selects features and vertices to form fixed-size model inputs while retaining the order of selected vertices. Local coordinates describe feature geometry, while normalised centroids and geometry types provide context for layer-level aggregation.

[View the two-level indexing diagram.](docs/assets/two-level-indexing.png)

### Bidirectional geometric representation learning

For polylines and polygon boundaries, a shared Mamba block processes vertex sequences in both traversal directions. Reverse outputs are aligned with the original sequence and combined through symmetric fusion and a residual connection. Masked pooling produces feature descriptors; point features use a coordinate-based multilayer perceptron. Gated aggregation combines these descriptors with spatial and geometry-type information into a layer representation.

Attack-oriented contrastive pre-training encourages representations of the same map to remain consistent under perturbations while separating different maps.

![Bidirectional Mamba encoding, gated aggregation and contrastive pre-training](docs/assets/bidirectional-mamba.png)

### Copyright fusion and zero-watermark verification

A ConvNeXt-based encoder extracts copyright-image features. Host-dominant fusion introduces this information through a bounded gated residual, using the vector representation as its base. With both encoders frozen, fusion training combines attacked-positive consistency with wrong-host and wrong-image constraints to learn stable, distinguishable map–copyright associations.

The fused representation is quantised into a binary feature code and combined with a binary copyright payload by exclusive OR (XOR) to construct the zero-watermark. Verification re-encodes a candidate map and copyright image, compares the continuous representations, and recovers the payload using the stored record. Normalised correlation (NC) measures representation similarity; bit-correct rate (BCR) measures payload recovery.

## Code structure

| Component | Main files and purpose |
| --- | --- |
| Models | `models/mamba_layer_encoder.py`, `image_encoder.py` and `fusion.py`: vector encoding, copyright-image encoding and gated fusion |
| Registration and verification | `system.py` integrates input preparation and model inference; `watermark_manager.py` saves registration records and verifies candidates |
| Training | `preprocess_data.py` prepares coordinate arrays; `training/` contains data loading, vector-encoder pre-training and fusion training |
| Attacks | `attacks/vector_attacks.py` implements geometric perturbations and geometry statistics |
| Evaluation | `evaluation/` provides geometric-attack, deletion/cropping and identity-discrimination evaluations |
| Checks and illustrations | `tests/` contains metric checks and a copyright-image evaluation shortcut; `docs/assets/` contains method diagrams |

## Data

The dataset is available on [Figshare](https://figshare.com/s/224d64446baa38735733). Download and extract the data, then set the input paths in the training or evaluation commands to match your local files. Keep all component files of each Shapefile together.

## Environment

Use Python 3.11 on Linux with an NVIDIA GPU. The dependencies include PyTorch 2.5.1, torchvision 0.20.1 and Mamba-SSM 2.2.4. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt --no-build-isolation
```

This example uses the CUDA 12.4 PyTorch build. Building Mamba from source requires a compatible CUDA toolkit and C++ compiler. See [requirements.txt](requirements.txt) for package versions.

## Basic usage

Register a vector layer with a copyright image and trained model weights:

```bash
python watermark_manager.py \
  --mamba-checkpoint weights/vector.pth \
  --image-checkpoint weights/image.pth \
  --fusion-checkpoint weights/fusion.pth \
  register \
  --layer maps/original.shp \
  --copyright-image images/logo.png \
  --registration results/map.npz
```

Replace the example paths with your own files. Weight directories and filenames are freely chosen; the three files must contain compatible vector-encoder, image-encoder and fusion weights. Keep the component files of each Shapefile together.

To verify a candidate, replace `register` with `verify`, change the layer path, and reuse the saved registration record and corresponding copyright image. The interface reports NC, BCR and an acceptance decision. The copyright image supplies the binary payload by default; a separate payload can be provided when needed.

### Training and evaluation

For training, prepare coordinate arrays with `preprocess_data.py`, pre-train the vector encoder with `training/train_mamba_encoder.py`, and train fusion with `training/train_fusion.py` using trained vector and image encoders.

The evaluation scripts cover three tasks:

- **Geometric robustness** — `evaluation/evaluate_final_model.py` tests coordinate noise, rotation, scaling, shearing and simplification, and includes a host-discrimination check.
- **Deletion and cropping robustness** — `evaluation/evaluate_structural_loss.py` tests feature deletion, vertex deletion and spatial cropping.
- **Identity discrimination** — `evaluation/evaluate_distinguishability.py` compares different hosts and copyright images through similarity matrices and false acceptance rates.

Choose the input layers, copyright images and model weights for your task, then save each evaluation to a separate output directory. Adapt the preprocessing filename filter and the host lists in evaluation scripts when using your own dataset. Each entry point provides its available options through `--help`, for example:

```bash
python -m evaluation.evaluate_distinguishability --help
```

## Licence

Released under the [MIT License](LICENSE). Third-party dependencies and input assets retain their respective licences.
