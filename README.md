# Hyper-TSign

Main-model implementation of **Hyper-TSign: Geometry-Guided Hypergraph Modeling for Event-Based Traffic Sign Classification**.

This compact source package contains the EventTSign event-image loader, the ConvNeXt V2 Nano backbone, competitive primitive modeling (CPM), geometry-guided hypergraph reasoning, and training/evaluation entry points.

## Model

```text
3-channel event image [B, 3, 224, 224]
  -> ConvNeXt V2 Nano
     -> stage-3 features [B, 320, 14, 14]
        -> CPM: 16 soft tokens and 11-dimensional geometry
        -> proximity/orientation hyperedges with shared semantic affinity
        -> degree-normalized hypergraph propagation and weighted pooling
     -> final-stage global feature [B, 640]
  -> residual fusion -> 58-class classifier
```

The retained configuration uses 16 hyperedges per relation, competitive membership normalization, 64-dimensional token/semantic features, and one 128-dimensional reasoning layer. The implementation keeps the original internal `gahn` naming for checkpoint compatibility.

## Setup

Use Python 3.10 or later. Install a matching PyTorch/torchvision pair for your CPU or CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The commands below are shown from this directory. Relative configuration, dataset, backbone/checkpoint, and output paths resolve from this package directory, independently of the working directory. The package does not depend on files in the original research workspace.

## Dataset

Prepare the supplied EventTSign images and the two public split CSV files:

```text
datasets/eventtsign/
  train_labels.csv
  test_labels.csv
  train/
    rgb/
      train_low_000000.png
      ...
    event/
      train_low_000000.png
      ...
  test/
    rgb/
      test_high_000000.png
      ...
    event/
      test_high_000000.png
      ...
```

The public dataset retains exactly two CSV files, `train_labels.csv` and `test_labels.csv`, with these seven columns. Example training-row format:

```csv
sample_id,split,lighting,lighting_id,class_id,rgb_path,event_path
train_000000,train,low,0,0,train/rgb/train_low_000000.png,train/event/train_low_000000.png
```

Use the corresponding test rows in `test_labels.csv`, with `split=test` and paths under `test/`. Both image-path columns are relative to `data.root`, using forward slashes; source-machine paths and internal `orig_*` columns are omitted. The default `data.root` is `datasets/eventtsign`, relative to this package directory. The model reads only `event_path`; the paired `rgb_path` is retained in the public labels, and RGB files are unnecessary for this model. Existing CSV files with additional columns are accepted. `class_id` is used directly without remapping and must be in `[0, num_classes - 1]`. Lighting IDs follow the original loader: `0=Low`, `1=Normal`, `2=High`.

Images are converted to RGB, resized to 224 x 224, and converted to tensors in `[0, 1]`. There is no added normalization or augmentation. This is rendered event-image classification; no raw-event conversion or paired RGB input is required. Dataset images and original split CSVs are not bundled.

## Backbone initialization

The default configuration expects the official ConvNeXt V2 **Nano** ImageNet-22K checkpoint at:

```text
weights/convnextv2_nano_22k_224_ema.pt
```

Obtain it from the [official model table](https://github.com/facebookresearch/ConvNeXt-V2#imagenet-22k-fine-tuned-models), or set `model.pretrained_path` to your local file. The default initialization file is [convnextv2_nano_22k_224_ema.pt](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt). A missing or incompatible initialization checkpoint raises an error. `--no-pretrained` explicitly selects random initialization and is not equivalent to the pretrained experiment.

## Train

```bash
python train.py --config configs/hyper_tsign.yaml --data-root datasets/eventtsign
```

The supplied settings retain seed 42, 100 epochs, batch size 128, cross-entropy loss, AdamW (learning rate 5e-4, weight decay 0.05), five warmup epochs, and cosine scheduling. Use `--device cpu` or `--device cuda:0` to override the device. Adjust `data.workers` and `train.batch_size` for your machine; record such changes when comparing runs.

Outputs are saved under `outputs/hyper_tsign/`:

- `config.json`: resolved run configuration.
- `metrics.jsonl`: per-epoch loss, accuracy, lighting-group accuracy/counts, and learning rate.
- `last.pt`: last model, optimizer/scheduler states, configuration, and run metadata.
- `best.pt`: checkpoint with the highest test accuracy so far, using the original strict improvement rule. It is absent if accuracy never exceeds zero.

Each run requires an empty output directory. Change `experiment_name` or `output_dir` for another run; the script refuses to overwrite existing run files and does not implement training resumption.

**Selection protocol:** the inherited training script evaluates `test_labels.csv` every epoch and selects `best.pt` by test accuracy. This package preserves that behavior. It does not implement independent validation-set selection, and the selected test accuracy must not be described as an untouched held-out test estimate. The cleanup does not establish which historical run produced any paper result.

## Evaluate

```bash
python evaluate.py --checkpoint outputs/hyper_tsign/best.pt --data-root datasets/eventtsign --output outputs/evaluation.json
```

Evaluation uses the configuration saved in the checkpoint. For an older main-model checkpoint without embedded configuration, pass `--config configs/hyper_tsign.yaml` (also the fallback). Model weights load strictly. The separate backbone initialization file and training CSV are not needed for evaluation. Accuracies in JSON are fractions; printed lighting accuracies are percentages. A lighting group with zero samples has accuracy `0.0` and count `0`.

## Package scope

```text
configs/hyper_tsign.yaml
data/dataset.py
eventsign/config.py
eventsign/paths.py
models/eventsign_model.py
models/segment_hypergraph_branch.py
models/backbones/convnextv2_backbone.py
train.py
evaluate.py
```

Only the main ConvNeXt V2 Nano + CPM + geometry-guided hypergraph path is retained. Alternative backbones, baseline/comparison methods, component ablations, binary/structural filters, multi-view models, robustness experiments, visualization tools, N-Caltech101/raw-event adapters, cloud logging, and private manuscript/review files are excluded.

The backbone's unused original ImageNet head and shared-module aliases remain solely to preserve existing main-model checkpoint keys. They are not used to produce classification logits.

## Verification

Local CPU checks used Python 3.13.13, PyTorch 2.13.0+cpu, torchvision 0.28.0+cpu, and timm 1.0.30. Against the original main configuration, all 358 state-dictionary keys and shapes matched, seeded initialization was identical, and batch-2 224 x 224 evaluation/training outputs and input/active-parameter gradients had maximum absolute difference zero. Six synthetic images also produced identical loader tensors and labels. A one-epoch synthetic run completed training, checkpoint saving, and strict standalone evaluation with identical reloaded metrics. These are code-equivalence and integration checks, not full dataset training, CUDA validation, or paper-accuracy reproduction.

The cleaned seven-column labels were checked against all retained original fields: 12,380 training and 5,011 test samples, with all 34,782 RGB/event file references present. Real train/test batches loaded successfully. A separate one-epoch CPU check with two real images per split passed training, checkpoint saving, and standalone evaluation from outside the project directory. Evaluation also passed after copying the package, checkpoint, and test-only data to a new directory using the public relative dataset layout. These checks validate loading and path portability, not full training or accuracy reproduction.

## Release status and attribution

The authors' original Hyper-TSign code is released under the [MIT License](LICENSE). Trained Hyper-TSign weights, dataset files, and reproduced paper metrics are not included. Upstream ConvNeXt V2 attribution and its original notice are included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/ConvNeXt-V2-LICENSE.txt](licenses/ConvNeXt-V2-LICENSE.txt).
