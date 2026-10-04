# Third-party notices

## ConvNeXt V2

`models/backbones/convnextv2_backbone.py` contains a local adaptation of the ConvNeXt V2 architecture, including its block, LayerNorm, and global response normalization.

- Upstream: [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2).
- Copyright (c) Meta Platforms, Inc. and affiliates.
- Source definitions: [models/convnextv2.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py) and [models/utils.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py).
- The upstream README identifies code as MIT-licensed and ImageNet pretrained/fine-tuned weights as CC BY-NC. See the [upstream license section](https://github.com/facebookresearch/ConvNeXt-V2#license).
- The upstream [LICENSE](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE), including both notices, is retained in `licenses/ConvNeXt-V2-LICENSE.txt` (retrieved 2026-10-04).

Local changes expose stage-3 and global features, restrict the wrapper to Nano, retain checkpoint-compatible module names, and load an explicitly supplied initialization checkpoint. No upstream weights are redistributed here.

PyTorch, torchvision, timm, NumPy, Pillow, PyYAML, and tqdm are installed dependencies and retain their own licenses. Their source packages are not vendored in this release.

The authors' original Hyper-TSign code is covered by the repository's [MIT License](LICENSE). Upstream notices and the separately licensed pretrained weights retain their original terms.
