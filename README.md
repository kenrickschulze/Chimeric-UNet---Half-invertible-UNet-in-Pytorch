# Chimeric UNet - Half invertible UNet in Pytorch
## Description
This repository contains the code to construct a Chimeric U-Net and to perform XAI analysis for a given model and dataset X. The Chimeric U-Net is a deep learning segmentation architecture with a non-invertible Encoder and an invertible Decoder with build-in explainability. For details see (REF TO PAPER).

Chimeric U-Net Schematic:
![Alt text](CNet-1.png?raw=true)

## Usage
### Construct the Chimeric U-Net

```python
from torch.utils.data import DataLoader
from segxai import WrapExp
from unet_models import cUNet, UNetConfig

model = cUNet(
    model_config=UNetConfig.chimMax([2, 2, 2, 2]),
    out_classes=...,
    in_channels=...,
    idwt=True,
)
```

### Train model...

### Wrap model in XAI Wrapper

``` python
model_xai = WrapExp(model_trained, n_channels=...)
```

### To construct a global embedding for your dataset X simply call...
```python 
embedding, emb_scores, X_coarse_grads, emb_labels, emb_idxs = model_xai.embedding(
    X,
    labels=...,
    pval=...,
    target_layer_idx=...,
    return_scores=True,
    return_vectorized=True,
    return_valid_labels=True,
    return_valid_idx=True,
)
```

### To create saliencies first extract target sample from DataLoader
```python
X: DataLoader = ...
x, _ = next(iter(X))
```


### ... and then compute the saliency map for the target class at positions specified on roi.
```python
saliencyMap = model_xai.salmap(x, target_cls=2, roi=...)

```