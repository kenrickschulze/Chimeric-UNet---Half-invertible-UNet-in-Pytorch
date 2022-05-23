from torch.utils.data import DataLoader
from segxai import WrapExp
from unet_models import cUNet, UNetConfig

if __name__ == "__main__":
    # # construct the Chimeric Unet
    model = cUNet(
        model_config=UNetConfig.chimMax([2, 2, 2, 2]),
        out_classes=...,
        in_channels=...,
        idwt=True,
    )

    # train the model on your dataset...

    # load dataset
    X: DataLoader = ...

    # wrap model in xai wrapper
    model_xai = WrapExp(model, n_channels=...)

    # # create global embedding
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

    # # create local saliency maps
    # extract sample to analyse
    x, _ = next(iter(X))
    saliencyM = model_xai.salmap(x, target_cls=2, roi=...)
