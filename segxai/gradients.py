import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
from typing import List, Union, Tuple

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.manifold import TSNE

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


from .xai_utils import *
from .xai_utils import _unpack_tuple


class WrapExp:
    def __init__(self, model, n_channels: int, pb_func_name: str = "pullback") -> None:
        """
        Args:
            model (nn.Module):
                Pytorch Model
            n_channels (int):
                Number of in channels, the model was trained on.
            pb_func_name (str, optional):
                Method call of pull-back function.
                Defaults: "pullback".
        """
        self.model = model
        self.model.eval()
        self.n_channels = n_channels
        self.pbfunc = getattr(self.model, pb_func_name)
        self._invertibility_check()

    def salmap(
        self,
        x: torch.tensor,
        target_cls: int,
        roi: Union[List[Tuple[int, int]], str],
        collaps_mode:str ="hadamard",
        h: float = 0.1,
        mesh: List[int] = [-2, -1, 1, -2],
        return_ori_sm: bool = False,
    ):
        """Creates saliency map for given sample x.

        Args:
            x (torch.tensor): Data sample, for which the saliency map is computed.
            target_cls (int):
                Class for which the gradients are be computed.
            roi (List[Tuple[int, int]] | str):
                List of coordinates to evaluate gradients. If set to "predicted", roi is build on
                coordinates of postive predictions for target_cls.
            collaps_mode (str):
                Defines how to collaps gradients and activations.
                Defaults: "hadamard"
            h (float, optional):
                Scaling of target logit. 
                Defaults: 0.1.
            mesh (List[int], optional):
                Intervall on which gradients are computed.
                Defaults: [-2, -1, 1, -2].
            return_ori_sm (bool, optional):
                If True, also returns saliency map prior to upsampling.
                Defaults: False.

        Returns:
            np.array: _description_
        """

        sm = []

        with torch.no_grad():
            out_fw_lgts = self.model(x)
            # transform prediction with softmax and remove batch dim
            out_fw_soft = torch_softmax(out_fw_lgts[0])
            out_fw_pred = out_fw_soft.argmax(0)

            # pullback without pertubation of y_c
            Z = self._pull_back(out_fw_lgts)
            z_y = Z[-1][0, :, :, :]

            num_der = torch.zeros_like(z_y)

            if roi == "predicted":
                roi = zip(*torch.where(out_fw_pred == target_cls))

            for x, y in roi:
                weight = out_fw_soft[target_cls, x, y].item()

                h_prime = h * out_fw_lgts[0, target_cls, x, y].item()

                approx_steps = []

                for h_point in mesh:
                    fw_copy = out_fw_lgts.clone()

                    # change pixel by amound of point we need to evaluate derivative
                    fw_copy[0, target_cls, x, y] += h_prime * h_point

                    # pull_back
                    z_delta_yc = self._pull_back(fw_copy)[-1][0, :, :, :]

                    grad_fstord = z_y - z_delta_yc
                    approx_steps.append(grad_fstord)

                num_der += self.gradients_fine(approx_steps, h_prime, weight)

        sm = collaps(z_y.numpy(), num_der.numpy(), method=collaps_mode)
        Z_avg = [z[0].numpy().sum(0) for z in Z]
        ret = (sequpsamp(sm, Z_avg),)

        if return_ori_sm:
            ret += (sm,)

        return _unpack_tuple(ret)

    def embedding(
        self,
        X: torch.utils.data.DataLoader,
        labels: List[int],
        pval: Union[int, float] = 0.3,
        target_layer_idx: int = 0,
        return_scores: bool = True,
        return_vectorized: bool = True,
        return_valid_labels: bool = True,
        return_valid_idx: bool = True,
        **kwargs_emb,
    ):
        """ Wrapper function to create embedding for X.

        Args:
            X (torch.utils.data.DataLoader): 
                DataLoader to create embedding.
            labels (List[int]): 
                List of labels present in the DataLoader. For each label, if it was predicted
                for a given sample, a pull-back is performed.
            pval (Union[int, float], optional): 
                Defines how much x should be perturbed. 
                If pval is given as int, pval is added to channel for positive predtion pixels.
                If pval is given as float, pval is multiplied with target channel
                for all logits greator 0.
                Defaults: 0.3.
            target_layer_idx (int, optional):
                Layer for which the embedding is computed. 
                For example Layer = 0, yields to the first skip_connection of the network,
                i.e. with the highest spatial resolution.
                Defaults: 0.
            return_scores (bool, optional): 
                If true, returns scores per sample per class in terms of IoU for positive predictions.
                Note, classes not predicted for a sample get np.nan assinged by default.
                Defaults: True.
            return_vectorized (bool, optional): 
                If true, returns extracted latent space. 
                Defaults: True.
            return_valid_labels (bool, optional):
                If true, returns labels for the embedding.
                Defaults: True.
            return_valid_idx (bool, optional): 
                If the model did not predict a class for sample x, np.nans are inserted in the vectorizing step
                and prior to the embedding filtered. 
                If true, returns non-nan idxs for unfiltered embedding.
                Defaults to True.

            **kwargs_emb (dict, optional):
                Provide parameters for t-SNE. 
                See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
                for possible parameters.
        """

        if target_layer_idx < 0:
            raise NotImplementedError("Negative layer indexing not supported")

        else:
            print("...create embedding")

        vectorized_X = self.vectorize(X, labels, pval, return_scores, target_layer_idx)

        if return_scores:
            vec, scores = vectorized_X
        else:
            vec = vectorized_X
            scores = None

        n_samples = len(X.dataset)
        vec_labels = np.repeat(labels, n_samples)

        # CxSxA -> (C*S)xA
        vec = self._flatten_vectorized(vec)

        # filter nans from negative predictions
        vec_filtered, valid_idx = self._filter_nans(vec, return_idx=True)

        vec_labels_filtered = vec_labels[valid_idx]

        ret = (self._embed(vec_filtered, **kwargs_emb),)

        if return_scores:
            scores = scores.ravel()[valid_idx]
            ret += (scores,)

        if return_vectorized:
            ret += (vec_filtered,)

        if return_valid_labels:
            ret += (vec_labels_filtered,)

        if return_valid_idx:
            ret += (valid_idx,)

        return _unpack_tuple(ret)

    def _embed(self, x:Union[List[np.array], np.array], **kwargs) -> list:
        """
        Args:
            x (Union[List[np.array], np.array]): 
            For multiple latent spaces, List containing the vectorized activations which are embedded,
            each given as (S*C)xD.
            For single embedding, only provide np.array as  (S*C)xD.
     
        """

        x_copy = x.copy()
        if type(x_copy) != list:
            x_copy = [x_copy]

        # normalize
        x_norm = [preprocessing.normalize(exp, norm="l2") for exp in x_copy]

        # stack experiments in order to embeb together
        # (EXP0+...+EXPn) x DIM
        x_norm = np.concatenate(x_norm)

        # shuffle data and store orginal order
        shuffle_idx, shuffle_idx_inv = self._shuffling_idxs(
            x_norm.shape[0], return_inverse=True
        )
        x_norm = x_norm[shuffle_idx]

        x_norm = TSNE(**kwargs).fit_transform(x_norm)

        # reconstruct original ordering
        x_norm = x_norm[shuffle_idx_inv]

        split_pos = np.cumsum([exp.shape[0] for exp in x_copy])
        x_copy = np.array_split(x_norm, split_pos)[:-1]

        return _unpack_tuple(x_copy)

    def vectorize(
        self,
        X,
        labels: List[int],
        pval: Union[int, float],
        return_scores: bool,
        tgl: int,
        **kwargs,
    ):

        n_samples = len(X.dataset)
        n_classes = len(labels)

        self.model.eval()
        with torch.no_grad():
            for sidx, (src, mask) in tqdm(enumerate(X), total=n_samples):
                if src.shape[0] > 1:
                    raise NotImplemented(
                        "Only a batch size of one is currently supported"
                    )
                out_fw_lgts = self.model(src)
                pred = logit2pred(out_fw_lgts)
                predicted_labels = torch.unique(pred)

                # pullback without pertubation of y_c
                z_y = self._pull_back(out_fw_lgts)[tgl][0, :, :, :]

                # get score of sample (class wise) for positive predictions
                score_y = iou_positives(pred, mask, labels)

                for lidx, label in enumerate(labels):
                    # compute coarse gradient approximation
                    if label in predicted_labels:
                        z_delta_yc = self._pull_back(out_fw_lgts, label, pval)[tgl][
                            0, :, :, :
                        ]
                        corase_grad = z_y - z_delta_yc
                    else:
                        # return shape to be filled with fill value in vectorizing step
                        corase_grad = z_y.numpy().shape

                    # vectorize coarse gradients
                    n_positives = int(
                        torch.ceil((pred == label).sum() / 4 ** (tgl + 1)).item()
                    )
                    vect_coarse_grad = self._vectorize_sample(corase_grad, n_positives)

                    if sidx + lidx == 0:
                        vcg_container = np.zeros(
                            (n_classes, n_samples, *vect_coarse_grad.shape)
                        )
                        score_container = np.zeros((n_classes, n_samples))

                    vcg_container[lidx, sidx, :] = vect_coarse_grad
                    score_container[lidx, sidx] = score_y[label]

            ret = (vcg_container,)
            if return_scores:
                ret += (score_container,)

            return _unpack_tuple(ret)

    def _pull_back(
        self, x: torch.Tensor, c: int = None, pval: Union[int, float, None] = None
    ):
        out = x.clone()
        if pval is not None:
            # if perturb value (pval) of type int: add it to channel c, where model predicted c on pixellvl
            if isinstance(pval, int):
                # -2 for 2 spatial dimensions (HxW)
                mask = (out[0].argmax(0) == c)
                out[0, c, mask] += pval

            elif isinstance(pval, float):
                # filter positive values
                mask = torch.where(out[0, c] > 0, True, False)
                out[0, c, mask] *= pval

        return self.pbfunc(out)

    def _vectorize_sample(self, x, n_positives: int = None, fill_val=np.nan):
        if isinstance(x, tuple):
            x = np.full(x[0], fill_val)
        else:
            x = x.numpy()
            n_channsels, h, w = x.shape

            x = x.reshape(n_channsels, -1)

            kth_max_actis = self._kthMax_alongAxis1d(x, n_positives)
            vectorized = np.nanmean(
                np.where(x >= kth_max_actis[:, None], x, np.nan),
                axis=1,
            )

            return vectorized

    @staticmethod
    def _kthMax_alongAxis1d(x, k: int):
        return x[np.arange(x.shape[0]), np.argsort(x)[:, -k]]

    @staticmethod
    def _flatten_vectorized(x: np.array) -> np.array:
        """CxSxA -> (C*S)xA with (c0_s0,c0_s1...cn_sn)

        Args:
            x (np.array):array to be flattend

        Returns:
            np.array: _description_
        """
        return x.reshape(np.prod(x.shape[: x.ndim - 1]), -1)

    @staticmethod
    def _shuffling_idxs(x: int, return_inverse: bool = False, seed: int = 7):
        shuf_idx = np.arange(x)
        np.random.seed(seed)
        np.random.shuffle(shuf_idx)

        ret = (shuf_idx,)

        if return_inverse:
            unshuf_idx = np.zeros_like(shuf_idx)
            unshuf_idx[shuf_idx] = np.arange(x)
            ret += (unshuf_idx,)

        return _unpack_tuple(ret)

    @staticmethod
    def _filter_nans(x, return_idx=False):
        valid_idx = ~np.isnan(x).any(axis=1)
        ret = (x[valid_idx],)
        if return_idx:
            ret += (valid_idx,)
        return _unpack_tuple(ret)

    def _invertibility_check(
        self,
    ) -> None:
        x = torch.ones((1, self.n_channels, 2**10, 2**10))
        out = self.model(x)

        bw = self._pull_back(out)[-1][0, :, :, :]
        fw = self.model.fw_skipconnections(x)[-1][0, :, :, :]

        diff = fw - bw
        diff_ratio = 100 * (diff.abs().sum().item() / fw.abs().sum().item())

        if diff_ratio > 1:
            sys.exit(f"Pull-back error tollerance of {diff_ratio:2f}% greater than 1%")

        print("... invertibility checked!")

    @staticmethod
    def knn_scoring(
        X_trn,
        X_tst,
        labels_trn,
        labels_tst,
        scores_trn_filt,
        n_neighbors: int = 10,
        return_dist=False,
        **kwargs_knn,
    ) -> np.array:

        """
        computes knn scoring for each layer of tst set, given layers and scores of trn set
        """
        knn_trn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="ball_tree", **kwargs_knn
        ).fit(X_trn)

        # get knn for tst sample within trainset
        dist, knn_idxs = knn_trn.kneighbors(X_tst)

        idx2label = labels_trn[knn_idxs]
        idx2score = scores_trn_filt[knn_idxs]

        # transform trn samples where label == tst_label to its IoU score
        labels2Iou = np.where(idx2label == labels_tst[:, np.newaxis], idx2score, np.nan)

        # sum TP IoU scores and devide by # of nn
        knn_scores = np.nansum(labels2Iou, axis=1) / n_neighbors

        ret = (knn_scores,)

        if return_dist:
            ret += (dist,)

        return _unpack_tuple(ret)

    @staticmethod
    def gradients_fine(layer, h_prime, weight=1) -> np.array:
        """numerical derivate for each layer

        Args:
            X ([type]): [description]
            h_prime ([type]): [description]

        Returns:
            [type]: [description]
        """
        num_der = (
            weight * (-layer[3] + 8 * layer[2] - 8 * layer[1] + layer[0]) / 12 * h_prime
        )

        return num_der
