# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import torch.nn.functional

from . import CityscapesDataset

import os
import numpy as np
import cv2
import os.path as osp
from collections import OrderedDict
from functools import reduce
import warnings
import scipy
from scipy import ndimage
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import measurements
import fastremap
import pickle as pkl
from scipy import io as sio

import mmcv
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from numba import njit
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations, LoadAnnotationsNpy
from .nuclei import get_dice_1, get_fast_pq, get_fast_aji, get_fast_aji_plus, remap_label, pair_coordinates
from .custom import CustomDataset


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    (might have issues at borders between cells, todo: check and fix)

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = ndimage.find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:
                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
                        msk[k] = ndimage.binary_fill_holes(msk[k])
                else:
                    msk = ndimage.binary_fill_holes(msk)
                masks[slc][msk] = (j + 1)
                j += 1
    return masks


@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 2D

    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)

    dP: float32, 3D array
        flows [axis x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j, 0]
            x = inds[j, 1]
            p0, p1 = int(p[0, y, x]), int(p[1, y, x])
            step = dP[:, p0, p1]
            for k in range(p.shape[0]):
                p[k, y, x] = min(shape[k] - 1, max(0, p[k, y, x] + step[k]))
    return p


def steps2D_interp(p, dP, niter):
    shape = dP.shape[1:]
    dPt = np.zeros(p.shape, np.float32)

    for t in range(niter):
        map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
        for k in range(len(p)):
            p[k] = np.minimum(shape[k] - 1, np.maximum(0, p[k] + dPt[k]))
    return p


@njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])',
       '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y

    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C, Ly, Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly - 1, max(0, yc_floor[i]))
        xf = min(Lx - 1, max(0, xc_floor[i]))
        yf1 = min(Ly - 1, yf + 1)
        xf1 = min(Lx - 1, xf + 1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c, i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                       np.float32(I[c, yf, xf1]) * (1 - y) * x +
                       np.float32(I[c, yf1, xf]) * y * (1 - x) +
                       np.float32(I[c, yf1, xf1]) * y * x)


def get_masks(p, iscell=None, rpad=20):
    """ create masks using pixel convergence after running dynamics

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.
    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims == 3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               np.arange(shape0[2]), indexing='ij')
        elif dims == 2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims == 3:
        expand = np.nonzero(np.ones((3, 3, 3)))
    else:
        expand = np.nonzero(np.ones((3, 3)))
    for e in expand:
        e = np.expand_dims(e, 1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter == 0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  # convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0


def remove_bad_flow_masks(masks, flows, threshold=0.4):
    """ remove masks which have inconsistent flows

    Uses metrics.flow_error to compute flows from predicted masks
    and compare flows to predicted flows from network. Discards
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """
    merrors, _ = flow_error(masks, flows)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def flow_error(maski, dP_net):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------

    maski: ND-array (int)
        masks produced from running dynamics on dP_net,
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float)
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks

    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return

    # flows predicted from estimated masks
    dP_masks = masks_to_flows(maski)
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += ndimage.mean((dP_masks[i] - dP_net[i] / 5.) ** 2, maski,
                                    index=np.arange(1, maski.max() + 1))

    return flow_errors, dP_masks


def masks_to_flows(masks):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask
        in which it resides

    """
    if masks.max() == 0:
        return np.zeros((2, *masks.shape), 'float32')

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y - 1, y + 1,
                           y, y, y - 1,
                           y - 1, y + 1, y + 1), axis=0)
    neighborsX = np.stack((x, x, x,
                           x - 1, x + 1, x - 1,
                           x + 1, x - 1, x + 1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)

    centers = np.zeros((masks.max(), 2), 'int')
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi, xi = np.nonzero(masks[sr, sc] == (i + 1))
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i, 0] = ymed + sr.start
            centers[i, 1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx,
                             n_iter=n_iter)

    # normalize
    mu /= (1e-20 + (mu ** 2).sum(axis=0) ** 0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0), dtype=np.float32)
    mu0[:, y - 1, x - 1] = mu
    mu_c = np.zeros_like(mu0)

    # _, ax = plt.subplots(2, 2, figsize=(11, 11))
    # ax[0][0].imshow(((mu0[0] + 1) * 127).astype(np.uint8), cmap="gray")
    # ax[0][1].imshow(((mu0[1] + 1) * 127).astype(np.uint8), cmap="gray")
    # ax[1][0].imshow(masks)
    # plt.show()
    return mu0


def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200):
    """ runs diffusion on GPU to generate flows for training images or quality control

    neighbors is 9 x pixels in masks,
    centers are mask centers,
    isneighbor is valid neighbor boolean 9 x pixels

    """
    nimg = neighbors.shape[0] // 9

    T = np.zeros((nimg, Ly, Lx), dtype=np.float32)
    meds = centers.astype(np.int64)
    for i in range(n_iter):
        T[:, meds[:, 0], meds[:, 1]] += 1
        Tneigh = T[:, neighbors[:, :, 0], neighbors[:, :, 1]]
        Tneigh *= isneighbor
        T[:, neighbors[0, :, 0], neighbors[0, :, 1]] = Tneigh.mean(axis=1)

    T = np.log(1. + T)
    # gradient positions
    grads = T[:, neighbors[[2, 1, 4, 3], :, 0], neighbors[[2, 1, 4, 3], :, 1]]

    dy = grads[:, 0] - grads[:, 1]
    dx = grads[:, 2] - grads[:, 3]
    del grads
    mu_torch = np.stack((dy.squeeze(), dx.squeeze()), axis=-2)
    return mu_torch


def follow_flows(dP, mask=None, niter=200, interp=True):
    """ define pixels and run dynamics to recover masks in 2D

    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D)
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    inds: int32, 3D or 4D array
        indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    p = np.array(p).astype(np.float32)

    inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T

    if inds.ndim < 2 or inds.shape[0] < 5:
        return p, None

    if not interp:
        p = steps2D(p, dP.astype(np.float32), inds, niter)

    else:
        p_interp = steps2D_interp(p[:, inds[:, 0], inds[:, 1]], dP, niter)
        p[:, inds[:, 0], inds[:, 1]] = p_interp
    return p, inds


def result_to_inst(result, p=None, niter=200, interp=True, flow_threshold=0.4,
                   min_size=15):
    """
    网络输出转换成instance map

    Args:
        result: results (numpy.array): 前两个通道是梯度，然后两个通道是前景背景，后N个通道是常规等的语义分割(包括背景)

    Returns:

    """
    # 2,h,w
    # h, w
    cellprob = torch.from_numpy(result[2:4, ...])
    cellprob = torch.nn.functional.softmax(cellprob, dim=0)
    cellprob = cellprob[1, ...].numpy()
    cp_mask = cellprob > 0.5

    # 2, h, w
    dp = result[:2, ...]
    if np.any(cp_mask):
        if p is None:
            p, inds = follow_flows(dp * cp_mask / 5., niter=niter, interp=interp)
            if inds is None:
                shape = cellprob.shape
                mask = np.zeros(shape, np.uint32)
                p = np.zeros((len(shape), *shape), np.uint32)
                return mask, p

        # calculate masks
        mask = get_masks(p, iscell=cp_mask)
        if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask = remove_bad_flow_masks(mask, dp, threshold=flow_threshold)
    else:
        shape = cellprob.shape
        mask = np.zeros(shape, np.uint32)
        p = np.zeros((len(shape), *shape), np.uint32)
        return mask, p

    mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)
    return mask, p


@DATASETS.register_module()
class NucleiCellPoseDataset(CustomDataset):
    CLASSES = ('bg', '1', '2', '3', '4')
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(NucleiCellPoseDataset, self).__init__(**kwargs)

    def get_gt(self):
        """Get ground truth for evaluation."""
        gts = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            results = sio.loadmat(seg_map)
            gts.append({
                "inst_map": results["inst_map"],
                "type_map": results["type_map"],
                "inst_type": results["inst_type"],
                "inst_centroid": results["inst_centroid"]
            })
        return gts

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gts=None,
                 **kwargs):
        """计算细胞核分割的指标，包括dice, aji, dq, sq, pq, aji_plus, 和分类的F1

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): 前两个通道是梯度，然后两个通道是前景背景，后N个通道是常规等的语义分割(包括背景,未softmax)
            metric (str | list[str]):
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gts (generator[dict]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        eval_results = {}

        if gts is None:
            gts = self.get_gt()

        # 测试miou
        if kwargs.get('miou') is True:
            results = np.stack(results, 0)  # N, 9, H, W
            seg_logits = results[:, 4:, ...]  # N, 5, H, W
            seg_probs = torch.softmax(torch.from_numpy(seg_logits), dim=1)
            seg_probs = seg_probs.numpy()
            pred = seg_logits.argmax(1)
            ret_metrics = eval_metrics(
                pred,
                [each["type_map"] for each in gts],
                num_classes=5,
                ignore_index=255,
                metrics=['mIoU'])

            class_names = tuple(range(5))

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })

            # each class table
            ret_metrics.pop('aAcc', None)
            ret_metrics_class = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            ret_metrics_class.update({'Class': class_names})
            ret_metrics_class.move_to_end('Class', last=False)

            # for logger
            class_table_data = PrettyTable()
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)

            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                if key == 'aAcc':
                    summary_table_data.add_column(key, [val])
                else:
                    summary_table_data.add_column('m' + key, [val])

            print_log('per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

            # each metric dict
            for key, value in ret_metrics_summary.items():
                if key == 'aAcc':
                    eval_results[key] = value / 100.0
                else:
                    eval_results['m' + key] = value / 100.0

            ret_metrics_class.pop('Class', None)
            for key, value in ret_metrics_class.items():
                eval_results.update({
                    key + '.' + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                })

            return eval_results

        if kwargs.get('count') is True:
            results = np.stack(results, 0)  # N, 9, H, W
            seg_logits = results[:, 4:, ...]  # N, 5, H, W
            seg_probs = torch.softmax(torch.from_numpy(seg_logits), dim=1)
            seg_probs = seg_probs.numpy()
            pred = seg_logits.argmax(1)
            # 降低 class1 阈值
            # pred[seg_probs[:, 1, :, :] > 0.35] = 1
            print_log("\nPred", logger)
            for i in range(5):
                eval_results[f"class_{i}"] = (pred == i).sum() / np.prod(pred.shape)
                print_log(f'class_{i}: {(pred == i).sum() / np.prod(pred.shape)}', logger)

            type_map = [each["type_map"] for each in gts]
            print_log(f"\n{type_map[0].shape}", logger)
            type_map = np.stack(type_map, 0)
            print_log(f"{type_map.shape}", logger)
            print_log("\nGT", logger)
            for i in range(5):
                print_log(f'class_{i}: {(type_map == i).sum() / np.prod(type_map.shape)}', logger)

            return eval_results

        num_classes = len(self.CLASSES)
        ret_metrics = self.run_nuclei_stat(results, gts)

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 3)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            summary_table_data.add_column(key, [val])

        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            eval_results[key] = value / 100.0

        return eval_results

    def run_nuclei_stat(self, results, gts):
        """
        计算各种指标，包括dice, aji, dq, sq, pq, aji_plus, 和分类的F1
        Args:
            results: results (list[numpy.Array]): 前两个通道是梯度，然后两个通道是前景背景，后N个通道是常规等的语义分割(包括背景)
            gts: list of dict("inst_map": ,
                             "type_map" ,
                            "inst_type",
                            "inst_centroid")
            metric:

        Returns:

        """
        paired_all = []  # unique matched index pair
        unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all = []  # each index is 1 independent data point
        pred_inst_type_all = []  # each index is 1 independent data point
        metrics = [[], [], [], [], [], []]
        for file_idx, (res, gt) in enumerate(zip(results, gts)):
            pred = result_to_inst(res)[0]
            # to ensure that the instance numbering is contiguous
            pred = remap_label(pred, by_size=False)  # 0,1,2,...N

            binary_map = pred.copy()
            binary_map[binary_map > 0] = 1
            label_idx = np.unique(pred)  # 0,1,2,...N

            # [(y,x), ...]
            inst_centroid_yx = measurements.center_of_mass(binary_map, pred, label_idx[1:])
            inst_centroid_xy = [(each[1], each[0]) for each in inst_centroid_yx]

            # 分类的指标
            # dont squeeze, may be 1 instance exist
            true_centroid = (gt["inst_centroid"]).astype("float32")
            true_inst_type = (gt["inst_type"]).astype("int32")
            if true_centroid.shape[0] != 0:
                true_inst_type = true_inst_type[:, 0]
            else:  # no instance at all
                true_centroid = np.array([[0, 0]])
                true_inst_type = np.array([0])

            # dont squeeze, may be 1 instance exist
            pred_centroid = np.array(inst_centroid_xy).astype("float32")
            pred_inst_type = self.get_inst_type(res, pred)
            if pred_centroid.shape[0] == 0:
                pred_centroid = np.array([[0, 0]])
                pred_inst_type = np.array([0])

            # ! if take longer than 1min for 1000 vs 1000 pairing, sth is wrong with coord
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroid, pred_centroid, 12
            )

            # * Aggreate information
            # get the offset as each index represent 1 independent instance
            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
            )
            true_inst_type_all.append(true_inst_type)
            pred_inst_type_all.append(pred_inst_type)

            # increment the pairing index statistic
            if paired.shape[0] != 0:  # ! sanity
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

            gt_ins_map = remap_label(gt["inst_map"], by_size=False)
            if gt_ins_map.max() == 0:
                continue

            # 分割的指标
            metrics[0].append(get_dice_1(gt_ins_map, pred))
            if pred.max() == 0:
                metrics[1].append(0.0)  # avoid some bug
            else:
                metrics[1].append(get_fast_aji(gt_ins_map, pred))

            pq_info = get_fast_pq(gt_ins_map, pred, match_iou=0.5)[0]
            metrics[2].append(pq_info[0])  # dq
            metrics[3].append(pq_info[1])  # sq
            metrics[4].append(pq_info[2])  # pq
            metrics[5].append(get_fast_aji_plus(gt_ins_map, pred))

        metrics = np.array(metrics)
        metrics_avg = np.mean(metrics, axis=-1)

        # 分类的指标
        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        # overall
        # * quite meaningless for not exhaustive annotated dataset
        w = [1, 1]
        tp_d = paired_pred_type.shape[0]
        fp_d = unpaired_pred_type.shape[0]
        fn_d = unpaired_true_type.shape[0]

        tp_tn_dt = (paired_pred_type == paired_true_type).sum()
        fp_fn_dt = (paired_pred_type != paired_true_type).sum()

        acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
        f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

        w = [2, 2, 1, 1]

        type_uid_list = np.unique(true_inst_type_all)
        if np.any(type_uid_list == 0):
            type_uid_list = type_uid_list[1:]  # 去掉0

        # 分类指标
        results_list = [f1_d, ]
        for type_uid in type_uid_list:
            f1_type = self._f1_type(
                paired_true_type,
                paired_pred_type,
                unpaired_true_type,
                unpaired_pred_type,
                type_uid,
                w,
            )
            results_list.append(f1_type)

        ret_metrics = {"DICE": metrics_avg[0],
                       "AJI": metrics_avg[1],
                       "DQ": metrics_avg[2],
                       "SQ": metrics_avg[3],
                       "PQ": metrics_avg[4],
                       "Detection": f1_d}
        for i in range(1, len(results_list)):
            ret_metrics[f"Fc{i}"] = results_list[i]
        return ret_metrics

    def get_inst_type(self, result, pred):
        """
        计算每个nuclei instance 的类别

        Args:
            result: (numpy.Array): 前两个通道是梯度，然后两个通道是前景背景，后N个通道是常规等的语义分割(包括背景)
            pred:  (numpy.Array): nuclei instance background: 0, nuclei: 1-N

        Returns: (numpy.Array [N,]) types of each nuclei

        """
        pred_type = torch.from_numpy(result[4:])
        pred_type = torch.argmax(pred_type, dim=0).numpy()  # H, W
        inst_id_list = np.unique(pred)[1:]  # exclude bg
        slices = ndimage.find_objects(pred)

        pred_inst_type = []  # 函数返回值，List[N]，每个instance的类型
        for i, slc in enumerate(slices):
            inst_map = pred[slc]
            inst_type = pred_type[slc]
            inst_type = inst_type[inst_map == (i + 1)]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))  # [(type, num_pixel),...]
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_inst_type.append(inst_type)
        return np.array(pred_inst_type, dtype=np.int32)

    def _f1_type(self, paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
                2 * (tp_dt + tn_dt)
                + w[0] * fp_dt
                + w[1] * fn_dt
                + w[2] * fp_d
                + w[3] * fn_d
        )
        return f1_type
