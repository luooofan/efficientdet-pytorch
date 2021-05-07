""" EfficientDet Focal, Huber/Smooth L1 loss fns w/ jit support

Based on loss fn in Google's automl EfficientDet repository (Apache 2.0 license).
https://github.com/google/automl/tree/master/efficientdet

Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple

from .anchors import decode_box_outputs
from .iou_loss import *


def focal_loss_legacy(logits, targets, alpha: float, gamma: float, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    'Legacy focal loss matches the loss used in the official Tensorflow impl for initial
    model releases and some time after that. It eventually transitioned to the 'New' loss
    defined below.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

         normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))

    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    return weighted_loss / normalizer


def new_focal_loss(logits, targets, alpha: float, gamma: float, normalizer, label_smoothing: float = 0.01):
    """Compute the focal loss between `logits` and the golden `target` values.

    'New' is not the best descriptor, but this focal loss impl matches recent versions of
    the official Tensorflow impl of EfficientDet. It has support for label smoothing, however
    it is a bit slower, doesn't jit optimize well, and uses more memory.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating loss from hard and easy examples.
        normalizer: Divide loss by this value.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    # compute focal loss multipliers before label smoothing, such that it will not blow up the loss.
    pred_prob = logits.sigmoid()
    targets = targets.to(logits.dtype)
    onem_targets = 1. - targets
    p_t = (targets * pred_prob) + (onem_targets * (1. - pred_prob))
    alpha_factor = targets * alpha + onem_targets * (1. - alpha)
    modulating_factor = (1. - p_t) ** gamma

    # apply label smoothing for cross_entropy for each entry.
    if label_smoothing > 0.:
        targets = targets * (1. - label_smoothing) + .5 * label_smoothing
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # compute the final loss and return
    return (1 / normalizer) * alpha_factor * modulating_factor * ce


def huber_loss(
        input, target, delta: float = 1., weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def smooth_l1_loss(
        input, target, beta: float = 1. / 9, weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        err = torch.abs(input - target)
        loss = torch.where(err < beta, 0.5 * err.pow(2) / beta, err - 0.5 * beta)
    if weights is not None:
        loss *= weights
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def _box_loss(box_outputs, box_targets, num_positives, delta: float = 0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = huber_loss(box_outputs, box_targets, weights=mask, delta=delta, size_average=False)
    return box_loss / normalizer


def one_hot(x, num_classes: int):
    # NOTE: PyTorch one-hot does not handle -ve entries (no hot) like Tensorflow, so mask them out
    x_non_neg = (x >= 0).unsqueeze(-1)
    onehot = torch.zeros(x.shape + (num_classes,), device=x.device, dtype=torch.float32)
    return onehot.scatter(-1, x.unsqueeze(-1) * x_non_neg, 1) * x_non_neg


class IouLoss(nn.Module):

    def __init__(self, losstype='Giou', reduction='mean'):
        super(IouLoss, self).__init__()
        self.reduction = reduction
        self.loss = losstype

    def forward(self, target_bboxes, pred_bboxes):
        num = target_bboxes.shape[0]
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - compute_iou(target_bboxes, pred_bboxes))
        elif self.loss == 'Giou':
            loss = torch.sum(1.0 - compute_g_iou(target_bboxes, pred_bboxes))
        elif self.loss == 'Diou':
            loss = torch.sum(1.0 - compute_d_iou(target_bboxes, pred_bboxes))
        else:
            loss = torch.sum(1.0 - compute_c_iou(target_bboxes, pred_bboxes))

        if self.reduction == 'mean':
            return loss / num
        else:
            return loss


def loss_fn(
        cls_outputs: List[torch.Tensor],
        box_outputs: List[torch.Tensor],
        cls_targets: List[torch.Tensor],
        box_targets: List[torch.Tensor],
        num_positives: torch.Tensor,
        anchors: List[torch.Tensor],
        positive_indices: List[torch.Tensor],
        num_classes: int,
        alpha: float,
        gamma: float,
        delta: float,
        box_loss_weight: float,
        label_smoothing: float = 0.,
        legacy_focal: bool = False,
        box_loss_type: str = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes total detection loss.
    Computes total detection loss including box and class loss from all levels.
    Args:
        cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
            at each feature level (index)

        box_outputs: a List with values representing box regression targets in
            [batch_size, height, width, num_anchors * 4] at each feature level (index)

        cls_targets: groundtruth class targets.

        box_targets: groundtrusth box targets.

        num_positives: num positive grountruth anchors

        anchors: a List with values representing anchors in
            [batch_size, height, width, num_anchors * 4] at each feature level (index)

        positive_indices: a List with values representing positive indices, which length is equal to batch_size

    Returns:
        total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

        cls_loss: an integer tensor representing total class loss.

        box_loss: an integer tensor representing total box regression loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = (num_positives.sum() + 1.0).float()
    levels = len(cls_outputs)

    cls_losses = []
    box_losses = []

    if box_loss_type is not None:
        box_loss_fc = IouLoss(losstype=box_loss_type, reduction=None)
    count = 0

    for l in range(levels):
        cls_targets_at_level = cls_targets[l]
        box_targets_at_level = box_targets[l]

        # Onehot encoding for classification labels.
        cls_targets_at_level_oh = one_hot(cls_targets_at_level, num_classes)

        bs, height, width, num_anchors_per_postion, _ = cls_targets_at_level_oh.shape
        cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
        cls_outputs_at_level = cls_outputs[l].permute(0, 2, 3, 1).float()
        if legacy_focal:
            cls_loss = focal_loss_legacy(
                cls_outputs_at_level, cls_targets_at_level_oh,
                alpha=alpha, gamma=gamma, normalizer=num_positives_sum)
        else:
            cls_loss = new_focal_loss(
                cls_outputs_at_level, cls_targets_at_level_oh,
                alpha=alpha, gamma=gamma, normalizer=num_positives_sum, label_smoothing=label_smoothing)
        cls_loss = cls_loss.view(bs, height, width, -1, num_classes)
        cls_loss = cls_loss * (cls_targets_at_level != -2).unsqueeze(-1)
        cls_losses.append(cls_loss.sum())   # FIXME reference code added a clamp here at some point ...clamp(0, 2))

        box_outputs_at_level = box_outputs[l].permute(0, 2, 3, 1).float()

        if box_loss_type is None:
            box_losses.append(_box_loss(
                box_outputs_at_level,
                box_targets_at_level,
                num_positives_sum,
                delta=delta))
        else:
            anchor_at_level = anchors[l]
            next_count = count + height*width*num_anchors_per_postion

            # select the positive_indices between count and next_count, which are belong to current level
            tmp = [indices[(indices >= count)] for indices in positive_indices]
            positive_indices_in_interval = ([indices[(indices < next_count)].long()-count for indices in tmp])

            loss = []
            for (pi, bo, bt, ac) in zip(positive_indices_in_interval, box_outputs_at_level, box_targets_at_level, anchor_at_level):
                # for one of the batch
                if pi.sum() > 0:
                    # if have positive_indices
                    valid_bo = bo.reshape(-1, 4)[pi]
                    valid_bt = bt.reshape(-1, 4)[pi]
                    valid_ac = ac.reshape(-1, 4)[pi]
                    bboxes2 = decode_box_outputs(valid_bo, valid_ac, output_xyxy=True)
                    bboxes1 = decode_box_outputs(valid_bt, valid_ac, output_xyxy=True)
                    losstmp = box_loss_fc(bboxes1, bboxes2)
                    loss.append(losstmp)

            if len(loss) > 0:
                box_losses.append(torch.stack(loss).mean())
            count = next_count

    # Sum per level losses to total loss.
    cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
    total_loss = cls_loss
    if len(box_losses) != 0:
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss += box_loss_weight * box_loss
    else:
        box_loss = torch.tensor([0], device=total_loss.device)
    return total_loss, cls_loss, box_loss


# loss_jit = torch.jit.script(loss_fn)


class DetectionLoss(nn.Module):

    __constants__ = ['num_classes']

    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight
        self.label_smoothing = config.label_smoothing
        self.legacy_focal = config.legacy_focal
        self.use_jit = config.jit_loss
        self.box_loss_type = config.box_loss_type

    def forward(
            self,
            cls_outputs: List[torch.Tensor],
            box_outputs: List[torch.Tensor],
            cls_targets: List[torch.Tensor],
            box_targets: List[torch.Tensor],
            num_positives: torch.Tensor,
            anchors: List[torch.Tensor],
            positive_indices: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        l_fn = loss_fn
        # if not torch.jit.is_scripting() and self.use_jit:
        #     # This branch only active if parent / bench itself isn't being scripted
        #     # NOTE: I haven't figured out what to do here wrt to tracing, is it an issue?
        #     l_fn = loss_jit

        return l_fn(
            cls_outputs, box_outputs, cls_targets, box_targets, num_positives, anchors, positive_indices,
            num_classes=self.num_classes, alpha=self.alpha, gamma=self.gamma, delta=self.delta,
            box_loss_weight=self.box_loss_weight, label_smoothing=self.label_smoothing, legacy_focal=self.legacy_focal, box_loss_type=self.box_loss_type)
