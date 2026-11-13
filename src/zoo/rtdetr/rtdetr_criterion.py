"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import os

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register
from src.zoo.rtdetr.hierarchical_gated_encoder import PotentialMapGuidanceLoss, generate_potential_map_targets


_DEBUG_POTENTIAL_MAP = False
@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

        # optional potential map guidance loss helper
        if 'potential_map' in self.losses:
            self.pmg_loss = PotentialMapGuidanceLoss()
        else:
            self.pmg_loss = None


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # ensure target is float (BCE/focal expect float targets)
        target = target.to(dtype=src_logits.dtype)
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'potential_map': self.loss_potential_map,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_potential_map(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute potential map guidance loss when outputs include 'potentials'.
        Expects outputs['potentials'] to be a list of [B,H,W] tensors.
        Targets are expected to contain 'boxes' in either absolute xyxy or normalized cxcywh format and 'size' or 'orig_size'.
        """
        device = next(iter(outputs.values())).device if len(outputs) > 0 else torch.device('cpu')
        if self.pmg_loss is None or 'potentials' not in outputs:
            if _DEBUG_POTENTIAL_MAP:
                print('[DEBUG_PMG] pmg_loss is None or potentials not in outputs; returning 0')
            return {'loss_potential_map': torch.tensor(0.0, device=device)}

        potentials = outputs['potentials']

        if _DEBUG_POTENTIAL_MAP:
            try:
                shapes = [tuple(p.shape) for p in potentials]
            except Exception:
                shapes = str(type(potentials))
            print(f"[DEBUG_PMG] potentials present, count={len(potentials)}, shapes={shapes}")

        # construct boxes_list in absolute xyxy pixels
        boxes_list = []
        W, H = 1.0, 1.0
        for ti, t in enumerate(targets):
            boxes = t.get('boxes', None)
            if boxes is None or boxes.numel() == 0:
                boxes_list.append(torch.zeros((0,4), dtype=torch.float32))
                if _DEBUG_POTENTIAL_MAP:
                    print(f"[DEBUG_PMG] target[{ti}] has no boxes")
                continue

            size = t.get('size', t.get('orig_size', None))
            if size is not None:
                s = size
                try:
                    if isinstance(s, torch.Tensor):
                        s = s.detach().cpu().numpy()
                    if len(s) == 2:
                        H = float(s[0]); W = float(s[1])
                except Exception:
                    W, H = 1.0, 1.0

            boxes_cpu = boxes.detach().cpu().clone()
            if boxes_cpu.numel() == 0:
                boxes_list.append(torch.zeros((0,4), dtype=torch.float32))
                if _DEBUG_POTENTIAL_MAP:
                    print(f"[DEBUG_PMG] target[{ti}] boxes empty after cpu clone")
                continue

            # If boxes likely normalized (max<=1), assume cxcywh normalized
            if boxes_cpu.max() <= 1.0:
                cx = boxes_cpu[:,0] * W
                cy = boxes_cpu[:,1] * H
                ww = boxes_cpu[:,2] * W
                hh = boxes_cpu[:,3] * H
                x1 = cx - ww/2
                y1 = cy - hh/2
                x2 = cx + ww/2
                y2 = cy + hh/2
                abox = torch.stack([x1,y1,x2,y2], dim=1)
            else:
                # assume already xyxy absolute
                abox = boxes_cpu

            boxes_list.append(abox)
            if _DEBUG_POTENTIAL_MAP:
                print(f"[DEBUG_PMG] target[{ti}] box count: {abox.shape[0]}, sample_box: {abox[0] if abox.shape[0]>0 else 'N/A'}")

        feature_map_shapes = [(p.shape[1], p.shape[2]) for p in potentials]
        masks = generate_potential_map_targets(boxes_list, (W, H), feature_map_shapes)
        masks = [m.to(potentials[0].device).float() for m in masks]

        if _DEBUG_POTENTIAL_MAP:
            try:
                mask_sums = [int(m.sum().item()) for m in masks]
            except Exception:
                mask_sums = str(type(masks))
            print(f"[DEBUG_PMG] generated masks count={len(masks)}, mask_sums_per_level={mask_sums}")

        loss = 0.0
        per_level_losses = []
        for li, (p, m) in enumerate(zip(potentials, masks)):
            lval = float(self.pmg_loss(p, m).item())
            per_level_losses.append(lval)
            loss = loss + self.pmg_loss(p, m)
            if _DEBUG_POTENTIAL_MAP:
                print(f"[DEBUG_PMG] level {li} loss={lval}, p_stats(min/max/mean)={(float(p.min().item()), float(p.max().item()), float(p.mean().item()))}, mask_sum={int(m.sum().item())}")
        loss = loss / max(1, len(masks))
        if _DEBUG_POTENTIAL_MAP:
            print(f"[DEBUG_PMG] per_level_losses={per_level_losses}, averaged_loss={float(loss.item())}")
        return {'loss_potential_map': loss}

    # 把类 SetCriterion 中的 forward 方法替换为下面实现（直接修改文件内对应方法）

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            pot_all = outputs.get('potentials', None)
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # make a shallow copy dict to avoid mutating original model outputs
                if isinstance(aux_outputs, dict):
                    aux = dict(aux_outputs)
                    # attach per-layer potentials if present and missing in aux
                    if pot_all is not None and 'potentials' not in aux:
                        pot = pot_all
                        if isinstance(pot, (list, tuple)):
                            # If there is a per-level potential, attach that single level as list
                            if i < len(pot):
                                aux['potentials'] = [pot[i]]
                            else:
                                # fallback: attach the whole list for compatibility
                                aux['potentials'] = pot_all
                        else:
                            aux['potentials'] = pot_all
                else:
                    aux = aux_outputs

                indices = self.matcher(aux, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'potential_map':
                        # Potential-map is an image-level guidance computed from encoder outputs.
                        # It should be applied only to the main output (or a single selected layer),
                        # skipping it for aux outputs prevents the same supervision being counted
                        # repeatedly for each aux layer (which would effectively multiply its weight).
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            pot_all = outputs.get('potentials', None)
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # attach per-layer potentials to dn aux entries if present and missing
                if isinstance(aux_outputs, dict):
                    aux = dict(aux_outputs)
                    if pot_all is not None and 'potentials' not in aux:
                        pot = pot_all
                        if isinstance(pot, (list, tuple)):
                            if i < len(pot):
                                aux['potentials'] = [pot[i]]
                            else:
                                aux['potentials'] = pot_all
                        else:
                            aux['potentials'] = pot_all
                        if _DEBUG_POTENTIAL_MAP:
                            print(f"[DEBUG_PMG] attached potentials to dn_aux_outputs[{i}] -> type={type(aux['potentials'])}")
                else:
                    aux = aux_outputs

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'potential_map':
                        # skip potential_map for dn aux outputs as well
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
