from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
import torch

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)


from mmseg.models.segmentors.encoder_decoder import EncoderDecoder as BaseEncoderDecoder

@MODELS.register_module()
class EncoderDecoderCL(BaseEncoderDecoder):
    def __init__(self, 
                pre_num_cls: int,
                new_num_cls: int,
                temperature: float = 1.0,
                cls_factor: float = 5.0,
                attn_factor: float = 10.0,
                zero_gt: bool = True,
                freeze_backbone: bool = True,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        # assert pre_num_cls + new_num_cls == kwargs["decode_head"]["num_classes"]
        # self.backbone_freeze = copy.deepcopy(self.backbone)
        # for param in self.backbone_freeze.parameters():
        #     param.requires_grad = False
        self.zero_gt = zero_gt
        decode_head_cfg = kwargs["decode_head"]
        decode_head_cfg["num_classes"] = pre_num_cls
        # self._init_decode_head_freeze(decode_head_cfg)
        self.pre_num_cls = pre_num_cls
        self.new_num_cls = new_num_cls
        self.first_iter = True
        self.orig_decode_head_cfg = decode_head_cfg
        self.temperature = temperature
        self.cls_factor = cls_factor
        self.attn_factor = attn_factor
        self.freeze_backbone = freeze_backbone


    def _init_decode_head_freeze(self, decode_head: ConfigType) -> None:
        self.decode_head_freeze = MODELS.build(decode_head)
        for param in self.decode_head_freeze.parameters():
            param.requires_grad = False

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # calculate the distill targets
        with torch.no_grad():
            # x_freeze = self.backbone_freeze(inputs)
            if self.freeze_backbone:
                x_freeze = self.extract_feat(inputs)
                
            start_cls = 0
            orig_results = {}
            for i_head in range(self.current_task + 1):
                if not self.freeze_backbone:
                    backbone = getattr(self, 'backbone_freeze_{}'.format(i_head))
                    x_freeze = backbone(inputs)

                head = getattr(self, 'decode_head_freeze_{}'.format(i_head))
                end_cls = head.loss_decode.criterion.ignore_range
                head_results = head.forward(x_freeze)
                cls_logits = head_results["pred_logits"][:,start_cls:end_cls]
                mask_logits = head_results["pred_masks"][:,start_cls:end_cls]
                if 'aux_outputs' in head_results:
                    orig_aux_results = []
                    for i, aux_outputs in enumerate(head_results["aux_outputs"]):
                        aux_outputs["pred_logits"] = aux_outputs["pred_logits"][:,start_cls:end_cls]
                        aux_outputs["pred_masks"] = aux_outputs["pred_masks"][:,start_cls:end_cls]
                        orig_aux_results.append(aux_outputs)

                if "pred_logits" not in orig_results:
                    orig_results["pred_logits"] = cls_logits
                    orig_results["pred_masks"] = mask_logits
                    if 'aux_outputs' in head_results:
                        orig_results["aux_outputs"] = orig_aux_results
                else:
                    orig_results["pred_logits"] = torch.cat((orig_results["pred_logits"], cls_logits), dim=1)
                    orig_results["pred_masks"] = torch.cat((orig_results["pred_masks"], mask_logits), dim=1)
                    if 'aux_outputs' in head_results:
                        for i, aux_outputs in enumerate(orig_results["aux_outputs"]):
                            aux_outputs["pred_logits"] = torch.cat((aux_outputs["pred_logits"], orig_aux_results[i]["pred_logits"]), dim=1)
                            aux_outputs["pred_masks"] = torch.cat((aux_outputs["pred_masks"], orig_aux_results[i]["pred_masks"]), dim=1)
                start_cls = end_cls
            # orig_results = self.decode_head_freeze(x_freeze)

        if self.freeze_backbone:
            x = x_freeze
        else:
            x = self.extract_feat(inputs)

        losses = dict()
        # normal mmsegmentation _decode_head_forward_train
        seg_logits = self.decode_head.forward(x)
        loss_decode = self.decode_head.loss_by_feat(seg_logits, data_samples)

        # distill loss
        loss_cls_diff = F.binary_cross_entropy_with_logits\
            (seg_logits["pred_logits"][:,:self.pre_num_cls], (orig_results["pred_logits"]* self.temperature).sigmoid(), reduction="mean")
        
        # elimate the current gt area
        gt = self.decode_head._stack_batch_gt(data_samples)
        gt = gt != self.decode_head.ignore_index
        pesudo_gt = (orig_results["pred_masks"] * self.temperature).sigmoid()
        pesudo_gt = F.interpolate(pesudo_gt, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        if self.zero_gt:
            pesudo_gt[gt.expand_as(pesudo_gt)] = 0.0
        pred_logits = seg_logits["pred_masks"][:,:self.pre_num_cls]
        pred_logits = F.interpolate(pred_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        
        loss_attn_diff = F.binary_cross_entropy_with_logits(pred_logits, pesudo_gt, reduction="mean")
        loss_decode["loss_cls_diff"] = loss_cls_diff * self.cls_factor
        loss_decode["loss_attn_diff"] = loss_attn_diff * self.attn_factor
        if 'aux_outputs' in orig_results:
            for i, aux_outputs in enumerate(orig_results["aux_outputs"]):
                loss_cls_diff = F.binary_cross_entropy_with_logits\
                    (seg_logits["aux_outputs"][i]["pred_logits"][:,:self.pre_num_cls], (aux_outputs["pred_logits"]* self.temperature).sigmoid(), reduction="mean")
                pesudo_gt = (aux_outputs["pred_masks"] * self.temperature).sigmoid()
                pesudo_gt = F.interpolate(pesudo_gt, size=gt.shape[-2:], mode="bilinear", align_corners=False)
                if self.zero_gt:
                    pesudo_gt[gt.expand_as(pesudo_gt)] = 0.0
                pred_logits = seg_logits["aux_outputs"][i]["pred_masks"][:,:self.pre_num_cls]
                pred_logits = F.interpolate(pred_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)

                loss_attn_diff = F.binary_cross_entropy_with_logits(pred_logits, pesudo_gt, reduction="mean")
                loss_decode["loss_cls_diff_aux{}".format(i)] = loss_cls_diff * self.cls_factor
                loss_decode["loss_attn_diff_aux{}".format(i)] = loss_attn_diff * self.attn_factor

        losses.update(add_prefix(loss_decode, 'decode'))

        return losses