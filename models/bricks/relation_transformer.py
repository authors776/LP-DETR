import copy
import functools
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional

from models.bricks.misc import Conv2dNormActivation
from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import get_sine_pos_embed
from util.misc import inverse_sigmoid

import torch.distributed as dist
import logging
import os


class RelationTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        hybrid_num_proposals: int = 900,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.hybrid_num_proposals = hybrid_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.hybrid_tgt_embed = nn.Embedding(hybrid_num_proposals, self.embed_dim)
        self.hybrid_class_head = nn.Linear(self.embed_dim, num_classes)
        self.hybrid_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        nn.init.normal_(self.hybrid_tgt_embed.weight)
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        nn.init.constant_(self.hybrid_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query=None,
        noised_box_query=None,
        attn_mask=None,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        topk, num_classes = self.two_stage_num_proposals, self.num_classes
        topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        topk = self.hybrid_num_proposals if self.training else 0
        if self.training:
            # get hybrid classes and coordinates, target and reference points
            hybrid_enc_class = self.hybrid_class_head(output_memory)
            hybrid_enc_coord = self.hybrid_bbox_head(output_memory) + output_proposals
            hybrid_enc_coord = hybrid_enc_coord.sigmoid()
            topk_index = torch.topk(hybrid_enc_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
            hybrid_enc_class = hybrid_enc_class.gather(
                1, topk_index.expand(-1, -1, self.num_classes)
            )
            hybrid_enc_coord = hybrid_enc_coord.gather(1, topk_index.expand(-1, -1, 4))
            hybrid_reference_points = hybrid_enc_coord.detach()
            hybrid_target = self.hybrid_tgt_embed.weight.expand(
                multi_level_feats[0].shape[0], -1, -1
            )
        else:
            hybrid_enc_class = None
            hybrid_enc_coord = None

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        if self.training:
            hybrid_classes, hybrid_coords = self.decoder(
                query=hybrid_target,
                value=memory,
                key_padding_mask=mask_flatten,
                reference_points=hybrid_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                skip_relation=True,
            )
        else:
            hybrid_classes = hybrid_coords = None

        return (
            outputs_classes,
            outputs_coords,
            enc_outputs_class,
            enc_outputs_coord,
            hybrid_classes,
            hybrid_coords,
            hybrid_enc_class,
            hybrid_enc_coord,
        )


class RelationTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim
        self.memory_fusion = nn.Sequential(
            nn.Linear((num_layers + 1) * self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None
    ):
        queries = [query]
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )
            queries.append(query)
        query = torch.cat(queries, -1)
        query = self.memory_fusion(query)
        return query


class RelationTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class RelationTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes, num_votes=16):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_votes = num_votes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        # multi-scale relation
        self.multi_scale_relation = WeightedLayerBoxRelationEncoder(16, self.num_heads, num_layers=self.num_layers)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initialize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
        skip_relation=False,
    ):
        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        if not skip_relation:
            pos_relation = self.multi_scale_relation(reference_points, 0).flatten(0, 1)
            if attn_mask is not None:
                pos_relation.masked_fill_(attn_mask, float("-inf"))

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            query_pos = query_pos * self.query_scale(query) if layer_idx != 0 else query_pos

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=pos_relation,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            # 偏移量预测
            output_coord = self.bbox_head[layer_idx](self.norm(query))
            output_coord = output_coord + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            if not skip_relation:
                tgt_boxes = output_coord
                pos_relation = self.multi_scale_relation(tgt_boxes, layer_idx + 1).flatten(0, 1)
                if attn_mask is not None:
                    pos_relation.masked_fill_(attn_mask, float("-inf"))

            # iterative bounding box refinement
            reference_points = inverse_sigmoid(reference_points.detach())
            reference_points = self.bbox_head[layer_idx](query) + reference_points
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


class RelationTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        n_relation_heads=4,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.num_relation_heads = n_relation_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = RelationAwareAttention(
            embed_dim, n_heads, n_relation_heads, dropout=dropout,batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
            need_weights=False,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed



def box_giou(boxes1, boxes2):
    """
    Calculate GIoU between two sets of boxes
    boxes1, boxes2: [B, N, 4] (x_c, y_c, w, h) format
    """
    # Convert from center format to min/max format
    boxes1_x0y0 = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_x1y1 = boxes1[..., :2] + boxes1[..., 2:] / 2
    boxes2_x0y0 = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_x1y1 = boxes2[..., :2] + boxes2[..., 2:] / 2

    # Calculate intersection
    intersect_mins = torch.max(boxes1_x0y0.unsqueeze(2), boxes2_x0y0.unsqueeze(1))
    intersect_maxs = torch.min(boxes1_x1y1.unsqueeze(2), boxes2_x1y1.unsqueeze(1))
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))

    # Calculate intersection area
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculate union area
    area1 = (boxes1[..., 2] * boxes1[..., 3]).unsqueeze(2)
    area2 = (boxes2[..., 2] * boxes2[..., 3]).unsqueeze(1)
    union_area = area1 + area2 - intersect_area

    # Calculate IoU
    iou = intersect_area / (union_area + 1e-6)

    # Calculate enclosing box
    enclose_mins = torch.min(boxes1_x0y0.unsqueeze(2), boxes2_x0y0.unsqueeze(1))
    enclose_maxs = torch.max(boxes1_x1y1.unsqueeze(2), boxes2_x1y1.unsqueeze(1))
    enclose_wh = torch.max(enclose_maxs - enclose_mins, torch.zeros_like(enclose_maxs))

    # Calculate enclosing area
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    # Calculate GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)

    return giou


def box_ciou(boxes1, boxes2):
    """
    Calculate CIoU between two sets of boxes
    boxes1, boxes2: [B, N, 4] (x_c, y_c, w, h) format
    """
    # Convert from center format to min/max format
    boxes1_x0y0 = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_x1y1 = boxes1[..., :2] + boxes1[..., 2:] / 2
    boxes2_x0y0 = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_x1y1 = boxes2[..., :2] + boxes2[..., 2:] / 2

    # Calculate intersection
    intersect_mins = torch.max(boxes1_x0y0.unsqueeze(2), boxes2_x0y0.unsqueeze(1))
    intersect_maxs = torch.min(boxes1_x1y1.unsqueeze(2), boxes2_x1y1.unsqueeze(1))
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))

    # Calculate intersection area
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculate union area
    area1 = (boxes1[..., 2] * boxes1[..., 3]).unsqueeze(2)
    area2 = (boxes2[..., 2] * boxes2[..., 3]).unsqueeze(1)
    union_area = area1 + area2 - intersect_area

    # Calculate IoU
    iou = intersect_area / (union_area + 1e-6)

    # Calculate enclosing box
    enclose_mins = torch.min(boxes1_x0y0.unsqueeze(2), boxes2_x0y0.unsqueeze(1))
    enclose_maxs = torch.max(boxes1_x1y1.unsqueeze(2), boxes2_x1y1.unsqueeze(1))
    enclose_wh = torch.max(enclose_maxs - enclose_mins, torch.zeros_like(enclose_maxs))

    # Calculate center distance
    center1 = boxes1[..., :2].unsqueeze(2)
    center2 = boxes2[..., :2].unsqueeze(1)
    center_distance = torch.sum((center1 - center2) ** 2, dim=-1)

    # Calculate diagonal distance of enclosing box
    enclose_diagonal = torch.sum(enclose_wh ** 2, dim=-1)

    # Calculate aspect ratio term
    w1 = boxes1[..., 2].unsqueeze(2)
    h1 = boxes1[..., 3].unsqueeze(2)
    w2 = boxes2[..., 2].unsqueeze(1)
    h2 = boxes2[..., 3].unsqueeze(1)

    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
    )

    # Calculate alpha term
    alpha = v / (1 - iou + v + 1e-6)

    # Calculate CIoU
    ciou = iou - (center_distance / (enclose_diagonal + 1e-6) + alpha * v)

    return ciou




class WeightedLayerBoxRelationEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.,
        scale=100.,
        activation_layer=nn.ReLU,
        inplace=True,
        num_layers=6,
    ):
        super().__init__()
        # initialize position projection
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 5,  # 5 = 2(distance) + 2(scale) + 1(gIoU)
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

        # multi-scale spatial realtion weights
        self.scale_weights = nn.Parameter(torch.ones(num_layers, 3))  # 3 = local, medium, global
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.register_buffer('layer_idx', torch.zeros(1, dtype=torch.long))
        self.eps = 1e-5
        self.register_buffer('print_counter', torch.zeros(1, dtype=torch.long))
        self.logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)

    def forward(self, src_boxes: Tensor, layer_idx: Optional[int] = None):
        tgt_boxes = src_boxes
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")

        # use current layer index or given layer index
        curr_layer = self.layer_idx.item() if layer_idx is None else layer_idx
        curr_layer = min(curr_layer, self.num_layers - 1)
        layer_ratio = curr_layer / (self.num_layers - 1)

        with torch.no_grad():
            xy1, wh1 = src_boxes.split([2, 2], -1)
            xy2, wh2 = tgt_boxes.split([2, 2], -1)

            med_scale = 1.0 + layer_ratio * 2.0

            # calculate relative distance
            delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
            relative_dist = delta_xy / (wh1.unsqueeze(-2) + self.eps)

            # calculate multi-scale distance
            local_distance = torch.log(relative_dist)
            medium_distance = torch.log(relative_dist / med_scale)
            global_distance = torch.ones_like(local_distance)

            # calculate scale
            wh_ratio = torch.log(
                (wh1.unsqueeze(-2) + self.eps) / (wh2.unsqueeze(-3) + self.eps))
            # calculate multi-scale scale
            local_scale = wh_ratio
            medium_scale = wh_ratio / med_scale
            global_scale = torch.ones_like(local_scale)

            # IoU
            iou = box_ciou(src_boxes, tgt_boxes)

            # combine multi-scale features
            local_features = torch.cat([
                local_distance,
                local_scale,
                iou.unsqueeze(-1)
            ], dim=-1)

            medium_features = torch.cat([
                medium_distance,
                medium_scale,
                iou.unsqueeze(-1)
            ], dim=-1)

            global_features = torch.cat([
                global_distance,
                global_scale,
                iou.unsqueeze(-1)
            ], dim=-1)

            local_pos = self.pos_func(local_features).permute(0, 3, 1, 2)
            medium_pos = self.pos_func(medium_features).permute(0, 3, 1, 2)
            global_pos = self.pos_func(global_features).permute(0, 3, 1, 2)

            # stack all scales [B, 3, embed_dim * 5, N, N]
            stacked_pos = torch.stack([local_pos, medium_pos, global_pos], dim=1)

        # apply learnable weights (ensure gradient propagation)
        weights = F.softmax(self.scale_weights[curr_layer], dim=0)  # [3]
        weights = weights.view(1, 3, 1, 1, 1)  # [1, 3, 1, 1, 1]

        # use einsum for weighted sum
        pos_embed = torch.einsum('bscdn,s->bcdn', stacked_pos, weights.squeeze())

        # project to final attention weights
        pos_embed = self.pos_proj(pos_embed)

        # update layer index
        if layer_idx is None:
            self.layer_idx += 1
            if self.layer_idx >= self.num_layers:
                self.layer_idx.zero_()

        # monitor weight changes
        if self.training and (dist.get_rank() == 0):
            self.print_counter += 1
            if self.print_counter % 100 == 0:  # print every 100 iterations
                weights = F.softmax(self.scale_weights, dim=1).detach().cpu().numpy()
                self.logger.info("\nCurrent weights for each layer:")
                for i in range(self.num_layers):
                    self.logger.info(f"Layer {i}: Local={weights[i,0]:.3f}, "
                          f"Medium={weights[i,1]:.3f}, "
                          f"Global={weights[i,2]:.3f}\n")

        return pos_embed.clone()



class RelationAwareAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_relation_heads,
        dropout=0.1,
        batch_first=True
    ):
        super().__init__()
        assert num_relation_heads > 0, "num_relation_heads must be greater than 0"
        assert num_relation_heads <= num_heads, "num_relation_heads cannot exceed num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_relation_heads = num_relation_heads
        self.num_normal_heads = num_heads - num_relation_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query,
        key,
        value,
        relation_weights=None,
        attn_mask=None,
        need_weights=False,
        skip_relation=False
    ):
        batch_size, num_queries = query.shape[:2]

        # project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # reshape to multi-head format
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # apply mask (if any)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        # according to skip_relation decide whether to apply relation_weights
        if not skip_relation and relation_weights is not None:
            if self.num_relation_heads == self.num_heads:
                # all heads are relation heads
                attn_weights = torch.exp(attn_weights) * relation_weights
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                # mix normal heads and relation heads
                # apply relation_weights to relation heads
                relation_attn = torch.exp(attn_weights[:, -self.num_relation_heads:]) * relation_weights
                relation_attn = relation_attn / (relation_attn.sum(dim=-1, keepdim=True) + 1e-6)

                # normal heads使用普通的softmax
                normal_attn = F.softmax(attn_weights[:, :self.num_normal_heads], dim=-1)

                # combine two attention weights
                attn_weights = torch.cat([normal_attn, relation_attn], dim=1)
        else:
            # all heads use normal softmax
            attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # calculate output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        output = self.norm(output)

        return output, None