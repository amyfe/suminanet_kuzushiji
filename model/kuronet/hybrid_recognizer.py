"""Top-Level Stage-2-Modell, das alle Option-C-Bausteine zusammenführt.
Damit Training und Inferenz nicht mit lose herumliegenden Modulen arbeiten.

Enthält
frozen or partially trainable backbone
feature projector
roi pool encoder
roi refinement head
roi ordering
roi token projector
context encoder
decoder


Forward-Schema:
outputs = model(
    images,
    coarse_boxes=None or detector_outputs,
    orientations=None,
    targets=None,
    teacher_forcing_ratio=...
)
zB {
    "projected_feats": ...,
    "coarse_boxes": ...,
    "coarse_scores": ...,
    "refined_boxes": ...,
    "refine_scores": ...,
    "roi_mask": ...,
    "token_feats": ...,
    "context_feats": ...,
    "decoder_logits": ...,
    "attn_weights": ...,
    "aux_logits": ...,
}


Ziel:

alle Module sauber verbinden
kein alter Mischzustand
klarer Forward-Pfad

Wichtig:

Dieses Modell macht Stage 2 Hybrid Recognition
Stage 1 Detector kann dabei extern eingefroren oder eingebunden sein
ich baue den Detector hier bewusst mit ein, damit der Datenfluss geschlossen ist


ToDos: 
1. Ich habe detector innerhalb des Modells gelassen

Das ist praktisch, aber auch eine Designentscheidung.

Vorteil:

geschlossener Datenfluss
einfachere Inferenz

Nachteil:

Training kann unübersichtlicher werden, wenn Backbone/Detector eingefroren werden sollen

Das ist okay, solange du im Training sauber steuerst:

for p in model.backbone.parameters():
    p.requires_grad = False
for p in model.detector.parameters():
    p.requires_grad = False
2. ordered["aux_logits"] mit Dummy-Zeros

Das ist bewusst pragmatisch gelöst, damit sort_batch() immer denselben Tensorfluss bekommt.
Schön ist das nicht.

Sauberer wäre später:

sort_batch() so erweitern, dass None-Tensors optional behandelt werden

Aber für eine erste belastbare Version ist diese Lösung okay.

3. roi_mask bleibt die Basis-Maskierung

Im Moment wird Maskierung noch nicht durch refine_scores verschärft.
Das heißt:

alle coarse proposals bleiben grundsätzlich im Sequenzpfad
refine_scores sind erstmal nur Zusatzsignal

Das ist absichtlich konservativ.
Später kannst du überlegen:

low-quality ROIs vor dem Kontextencoder zu filtern
4. Noch keine GT-Matching-Logik drin

Das gehört bewusst nicht ins Modell, sondern später in:

stage2_targets.py
stage2_losses.py

Das Modell soll nur Forward machen

"""

# model/kuronet/hybrid_recognizer.py

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from model.kuronet.backbone.feature_projector import FeatureProjector
from model.kuronet.context.roi_context import ROIContextEncoder
from model.kuronet.decoder.attention import SeqDecoderAttention
from model.kuronet.detection.proposal_utils import extract_coarse_proposals
from model.kuronet.roi.roi_ordering import ROIReadingOrder
from model.kuronet.roi.roi_pool import ROIPoolEncoder
from model.kuronet.roi.roi_refinement import ROIRefinementHead
from model.kuronet.roi.roi_tokens import ROITokenProjector


class HybridKuroNetRecognizer(nn.Module):
    """
    Option-C hybrid recognizer.

    Pipeline:
        image
        -> shared backbone features
        -> detector coarse proposals
        -> feature projection
        -> ROI pooling
        -> ROI refinement
        -> reading-order sorting
        -> ROI token projection
        -> context encoder
        -> autoregressive decoder

    Notes:
    - Detector/backbone can be frozen externally during Stage 2 training.
    - This model expects per-sample orientation strings.
    """

    def __init__(
        self,
        backbone: nn.Module,
        detector: nn.Module,
        backbone_out_channels: int,
        vocab_size: int,

        proj_dim: int = 256,
        roi_size: tuple[int, int] = (8, 6),
        roi_feat_dim: int = 256,
        refine_hidden_dim: int = 256,
        token_dim: int = 256,
        token_hidden_dim: int | None = None,
        token_use_score_branch: bool = True,
        context_hidden_dim: int = 256,
        context_num_layers: int = 1,
        decoder_embed_dim: int = 128,
        decoder_hidden_dim: int = 256,

        det_score_thresh: float = 0.40,
        det_top_k: int = 256,
        det_nms_iou: float = 0.5,
        det_min_box_size: float = 1.0,

        use_aux_head: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = backbone
        self.detector = detector

        self.det_score_thresh = float(det_score_thresh)
        self.det_top_k = int(det_top_k)
        self.det_nms_iou = float(det_nms_iou)
        self.det_min_box_size = float(det_min_box_size)

        self.feature_projector = FeatureProjector(
            in_channels=backbone_out_channels,
            out_channels=proj_dim,
            hidden_channels=proj_dim,
            dropout=dropout,
        )

        self.roi_pool = ROIPoolEncoder(
            in_channels=proj_dim,
            roi_size=roi_size,
            conv_channels=proj_dim,
            out_dim=roi_feat_dim,
            dropout=dropout,
            predict_aux_logits=use_aux_head,
            vocab_size=vocab_size if use_aux_head else None,
        )

        self.roi_refine = ROIRefinementHead(
            feat_dim=roi_feat_dim,
            hidden_dim=refine_hidden_dim,
            dropout=dropout,
        )

        self.roi_order = ROIReadingOrder(line_merge_thresh_ratio=0.6)

        self.roi_tokens = ROITokenProjector(
            roi_feat_dim=roi_feat_dim,
            token_dim=token_dim,
            hidden_dim=int(token_hidden_dim) if token_hidden_dim is not None else token_dim,
            dropout=dropout,
            use_score_branch=bool(token_use_score_branch),
        )

        self.context_encoder = ROIContextEncoder(
            in_dim=token_dim,
            hidden_dim=context_hidden_dim,
            out_dim=context_hidden_dim,
            num_layers=int(context_num_layers),
            dropout=dropout,
            mode="bigru",
            use_layernorm=True,
            use_residual=True,
        )

        self.aux_head_context = None
        if use_aux_head:
            self.aux_head_context = nn.Sequential(
                nn.Linear(context_hidden_dim, refine_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(refine_hidden_dim, vocab_size),
            )

        self.decoder = SeqDecoderAttention(
            embed_dim=decoder_embed_dim,
            hidden_dim=decoder_hidden_dim,
            vocab_size=vocab_size,
            enc_dim=context_hidden_dim,
            num_layers=1,
            init_from_encoder=True,
            sampling_method="argmax",
            dropout=dropout,
            pointer_window_radius=2,
        )

        self.vocab_size = vocab_size
        self.use_aux_head = bool(use_aux_head)

    def _extract_detector_proposals(
        self,
        shared_feats: torch.Tensor,
        image_size: tuple[int, int],
    ):
        """
        Run detector and convert outputs into coarse proposals.

        Detector is expected to return:
            {
                "heatmap": (B,1,Hf,Wf),
                "bbox":    (B,4,Hf,Wf),
                ...
            }
        """
        det_out = self.detector(shared_feats)

        if "heatmap" not in det_out or "bbox" not in det_out:
            raise ValueError("Detector output must contain 'heatmap' and 'bbox'.")

        coarse_boxes_list, coarse_scores_list = extract_coarse_proposals(
            heat_logits=det_out["heatmap"],
            bbox_reg=det_out["bbox"],
            image_size=image_size,
            score_thresh=self.det_score_thresh,
            top_k=self.det_top_k,
            nms_iou=self.det_nms_iou,
            min_size=self.det_min_box_size,
        )

        return det_out, coarse_boxes_list, coarse_scores_list

    def encode_images(
        self,
        images: torch.Tensor,
        orientations: List[str],
        coarse_boxes_list: Optional[List[torch.Tensor]] = None,
        coarse_scores_list: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        if images.dim() != 4:
            raise ValueError(f"images must have shape (B, C, H, W), got {tuple(images.shape)}")

        bsz, _, h_img, w_img = images.shape
        image_size = (h_img, w_img)

        if len(orientations) != bsz:
            raise ValueError(
                f"orientations length {len(orientations)} does not match batch size {bsz}"
            )

        shared_feats = self.backbone(images)

        if coarse_boxes_list is None or coarse_scores_list is None:
            det_out, coarse_boxes_list, coarse_scores_list = self._extract_detector_proposals(
                shared_feats=shared_feats,
                image_size=image_size,
            )
        else:
            det_out = None

        proj_feats = self.feature_projector(shared_feats)

        roi_out = self.roi_pool(
            feat_2d=proj_feats,
            proposal_boxes_list=coarse_boxes_list,
            proposal_scores_list=coarse_scores_list,
            image_size=image_size,
        )

        roi_aux_logits = roi_out.get("aux_logits", None)

        refine_out = self.roi_refine(
            roi_feats=roi_out["roi_feats"],
            roi_boxes=roi_out["roi_boxes"],
            roi_scores=roi_out["roi_scores"],
            roi_mask=roi_out["roi_mask"],
            image_size=image_size,
        )

        ordered = self.roi_order.sort_batch(
            boxes=refine_out["refined_boxes"],
            mask=roi_out["roi_mask"],
            orientations=orientations,
            roi_feats=roi_out["roi_feats"],
            refined_feats=refine_out["refined_feats"],
            refine_scores=refine_out["refine_scores"],
            aux_logits=roi_aux_logits if roi_aux_logits is not None
            else torch.zeros(
                (*roi_out["roi_feats"].shape[:2], self.vocab_size),
                device=roi_out["roi_feats"].device,
                dtype=roi_out["roi_feats"].dtype,
            ),
        )

        ordered_aux_logits = ordered["aux_logits"] if self.use_aux_head else None

        token_feats, token_mask = self.roi_tokens(
            roi_feats=ordered["roi_feats"],
            refined_feats=ordered["refined_feats"],
            refined_boxes=ordered["boxes"],
            refine_scores=ordered["refine_scores"],
            roi_mask=ordered["mask"],
            image_size=image_size,
        )

        context_feats, context_mask = self.context_encoder(
            seq=token_feats,
            mask=token_mask,
        )

        aux_logits_with_context = None
        if self.aux_head_context is not None:
            aux_logits_with_context = self.aux_head_context(context_feats)

        decoder_token_bias = None
        if self.use_aux_head:
            if aux_logits_with_context is not None:
                decoder_token_bias = aux_logits_with_context
            elif ordered_aux_logits is not None:
                decoder_token_bias = ordered_aux_logits

        return {
            "shared_feats": shared_feats,
            "detector_outputs": det_out,
            "coarse_boxes_list": coarse_boxes_list,
            "coarse_scores_list": coarse_scores_list,
            "proj_feats": proj_feats,
            "roi_feats": roi_out["roi_feats"],
            "roi_local": roi_out["roi_local"],
            "roi_boxes": roi_out["roi_boxes"],
            "roi_scores": roi_out["roi_scores"],
            "roi_mask": roi_out["roi_mask"],
            "refined_feats": refine_out["refined_feats"],
            "refined_boxes": refine_out["refined_boxes"],
            "refine_scores": refine_out["refine_scores"],
            "box_deltas": refine_out["box_deltas"],
            "aux_logits": roi_aux_logits,
            "aux_logits_ordered": ordered_aux_logits,
            "aux_logits_with_context": aux_logits_with_context,
            "ordered_boxes": ordered["boxes"],
            "ordered_mask": ordered["mask"],
            "sort_indices": ordered["sort_indices"],
            "ordering_diagnostics": ordered.get("ordering_diagnostics", None),
            "token_feats": token_feats,
            "token_mask": token_mask,
            "context_feats": context_feats,
            "context_mask": context_mask,
            "decoder_token_bias": decoder_token_bias,
        }

    def decode_from_encoded(
        self,
        encoded: dict,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        teacher_prefix_steps: int = 0,
        input_seq: Optional[torch.Tensor] = None,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        max_len: Optional[int] = None,
        oracle_pointer_positions: Optional[torch.Tensor] = None,
        decode_constraints: Optional[dict] = None,
        force_full_rollout: bool = False,
    ) -> dict:
        decoder_logits, decoder_hidden, attn_weights, stop_logits, action_logis, pointer_positions, emitted_token_ids = self.decoder(
            enc_outputs=encoded["context_feats"],
            enc_mask=encoded["context_mask"],
            input_seq=input_seq,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            teacher_prefix_steps=teacher_prefix_steps,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_len,
            encoder_token_bias=encoded.get("decoder_token_bias", None),
            oracle_pointer_positions=oracle_pointer_positions,
            decode_constraints=decode_constraints,
            force_full_rollout=force_full_rollout,
        )

        outputs = dict(encoded)
        outputs.update({
            "decoder_logits": decoder_logits,
            "decoder_hidden": decoder_hidden,
            "attn_weights": attn_weights,
            "stop_logits": stop_logits,
            "action_logits": action_logis,
            "pointer_positions": pointer_positions,
            "emitted_token_ids": emitted_token_ids,
        })
        return outputs

    def forward(
        self,
        images: torch.Tensor,                     # (B, 3, H, W)
        orientations: List[str],
        targets: Optional[torch.Tensor] = None,  # (B, T_tgt)
        teacher_forcing_ratio: float = 0.0,

        # optional override for training/eval experiments:
        coarse_boxes_list: Optional[List[torch.Tensor]] = None,
        coarse_scores_list: Optional[List[torch.Tensor]] = None,

        # decoder control:
        input_seq: Optional[torch.Tensor] = None,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        max_len: Optional[int] = None,
        teacher_prefix_steps: int = 0,
        oracle_pointer_positions: Optional[torch.Tensor] = None,
        decode_constraints: Optional[dict] = None,
        force_full_rollout: bool = False,
    ) -> dict:
        encoded = self.encode_images(
            images=images,
            orientations=orientations,
            coarse_boxes_list=coarse_boxes_list,
            coarse_scores_list=coarse_scores_list,
        )
        return self.decode_from_encoded(
            encoded,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            teacher_prefix_steps=teacher_prefix_steps,
            input_seq=input_seq,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_len,
            oracle_pointer_positions=oracle_pointer_positions,
            decode_constraints=decode_constraints,
            force_full_rollout=force_full_rollout,
        )
