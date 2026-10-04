from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SoftSegmentTokenizer(nn.Module):
    """CPM: assign each feature-map position softly to K learnable tokens."""

    def __init__(
        self,
        feat_channels: int,
        num_tokens: int = 16,
        token_dim: int = 64,
        temperature: float = 0.7,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_tokens < 1 or token_dim < 1:
            raise ValueError("num_tokens and token_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.temperature = float(temperature)
        self.eps = float(eps)

        # Semantic features, event activity, and two spatial coordinates.
        self.key_proj = nn.Sequential(
            nn.Conv2d(feat_channels + 3, token_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(token_dim), token_dim),
            nn.GELU(),
            nn.Conv2d(token_dim, token_dim, kernel_size=1, bias=True),
        )
        self.token_queries = nn.Parameter(torch.empty(num_tokens, token_dim))
        self.token_bias = nn.Parameter(torch.zeros(num_tokens))
        nn.init.orthogonal_(self.token_queries)

    @staticmethod
    def _coordinate_maps(feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = feat_map.shape
        ys = torch.linspace(0.0, 1.0, h, device=feat_map.device, dtype=feat_map.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=feat_map.device, dtype=feat_map.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return xx.view(1, 1, h, w), yy.view(1, 1, h, w)

    def forward(
        self,
        response_map: torch.Tensor,
        feat_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = feat_map.shape
        response = F.interpolate(response_map, size=(h, w), mode="bilinear", align_corners=False)
        response_min = response.amin(dim=(-2, -1), keepdim=True)
        response_max = response.amax(dim=(-2, -1), keepdim=True)
        response = (response - response_min) / (response_max - response_min + self.eps)

        xx, yy = self._coordinate_maps(feat_map)
        xx = xx.expand(b, -1, -1, -1)
        yy = yy.expand(b, -1, -1, -1)
        keys = self.key_proj(torch.cat([feat_map, response, xx, yy], dim=1))
        keys = F.normalize(keys, dim=1, eps=self.eps)
        queries = F.normalize(self.token_queries, dim=1, eps=self.eps)
        logits = torch.einsum("bdhw,kd->bkhw", keys, queries)
        logits = logits / self.temperature + self.token_bias.view(1, -1, 1, 1)
        return F.softmax(logits, dim=1), response


class SegmentHypergraphBranch(nn.Module):
    """CPM tokens and competitive proximity/orientation hypergraph reasoning.

    Token geometry has 11 components: center x/y, area, width, height,
    aspect ratio, response mean/mass/RMS, axial orientation, and compactness.
    Both relations share the semantic affinity and propagation parameters.
    """

    def __init__(
        self,
        feat_channels: int,
        out_dim: int,
        num_tokens: int = 16,
        token_dim: int = 64,
        token_temperature: float = 0.7,
        num_relation_hyperedges: int | None = None,
        proximity_membership_temperature: float = 0.5,
        orientation_membership_temperature: float = 0.5,
        segment_feat_dim: int = 64,
        segment_hg_hidden_dim: int = 128,
        segment_hg_num_layers: int = 1,
        distance_scale: float = 0.28,
        angle_scale_deg: float = 25.0,
        scale_ratio: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.geom_dim = 11
        self.num_layers = int(segment_hg_num_layers)
        self.num_relation_hyperedges = (
            self.num_tokens if num_relation_hyperedges is None else int(num_relation_hyperedges)
        )
        if not 1 <= self.num_relation_hyperedges <= self.num_tokens:
            raise ValueError("num_relation_hyperedges must be between 1 and num_tokens.")
        if self.num_layers < 1:
            raise ValueError("segment_hg_num_layers must be positive.")
        if distance_scale <= 0 or angle_scale_deg <= 0 or scale_ratio <= 1:
            raise ValueError("Relation bandwidths must be positive and scale_ratio must exceed 1.")
        if proximity_membership_temperature <= 0 or orientation_membership_temperature <= 0:
            raise ValueError("Membership temperatures must be positive.")
        self.proximity_membership_temperature = float(proximity_membership_temperature)
        self.orientation_membership_temperature = float(orientation_membership_temperature)
        self.eps = float(eps)

        anchor_indices = torch.div(
            (torch.arange(self.num_relation_hyperedges) * 2 + 1) * self.num_tokens,
            2 * self.num_relation_hyperedges,
            rounding_mode="floor",
        ).clamp_max(self.num_tokens - 1)
        self.register_buffer("relation_anchor_indices", anchor_indices, persistent=False)

        self.tokenizer = SoftSegmentTokenizer(
            feat_channels=feat_channels,
            num_tokens=num_tokens,
            token_dim=token_dim,
            temperature=token_temperature,
            eps=eps,
        )
        self.learn_proj = nn.Linear(feat_channels, segment_feat_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(self.geom_dim + segment_feat_dim, segment_hg_hidden_dim),
            nn.GELU(),
            nn.Linear(segment_hg_hidden_dim, segment_hg_hidden_dim),
        )
        self.query_to_node = nn.Linear(token_dim, segment_hg_hidden_dim, bias=False)
        self.relation_dim = max(segment_hg_hidden_dim // 2, 1)
        self.relation_q = nn.Linear(segment_hg_hidden_dim, self.relation_dim, bias=False)
        self.relation_k = nn.Linear(segment_hg_hidden_dim, self.relation_dim, bias=False)
        self.edge_mlp = self._make_mlp(segment_hg_hidden_dim)
        self.node_mlp = self._make_mlp(segment_hg_hidden_dim)
        self.message_scale = nn.Parameter(torch.tensor(0.1))
        self.node_norm = nn.LayerNorm(segment_hg_hidden_dim)
        self.out_proj = nn.Linear(segment_hg_hidden_dim, out_dim)
        self.log_distance_scale = nn.Parameter(torch.tensor(math.log(distance_scale)))
        self.log_angle_scale = nn.Parameter(torch.tensor(math.log(math.radians(angle_scale_deg))))
        self.log_scale_ratio = math.log(scale_ratio)

    @staticmethod
    def _make_mlp(dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def _soft_geometry(
        self,
        assignments: torch.Tensor,
        response: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable geometry and unnormalized token masses."""
        _, _, h, w = assignments.shape
        p = h * w
        weights = assignments.flatten(2)
        mass = weights.sum(dim=-1).clamp_min(self.eps)

        ys = torch.linspace(0.0, 1.0, h, device=assignments.device, dtype=assignments.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=assignments.device, dtype=assignments.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        x = xx.reshape(1, 1, p)
        y = yy.reshape(1, 1, p)
        cx = (weights * x).sum(dim=-1) / mass
        cy = (weights * y).sum(dim=-1) / mass
        dx = x - cx.unsqueeze(-1)
        dy = y - cy.unsqueeze(-1)
        var_x = (weights * dx.square()).sum(dim=-1) / mass
        var_y = (weights * dy.square()).sum(dim=-1) / mass
        cov_xy = (weights * dx * dy).sum(dim=-1) / mass

        # Uniform-interval variance provides a differentiable extent estimate.
        width = torch.sqrt(12.0 * var_x + self.eps).clamp(max=1.0)
        height = torch.sqrt(12.0 * var_y + self.eps).clamp(max=1.0)
        area = mass / float(p)
        angle = 0.5 * torch.atan2(2.0 * cov_xy, var_x - var_y + self.eps)
        compactness = (area / (width * height + self.eps)).clamp(max=1.0)
        ratio = (width / (height + self.eps)).clamp(max=10.0)
        response_flat = response.flatten(2)
        weighted_response = weights * response_flat
        mean_response = weighted_response.sum(dim=-1) / mass
        response_mass = weighted_response.sum(dim=-1) / float(p)
        rms_response = torch.sqrt(
            (weights * response_flat.square()).sum(dim=-1) / mass + self.eps
        )
        orientation = (angle + math.pi / 2.0) / math.pi
        geometry = torch.stack(
            [cx, cy, area, width, height, ratio, mean_response, response_mass,
             rms_response, orientation, compactness],
            dim=-1,
        )
        return geometry, mass

    def _pool_token_features(
        self,
        assignments: torch.Tensor,
        feat_map: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        weights = assignments.flatten(2)
        features = feat_map.flatten(2).transpose(1, 2)
        pooled = torch.einsum("bkp,bpc->bkc", weights, features)
        return pooled / mass.unsqueeze(-1)

    @staticmethod
    def _axial_angle_difference(angles: torch.Tensor) -> torch.Tensor:
        difference = (angles.unsqueeze(2) - angles.unsqueeze(1)).abs()
        return torch.minimum(difference, math.pi - difference)

    def _learned_affinity(self, nodes: torch.Tensor) -> torch.Tensor:
        queries = self.relation_q(nodes)
        keys = self.relation_k(nodes)
        affinity = torch.einsum("bnd,bmd->bnm", queries, keys)
        affinity = torch.sigmoid(affinity / math.sqrt(self.relation_dim))
        eye = torch.eye(self.num_tokens, device=nodes.device, dtype=nodes.dtype).unsqueeze(0)
        return affinity * (1.0 - eye) + eye

    @staticmethod
    def _normalize_membership(incidence: torch.Tensor, temperature: float) -> torch.Tensor:
        """Make member nodes compete within each token-anchored hyperedge."""
        original_dtype = incidence.dtype
        incidence_float = incidence.float()
        logits = incidence_float.clamp_min(torch.finfo(torch.float32).tiny).log()
        normalized = F.softmax(logits / temperature, dim=1)
        return normalized.to(dtype=original_dtype)

    def _soft_incidence(
        self,
        nodes: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = geometry[..., :2]
        areas = geometry[..., 2].clamp_min(self.eps)
        distances = torch.cdist(centers, centers, p=2)
        distance_scale = self.log_distance_scale.exp().clamp(min=0.02, max=2.0)
        angle_scale = self.log_angle_scale.exp().clamp(min=math.radians(2.0), max=math.pi / 2.0)
        area_difference = (areas.log().unsqueeze(2) - areas.log().unsqueeze(1)).abs()
        proximity = torch.exp(-0.5 * (distances / distance_scale).square())
        proximity = proximity * torch.exp(-area_difference / self.log_scale_ratio)
        angles = geometry[..., 9] * math.pi
        angle_difference = self._axial_angle_difference(angles)
        orientation = torch.exp(-0.5 * (angle_difference / angle_scale).square())
        orientation = orientation * torch.exp(-0.5 * (distances / (1.5 * distance_scale)).square())

        shared_affinity = self._learned_affinity(nodes)
        anchor_indices = self.relation_anchor_indices.to(nodes.device)
        proximity = proximity.index_select(dim=2, index=anchor_indices)
        orientation = orientation.index_select(dim=2, index=anchor_indices)
        proximity_affinity = shared_affinity.index_select(dim=2, index=anchor_indices)
        orientation_affinity = shared_affinity.index_select(dim=2, index=anchor_indices)
        proximity = self._normalize_membership(
            proximity * proximity_affinity, self.proximity_membership_temperature
        )
        orientation = self._normalize_membership(
            orientation * orientation_affinity, self.orientation_membership_temperature
        )
        return proximity, orientation

    def _relation_message(self, nodes: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        edge_degree = incidence.sum(dim=1).clamp_min(self.eps)
        edge_features = torch.einsum("bne,bnd->bed", incidence, nodes)
        edge_features = self.edge_mlp(edge_features / edge_degree.unsqueeze(-1))
        node_degree = incidence.sum(dim=2).clamp_min(self.eps)
        node_messages = torch.einsum("bne,bed->bnd", incidence, edge_features)
        return self.node_mlp(node_messages / node_degree.unsqueeze(-1))

    def _hypergraph_aggregate(
        self,
        nodes: torch.Tensor,
        proximity: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        out = nodes
        for _ in range(self.num_layers):
            incidence = torch.cat([proximity, orientation], dim=-1)
            update = self.message_scale * self._relation_message(out, incidence)
            out = self.node_norm(out + update)
        return out

    def forward(self, response_map: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        assignments, response = self.tokenizer(response_map, feat_map)
        geometry, mass = self._soft_geometry(assignments, response)
        token_features = self._pool_token_features(assignments, feat_map, mass)
        token_features = self.learn_proj(token_features)
        nodes = self.node_proj(torch.cat([geometry, token_features], dim=-1))
        nodes = nodes + self.query_to_node(self.tokenizer.token_queries).unsqueeze(0)
        proximity, orientation = self._soft_incidence(nodes, geometry)
        nodes = self._hypergraph_aggregate(nodes, proximity, orientation)
        pooling_weights = mass / mass.sum(dim=1, keepdim=True).clamp_min(self.eps)
        graph_repr = (nodes * pooling_weights.unsqueeze(-1)).sum(dim=1)
        return self.out_proj(graph_repr)
