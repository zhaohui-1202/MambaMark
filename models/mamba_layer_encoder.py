import torch
from torch import nn
import torch.nn.functional as F

class NativeMultiGeometryEncoder(nn.Module):
    POINT, LINE, POLYGON = (0, 1, 2)

    def __init__(self, dim=256, depth=2, polygon_rolls=4):
        super().__init__()
        from mamba_ssm import Mamba
        self.dim, self.polygon_rolls = (dim, polygon_rolls)
        self.vertex = nn.Linear(2, dim)
        self.blocks = nn.ModuleList([Mamba(d_model=dim, d_state=16, d_conv=4, expand=2) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.centroid = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.point = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.type_embedding = nn.Embedding(3, dim)
        self.feature_gate = nn.Linear(dim, 1)
        self.output = nn.LayerNorm(dim)

    def _scan(self, sequence, vertex_mask):
        x = self.vertex(sequence)
        for norm, block in zip(self.norms, self.blocks):
            z = norm(x)
            x = x + 0.5 * (block(z) + block(z.flip(1)).flip(1))
        weights = vertex_mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(1) / weights.sum(1).clamp_min(1)

    def _polygon(self, sequence, vertex_mask):
        valid = vertex_mask.to(sequence.dtype).unsqueeze(-1)
        centre = (sequence * valid).sum(1) / valid.sum(1).clamp_min(1)
        radius = ((sequence - centre.unsqueeze(1)) ** 2).sum(-1)
        radius = radius.masked_fill(~vertex_mask.bool(), -torch.inf)
        starts = radius.argmax(1)
        canonical, canonical_mask = ([], [])
        for row, start in enumerate(starts.tolist()):
            canonical.append(sequence[row].roll(-start, 0))
            canonical_mask.append(vertex_mask[row].roll(-start, 0))
        return self._scan(torch.stack(canonical), torch.stack(canonical_mask))

    def forward(self, vertices, vertex_mask, centroids, geometry_types, feature_mask):
        b, k, length, _ = vertices.shape
        flat_v = vertices.reshape(b * k, length, 2)
        flat_vm = vertex_mask.reshape(b * k, length)
        flat_c = centroids.reshape(b * k, 2)
        flat_t = geometry_types.reshape(b * k)
        encoded = torch.zeros(b * k, self.dim, device=vertices.device, dtype=vertices.dtype)
        point_idx = flat_t == self.POINT
        sequence_idx = flat_t != self.POINT
        polygon_idx = flat_t == self.POLYGON
        line_idx = flat_t == self.LINE
        if point_idx.any():
            encoded[point_idx] = self.point(flat_c[point_idx])
        if line_idx.any():
            encoded[line_idx] = self._scan(flat_v[line_idx], flat_vm[line_idx])
        if polygon_idx.any():
            encoded[polygon_idx] = self._polygon(flat_v[polygon_idx], flat_vm[polygon_idx])
        encoded = encoded + self.centroid(flat_c) + self.type_embedding(flat_t)
        encoded = encoded.reshape(b, k, self.dim)
        valid = feature_mask.bool()
        logits = self.feature_gate(encoded).squeeze(-1).masked_fill(~valid, -torch.inf)
        weights = torch.softmax(logits, dim=1).nan_to_num().unsqueeze(-1)
        return self.output((encoded * weights).sum(1))

class ProjectionHead(nn.Module):

    def __init__(self, dim=256, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.BatchNorm1d(dim * 2), nn.GELU(), nn.Linear(dim * 2, out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
