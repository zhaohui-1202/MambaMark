import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
GEOMETRY_TYPES = {'Point': 0, 'LineString': 1, 'Polygon': 2}

def set_seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def nt_xent(first, second, temperature=0.1):
    first, second = (F.normalize(first, dim=1), F.normalize(second, dim=1))
    both = torch.cat([first, second])
    logits = both @ both.T / temperature
    logits.fill_diagonal_(-torch.inf)
    count = len(first)
    labels = torch.arange(2 * count, device=both.device)
    labels = (labels + count) % (2 * count)
    return F.cross_entropy(logits, labels)

class LayerStore:

    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'manifest.json').read_text())
        self.by_type = {kind: [item for item in self.metadata if item['geometry_type'] == kind] for kind in GEOMETRY_TYPES}
        self.cache = {}

    def get(self, item):
        path = Path(item['output'])
        if not path.is_absolute():
            path = self.root / path
        key = str(path)
        if key not in self.cache:
            loaded = np.load(path)
            self.cache[key] = (loaded['coords'], loaded['offsets'])
        return self.cache[key]

def sample_layer(store, item, feature_count=32, vertex_count=64):
    coords, offsets = store.get(item)
    ids = np.random.choice(len(offsets) - 1, feature_count, replace=False)
    sequences, centroids = ([], [])
    kind = item['geometry_type']
    for index in ids:
        sequence = coords[offsets[index]:offsets[index + 1]].astype('float32')
        centroids.append(sequence.mean(0))
        sequence -= sequence.mean(0)
        radius = np.linalg.norm(sequence, axis=1).max() if len(sequence) > 1 else 1.0
        sequence /= max(radius, 1e-08)
        if kind == 'Point':
            sequence = np.zeros((1, 2), dtype='float32')
        positions = np.linspace(0, len(sequence) - 1, vertex_count).round().astype(int)
        sequences.append(sequence[positions])
    centroids = np.asarray(centroids, dtype='float32')
    centroids -= centroids.mean(0)
    centroids /= max(np.linalg.norm(centroids, axis=1).max(), 1e-08)
    return (np.stack(sequences), centroids, GEOMETRY_TYPES[kind])

def training_batch(store, feature_count=32, vertex_count=64):
    selected = []
    for kind in GEOMETRY_TYPES:
        selected.extend(random.sample(store.by_type[kind], 4))
    vertices, centroids, types = ([], [], [])
    for item in selected:
        vertex, centroid, kind = sample_layer(store, item, feature_count, vertex_count)
        vertices.append(vertex)
        centroids.append(centroid)
        types.append(np.full(feature_count, kind))
    device = 'cuda'
    vertices = torch.tensor(np.stack(vertices), device=device)
    centroids = torch.tensor(np.stack(centroids), device=device)
    types = torch.tensor(np.stack(types), device=device)
    vertex_mask = torch.ones(12, feature_count, vertex_count, dtype=torch.bool, device=device)
    feature_mask = torch.ones(12, feature_count, dtype=torch.bool, device=device)
    return (vertices, vertex_mask, centroids, types, feature_mask)
