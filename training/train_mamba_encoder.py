import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.mamba_layer_encoder import NativeMultiGeometryEncoder, ProjectionHead
from training.data import GEOMETRY_TYPES, LayerStore, nt_xent, set_seed, training_batch

def augmented_view(raw):
    vertices, vertex_mask, centroids, types, feature_mask = raw
    angle = torch.rand(len(vertices), device='cuda') * 6.283
    cosine, sine = (angle.cos(), angle.sin())
    rotation = torch.stack([cosine, -sine, sine, cosine], -1).reshape(-1, 2, 2)
    vertices = torch.einsum('bkld,bdh->bklh', vertices, rotation) + torch.randn_like(vertices) * 0.001
    centroids = torch.einsum('bkd,bdh->bkh', centroids, rotation)
    retain = 0.5 if random.random() < 0.3 else 0.75
    keep = torch.rand_like(feature_mask.float()) < retain
    keep[:, 0] = True
    return (vertices, vertex_mask, centroids, types, keep)

def hard_negative_loss(features, geometry_types, margin=0.75):
    normalized = F.normalize(features, dim=1)
    similarity = normalized @ normalized.T
    same_type = geometry_types[:, None].eq(geometry_types[None, :])
    same_type.fill_diagonal_(False)
    return F.relu(similarity[same_type] - margin).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    store = LayerStore(args.data)
    encoder = NativeMultiGeometryEncoder().cuda()
    projector = ProjectionHead().cuda()
    parameters = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.0001)
    history, started = ([], time.time())
    for step in range(1, args.steps + 1):
        raw = training_batch(store)
        first_host = encoder(*augmented_view(raw))
        second_host = encoder(*augmented_view(raw))
        first_projection = projector(first_host)
        second_projection = projector(second_host)
        geometry_types = raw[3][:, 0]
        projector_loss = nt_xent(first_projection, second_projection)
        encoder_loss = nt_xent(first_host, second_host)
        hard_loss = 0.5 * (hard_negative_loss(first_host, geometry_types) + hard_negative_loss(second_host, geometry_types))
        loss = projector_loss + 0.5 * encoder_loss + 0.5 * hard_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {'step': step, 'loss': float(loss), 'projector_nt_xent': float(projector_loss), 'encoder_nt_xent': float(encoder_loss), 'hard_negative': float(hard_loss), 'seconds': time.time() - started, 'gpu_mb': torch.cuda.max_memory_allocated() / 2 ** 20}
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.save({'encoder': encoder.state_dict(), 'projector': projector.state_dict()}, args.output / 'mamba_encoder_final.pth')
    (args.output / 'training_metrics.json').write_text(json.dumps(history, indent=2))
    (args.output / 'training_protocol.json').write_text(json.dumps({'seed': 42, 'steps': args.steps, 'geometry_types': list(GEOMETRY_TYPES), 'loss': 'projector NT-Xent + 0.5 encoder NT-Xent + 0.5 same-type hard margin', 'margin': 0.75, 'test_leakage': False}, indent=2))
if __name__ == '__main__':
    main()
