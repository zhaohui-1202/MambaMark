import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from models.fusion import HostDominantGatedFusion, load_state
from models.image_encoder import ImageEncoder
from models.mamba_layer_encoder import NativeMultiGeometryEncoder
from training.data import LayerStore, set_seed, training_batch

def deletion_view(raw):
    vertices, vertex_mask, centroids, types, feature_mask = raw
    keep = (torch.rand_like(feature_mask.float()) < random.choice([0.9, 0.8, 0.7, 0.6, 0.5])) & feature_mask
    keep[:, 0] = True
    return (vertices, vertex_mask, centroids, types, keep)

def noise_view(raw):
    vertices, vertex_mask, centroids, types, feature_mask = raw
    level = random.choice([0.0005, 0.001, 0.002, 0.005, 0.01])
    vertices = vertices + torch.randn_like(vertices) * level * vertex_mask.unsqueeze(-1)
    centroids = centroids + torch.randn_like(centroids) * (level * 0.25) * feature_mask.unsqueeze(-1)
    return (vertices, vertex_mask, centroids, types, feature_mask)

def simplification_view(raw):
    vertices, vertex_mask, centroids, types, feature_mask = raw
    length = vertices.shape[2]
    shortened = max(2, int(length * random.choice([0.9, 0.75, 0.5, 0.35, 0.25])))
    down = torch.linspace(0, length - 1, shortened, device=vertices.device).round().long()
    up = torch.linspace(0, shortened - 1, length, device=vertices.device).round().long()
    return (vertices[:, :, down][:, :, up], vertex_mask, centroids, types, feature_mask)

@torch.no_grad()
def encode_copyright_images(encoder, directory, count=100):
    paths = sorted(list(directory.glob('*.jpg')) + list(directory.glob('*.png')))[:count]
    if len(paths) < 2:
        raise RuntimeError('At least two copyright images are required')
    tensors = []
    for path in paths:
        array = np.asarray(Image.open(path).convert('RGB').resize((224, 224)), dtype=np.float32) / 255
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    outputs = []
    for part in torch.stack(tensors).split(16):
        encoded = encoder(part.cuda())
        outputs.append((encoded.mean(1) if encoded.dim() == 3 else encoded).cpu())
    return (paths, torch.cat(outputs))

def negative_margin(first, second, margin):
    return F.relu(F.cosine_similarity(first, second) - margin).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--mamba-checkpoint', type=Path, required=True)
    parser.add_argument('--image-checkpoint', type=Path, required=True)
    parser.add_argument('--copyright-images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    mamba = NativeMultiGeometryEncoder().cuda().eval()
    mamba.load_state_dict(load_state(args.mamba_checkpoint, 'encoder'), strict=True)
    for parameter in mamba.parameters():
        parameter.requires_grad_(False)
    image_encoder = ImageEncoder(pretrained=False).cuda().eval()
    image_encoder.load_state_dict(load_state(args.image_checkpoint), strict=True)
    for parameter in image_encoder.parameters():
        parameter.requires_grad_(False)
    image_paths, image_bank = encode_copyright_images(image_encoder, args.copyright_images)
    del image_encoder
    torch.cuda.empty_cache()
    store = LayerStore(args.data)
    fusion = HostDominantGatedFusion().cuda().train()
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=0.00015, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(42)
    history, started = ([], time.time())
    for step in range(1, args.steps + 1):
        raw = training_batch(store)
        with torch.no_grad():
            clean_host = mamba(*raw)
            deleted_host = mamba(*deletion_view(raw))
            noisy_host = mamba(*noise_view(raw))
            simplified_host = mamba(*simplification_view(raw))
        image_ids = torch.randint(len(image_bank), (len(clean_host),), generator=generator)
        image = image_bank[image_ids].cuda()
        clean = fusion(clean_host, image)
        deleted = fusion(deleted_host, image)
        noisy = fusion(noisy_host, image)
        simplified = fusion(simplified_host, image)
        wrong_host = fusion(clean_host.roll(1, 0), image)
        wrong_image = fusion(clean_host, image.roll(1, 0))
        delete_loss = (1 - F.cosine_similarity(clean, deleted)).mean()
        noise_loss = (1 - F.cosine_similarity(clean, noisy)).mean()
        simplify_loss = (1 - F.cosine_similarity(clean, simplified)).mean()
        host_negative = negative_margin(clean, wrong_host, 0.58)
        image_negative = negative_margin(clean, wrong_image, 0.88)
        gate_regularizer = (fusion.gate() - 0.18).square()
        loss = 12 * delete_loss + 18 * noise_loss + 18 * simplify_loss + 4 * host_negative + image_negative + gate_regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {'step': step, 'loss': float(loss), 'deletion_nc': float(F.cosine_similarity(clean, deleted).mean()), 'noise_nc': float(F.cosine_similarity(clean, noisy).mean()), 'simplification_nc': float(F.cosine_similarity(clean, simplified).mean()), 'wrong_host_nc': float(F.cosine_similarity(clean, wrong_host).mean()), 'gate': float(fusion.gate()), 'seconds': time.time() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.save({'fusion': fusion.state_dict()}, args.output / 'fusion_final.pth')
    (args.output / 'training_metrics.json').write_text(json.dumps(history, indent=2))
    (args.output / 'training_protocol.json').write_text(json.dumps({'seed': 42, 'steps': args.steps, 'mamba_frozen': True, 'image_encoder_frozen': True, 'single_fingerprint': True, 'copyright_image_count': len(image_paths), 'loss': '12*deletion + 18*noise + 18*simplification + 4*host_negative + image_negative', 'test_leakage': False}, indent=2))
if __name__ == '__main__':
    main()
