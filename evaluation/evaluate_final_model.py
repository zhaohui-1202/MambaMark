import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from attacks.vector_attacks import ShapefileAttacker, bit_correct_rate, vertex_count
from models.fusion import HostDominantGatedFusion, load_state
from models.image_encoder import ImageEncoder
from models.mamba_layer_encoder import NativeMultiGeometryEncoder
from preprocess_data import coordinates, geometry_kind
from training.data import GEOMETRY_TYPES

def cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else 0.0

class FinalFingerprint:

    def __init__(self, mamba_checkpoint, image_checkpoint, fusion_checkpoint, copyright_image):
        self.mamba = NativeMultiGeometryEncoder().cuda().eval()
        self.mamba.load_state_dict(load_state(mamba_checkpoint, 'encoder'), strict=True)
        self.image_encoder = ImageEncoder(pretrained=False).cuda().eval()
        self.image_encoder.load_state_dict(load_state(image_checkpoint), strict=True)
        self.fusion = HostDominantGatedFusion().cuda().eval()
        self.fusion.load_state_dict(load_state(fusion_checkpoint, 'fusion'), strict=True)
        array = np.asarray(Image.open(copyright_image).convert('RGB').resize((224, 224)), dtype=np.float32) / 255
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            feature = self.image_encoder(tensor)
            self.image_feature = feature.mean(1) if feature.dim() == 3 else feature

    def encode(self, path, feature_count=64, vertex_count=64):
        layer = gpd.read_file(path)
        return self.encode_layer(layer, feature_count, vertex_count)

    def encode_layer(self, layer, feature_count=64, vertex_count=64):
        valid = [g for g in layer.geometry if g is not None and (not g.is_empty) and geometry_kind(g)]
        if not valid:
            return None
        kinds = [geometry_kind(g) for g in valid]
        kind = max(set(kinds), key=kinds.count)
        valid = [g for g in valid if geometry_kind(g) == kind]
        ids = np.arange(feature_count) % len(valid) if len(valid) < feature_count else np.linspace(0, len(valid) - 1, feature_count).round().astype(int)
        sequences, centroids = ([], [])
        for index in ids:
            sequence = coordinates(valid[index], kind).astype('float32')
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
        device = 'cuda'
        inputs = (torch.tensor(np.stack(sequences), device=device)[None], torch.ones(1, feature_count, vertex_count, dtype=torch.bool, device=device), torch.tensor(centroids, device=device)[None], torch.full((1, feature_count), GEOMETRY_TYPES[kind], device=device), torch.ones(1, feature_count, dtype=torch.bool, device=device))
        with torch.no_grad():
            host = self.mamba(*inputs)
            return self.fusion(host, self.image_feature).cpu().numpy()[0]

def attack_settings():
    settings = [('No Attack', 'noise', {'intensity': 0})]
    settings += [(f'Noise {value}', 'noise', {'intensity': value}) for value in [5e-05, 0.0001, 0.0002, 0.0003]]
    settings += [(f'Rotate {value} deg', 'rotate', {'angle': value}) for value in [15, 30, 45, 60, 90, 180]]
    settings += [(f'Scale {value}x', 'scale', {'xfact': value, 'yfact': value}) for value in [0.5, 0.8, 1.2, 1.5, 2.0]]
    settings += [(f'Shear {value}', 'shear', {'xs': value, 'ys': value}) for value in [0.1, 0.2, 0.3, 0.4, 0.5]]
    settings += [(f'Simplify {value}', 'simplify', {'tolerance': value}) for value in [1e-06, 5e-06, 1e-05, 5e-05]]
    return settings

def evaluate_robustness(manager, layer_dir, output, threshold=0.9):
    attacker = ShapefileAttacker(output / 'attacked_files')
    rows = []
    np.random.seed(42)
    all_paths = [path for path in sorted(layer_dir.glob('*.shp')) if 'attack' not in path.name]
    for path in all_paths:
        if 'attack' in path.name:
            continue
        layer = gpd.read_file(path)
        if len(layer) < 64:
            continue
        reference = manager.encode(path)
        if reference is None:
            continue
        before_features, before_vertices = (len(layer), vertex_count(layer))
        for attack_name, method, parameters in attack_settings():
            started = time.perf_counter()
            try:
                attacked_path = getattr(attacker, f'attack_{method}')(layer, **parameters)
                attacked = gpd.read_file(attacked_path)
                candidate = manager.encode_layer(attacked)
                nc = cosine(reference, candidate)
                rows.append({'file': path.name, 'geometry': str(layer.geom_type.mode().iloc[0]), 'attack': attack_name, 'nc': nc, 'bcr': bit_correct_rate(reference, candidate), 'passed': nc > threshold, 'verification_seconds': time.perf_counter() - started, 'features_before': before_features, 'features_after': len(attacked), 'vertices_before': before_vertices, 'vertices_after': vertex_count(attacked)})
            except Exception as error:
                rows.append({'file': path.name, 'geometry': 'ERROR', 'attack': attack_name, 'error': str(error), 'verification_seconds': time.perf_counter() - started})
    details = pd.DataFrame(rows)
    details.to_csv(output / 'robustness_by_file.csv', index=False)
    summary = details.dropna(subset=['nc']).groupby('attack').agg(Avg_NC=('nc', 'mean'), Avg_BCR=('bcr', 'mean'), Min_NC=('nc', 'min'), Max_NC=('nc', 'max'), Success_Rate=('passed', 'mean'), Avg_Verification_Seconds=('verification_seconds', 'mean'), Samples=('nc', 'size')).reset_index()
    summary['Success_Rate'] *= 100
    summary.to_csv(output / 'robustness_summary.csv', index=False)

def evaluate_distinguishability(manager, layer_dir, output, threshold=0.9):
    names = ['gis_osm_buildings_a_free_1.shp', 'gis_osm_water_a_free_1.shp', 'gis_osm_roads_free_1.shp', 'gis_osm_pois_free_1.shp']
    labels = ['buildings', 'water', 'roads', 'pois']
    vectors = [manager.encode(layer_dir / name) for name in names]
    matrix = np.asarray([[cosine(first, second) for second in vectors] for first in vectors])
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / 'distinguishability_matrix.csv')
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(matrix, annot=True, fmt='.4f', vmin=0, vmax=1, cmap='Blues', square=True, linewidths=0.8, linecolor='white', xticklabels=labels, yticklabels=labels, annot_kws={'fontsize': 11}, cbar_kws={'label': 'Normalized Correlation (NC)', 'shrink': 0.88}, ax=axis)
    axis.set_ylim(len(labels), 0)
    axis.set_title('Layer-level Distinguishability', pad=12, weight='bold')
    axis.set_xlabel('Candidate layer')
    axis.set_ylabel('Registered layer')
    axis.tick_params(axis='x', rotation=0)
    axis.tick_params(axis='y', rotation=0)
    figure.tight_layout()
    figure.savefig(output / 'distinguishability_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    nonmatches = matrix[np.triu_indices(4, 1)]
    (output / 'distinguishability_summary.json').write_text(json.dumps({'nonmatch_mean': float(nonmatches.mean()), 'nonmatch_max': float(nonmatches.max()), 'false_accepts': int((nonmatches >= threshold).sum()), 'pairs': 6, 'threshold': threshold}, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=Path, required=True)
    parser.add_argument('--mamba-checkpoint', type=Path, required=True)
    parser.add_argument('--image-checkpoint', type=Path, required=True)
    parser.add_argument('--fusion-checkpoint', type=Path, required=True)
    parser.add_argument('--copyright-image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(42)
    torch.manual_seed(42)
    manager = FinalFingerprint(args.mamba_checkpoint, args.image_checkpoint, args.fusion_checkpoint, args.copyright_image)
    evaluate_distinguishability(manager, args.layers, args.output)
    evaluate_robustness(manager, args.layers, args.output)
    (args.output / 'protocol.json').write_text(json.dumps({'seed': 42, 'threshold': 0.9, 'fingerprint': 'single 256-dimensional fused fingerprint', 'geometry_types': ['Point', 'LineString', 'Polygon'], 'suite': 'common geometric attacks and layer distinguishability', 'structural_loss': 'run evaluation/evaluate_structural_loss.py separately'}, indent=2))
if __name__ == '__main__':
    main()
