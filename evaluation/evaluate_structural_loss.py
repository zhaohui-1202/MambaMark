from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from attacks.vector_attacks import bit_correct_rate, geometry_family
from evaluation.evaluate_final_model import FinalFingerprint, cosine
RATIOS = tuple((index / 10 for index in range(1, 10)))
ATTACKS = ('feature_deletion', 'vertex_deletion', 'spatial_crop')

def stable_rng(filename: str, attack: str, ratio: float, seed: int) -> np.random.Generator:
    payload = f'{seed}|{filename}|{attack}|{ratio:.1f}'.encode('utf-8')
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
    return np.random.default_rng(value % 2 ** 32)

def vertex_count(geometry) -> int:
    if geometry is None or geometry.is_empty:
        return 0
    if geometry.geom_type == 'Point':
        return 1
    if geometry.geom_type == 'MultiPoint':
        return len(geometry.geoms)
    if geometry.geom_type in ('LineString', 'LinearRing'):
        return len(geometry.coords)
    if geometry.geom_type == 'Polygon':
        return len(geometry.exterior.coords) + sum((len(ring.coords) for ring in geometry.interiors))
    return sum((vertex_count(part) for part in getattr(geometry, 'geoms', ())))

def layer_vertex_count(layer: gpd.GeoDataFrame) -> int:
    return int(sum((vertex_count(geometry) for geometry in layer.geometry)))

def delete_open(coords, ratio: float, rng: np.random.Generator):
    coordinates = list(coords)
    if len(coordinates) <= 2:
        return coordinates
    delete_count = min(int(round(len(coordinates) * ratio)), len(coordinates) - 2)
    candidates = np.arange(1, len(coordinates) - 1)
    deleted = set(rng.choice(candidates, delete_count, replace=False).tolist()) if delete_count else set()
    return [coordinate for index, coordinate in enumerate(coordinates) if index not in deleted]

def delete_ring(coords, ratio: float, rng: np.random.Generator):
    unique = list(coords)[:-1]
    if len(unique) <= 3:
        return list(coords)
    delete_count = min(int(round(len(unique) * ratio)), len(unique) - 3)
    deleted = set(rng.choice(len(unique), delete_count, replace=False).tolist()) if delete_count else set()
    retained = [coordinate for index, coordinate in enumerate(unique) if index not in deleted]
    return retained + [retained[0]]

def delete_vertices(geometry, ratio: float, rng: np.random.Generator):
    if geometry is None or geometry.is_empty:
        return geometry
    if isinstance(geometry, LineString):
        return LineString(delete_open(geometry.coords, ratio, rng))
    if isinstance(geometry, MultiLineString):
        return MultiLineString([delete_vertices(part, ratio, rng) for part in geometry.geoms])
    if isinstance(geometry, Polygon):
        candidate = Polygon(delete_ring(geometry.exterior.coords, ratio, rng), [delete_ring(ring.coords, ratio, rng) for ring in geometry.interiors])
        return candidate if candidate.is_valid else geometry
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([delete_vertices(part, ratio, rng) for part in geometry.geoms])
    return geometry

def nested_delete_open(coords, ratio: float, seed: int):
    coordinates = list(coords)
    if len(coordinates) <= 2:
        return coordinates
    candidates = np.arange(1, len(coordinates) - 1)
    delete_count = min(int(round(len(coordinates) * ratio)), len(coordinates) - 2)
    order = np.random.default_rng(seed).permutation(candidates)
    deleted = set(order[:delete_count].tolist())
    return [coordinate for index, coordinate in enumerate(coordinates) if index not in deleted]

def nested_delete_ring(coords, ratio: float, seed: int):
    unique = list(coords)[:-1]
    if len(unique) <= 3:
        return list(coords)
    candidates = np.arange(len(unique))
    delete_count = min(int(round(len(unique) * ratio)), len(unique) - 3)
    order = np.random.default_rng(seed).permutation(candidates)
    deleted = set(order[:delete_count].tolist())
    retained = [coordinate for index, coordinate in enumerate(unique) if index not in deleted]
    return retained + [retained[0]]

def nested_delete_vertices(geometry, ratio: float, seed: int):
    if geometry is None or geometry.is_empty:
        return geometry
    if isinstance(geometry, LineString):
        return LineString(nested_delete_open(geometry.coords, ratio, seed))
    if isinstance(geometry, MultiLineString):
        return MultiLineString([nested_delete_vertices(part, ratio, seed + index * 1009) for index, part in enumerate(geometry.geoms)])
    if isinstance(geometry, Polygon):
        candidate = Polygon(nested_delete_ring(geometry.exterior.coords, ratio, seed), [nested_delete_ring(ring.coords, ratio, seed + (index + 1) * 1009) for index, ring in enumerate(geometry.interiors)])
        return candidate if candidate.is_valid else geometry
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([nested_delete_vertices(part, ratio, seed + index * 10007) for index, part in enumerate(geometry.geoms)])
    return geometry

def apply_attack(layer: gpd.GeoDataFrame, attack: str, ratio: float, filename: str, seed: int, vertex_feature_fraction: float=0.2) -> gpd.GeoDataFrame:
    rng = stable_rng(filename, attack, ratio, seed)
    if attack == 'feature_deletion':
        retained = max(1, int(len(layer) * (1.0 - ratio)))
        indices = np.sort(rng.choice(len(layer), retained, replace=False))
        return layer.iloc[indices].copy().reset_index(drop=True)
    if attack == 'vertex_deletion':
        eligible = [index for index, geometry in enumerate(layer.geometry) if geometry_family(geometry) in ('LineString', 'Polygon')]
        selected_count = max(1, int(round(len(eligible) * vertex_feature_fraction)))
        subset_rng = stable_rng(filename, 'vertex_feature_subset', vertex_feature_fraction, seed)
        selected = set(subset_rng.choice(eligible, selected_count, replace=False).tolist())
        attacked = layer.copy()
        attacked.geometry = [nested_delete_vertices(geometry, ratio, int.from_bytes(hashlib.sha256(f'{seed}|{filename}|vertex|{index}'.encode()).digest()[:8], 'little') % 2 ** 32) if index in selected else geometry for index, geometry in enumerate(layer.geometry)]
        return attacked
    if attack == 'spatial_crop':
        family = max((geometry_family(geometry) for geometry in layer.geometry if geometry_family(geometry)), key=lambda value: sum((geometry_family(geometry) == value for geometry in layer.geometry)))
        min_x, min_y, max_x, max_y = layer.total_bounds
        retained_scale = np.sqrt(1.0 - ratio)
        centre_x, centre_y = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        half_width = (max_x - min_x) * retained_scale / 2.0
        half_height = (max_y - min_y) * retained_scale / 2.0
        window = box(centre_x - half_width, centre_y - half_height, centre_x + half_width, centre_y + half_height)
        clipped = shapely.clip_by_rect(layer.geometry.to_numpy(), *window.bounds)
        attacked = layer.copy()
        attacked.geometry = gpd.GeoSeries(clipped, index=layer.index, crs=layer.crs)
        valid = ~attacked.geometry.is_empty & attacked.geometry.notna()
        attacked = attacked[valid].copy()
        attacked = attacked[attacked.geometry.map(lambda geometry: geometry_family(geometry) == family)]
        if attacked.empty:
            raise ValueError('Spatial cropping removed every supported feature')
        return attacked.reset_index(drop=True)
    raise ValueError(f'Unknown structural attack: {attack}')

def summarize(details: pd.DataFrame, groups: list[str], include_seconds: bool=False):
    aggregations = {'Avg_NC': ('nc', 'mean'), 'Avg_BCR': ('bcr', 'mean'), 'Min_NC': ('nc', 'min'), 'Pass_Rate': ('passed', 'mean'), 'Avg_Actual_Feature_Loss': ('actual_feature_loss', 'mean'), 'Avg_Actual_Vertex_Loss': ('actual_vertex_loss', 'mean'), 'Avg_Edited_Feature_Fraction': ('edited_feature_fraction', 'mean'), 'Samples': ('nc', 'size')}
    if include_seconds:
        aggregations['Avg_Seconds'] = ('seconds', 'mean')
    result = details.groupby(groups, sort=False).agg(**aggregations).reset_index()
    result['Pass_Rate'] *= 100.0
    return result

def write_outputs(details: pd.DataFrame, output: Path, protocol: dict) -> None:
    details = details.sort_values(['attack', 'requested_ratio', 'file']).reset_index(drop=True)
    details.to_csv(output / 'structural_loss_by_file.csv', index=False)
    summarize(details, ['attack', 'requested_ratio'], True).to_csv(output / 'structural_loss_summary.csv', index=False)
    summarize(details, ['attack', 'geometry', 'requested_ratio']).to_csv(output / 'structural_loss_by_geometry.csv', index=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')

def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = FinalFingerprint(args.mamba_checkpoint, args.image_checkpoint, args.fusion_checkpoint, args.copyright_image)
    paths = [path for path in sorted(args.layers.glob('*.shp')) if 'attack' not in path.name]
    paths = paths[args.shard_index::args.shard_count]
    rows = []
    for path in paths:
        layer = gpd.read_file(path)
        layer = layer[~layer.geometry.is_empty & layer.geometry.notna()].reset_index(drop=True)
        if len(layer) < 64:
            continue
        geometry = max(layer.geom_type.value_counts().index, key=lambda value: int((layer.geom_type == value).sum()))
        reference = model.encode_layer(layer)
        before_features = len(layer)
        before_vertices = layer_vertex_count(layer)
        eligible_vertex_features = sum((geometry_family(geometry) in ('LineString', 'Polygon') for geometry in layer.geometry))
        edited_vertex_features = max(1, int(round(eligible_vertex_features * args.vertex_feature_fraction))) if eligible_vertex_features else 0
        for attack in args.attacks:
            if attack == 'vertex_deletion' and 'Point' in geometry:
                continue
            for ratio in RATIOS:
                attacked = apply_attack(layer, attack, ratio, path.name, args.seed, args.vertex_feature_fraction)
                after_features = len(attacked)
                after_vertices = layer_vertex_count(attacked)
                started = time.perf_counter()
                candidate = model.encode_layer(attacked)
                elapsed = time.perf_counter() - started
                score_nc = cosine(reference, candidate)
                rows.append({'file': path.name, 'geometry': geometry, 'attack': attack, 'sampling': 'equidistant_index', 'requested_ratio': ratio, 'features_before': before_features, 'features_after': after_features, 'vertices_before': before_vertices, 'vertices_after': after_vertices, 'actual_feature_loss': 1.0 - after_features / before_features, 'actual_vertex_loss': 1.0 - after_vertices / before_vertices, 'edited_feature_fraction': edited_vertex_features / eligible_vertex_features if attack == 'vertex_deletion' and eligible_vertex_features else np.nan, 'nc': score_nc, 'bcr': bit_correct_rate(reference, candidate), 'passed': score_nc > args.threshold, 'seconds': elapsed})
                print(path.name, attack, ratio, flush=True)
    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError('No structural-loss rows were produced. Check --layers, shard settings, the 64-feature minimum and requested attack applicability.')
    protocol = {'seed': args.seed, 'threshold': args.threshold, 'ratios': RATIOS, 'attacks': args.attacks, 'shard_index': args.shard_index, 'shard_count': args.shard_count, 'feature_deletion': 'uniform random complete-feature removal; retained order preserved', 'vertex_deletion': 'partial-layer nested deletion: a fixed deterministic subset of eligible features is edited at every severity; within selected features, deleted interior vertices are nested across 10%-90%; endpoints, source order, ring closure and polygon holes are preserved; FPS is not evaluated', 'vertex_feature_fraction': args.vertex_feature_fraction, 'sampling': 'two-scale equidistant index selection only (numpy.linspace); no FPS', 'spatial_crop': 'central window retaining 1-ratio of the original bounding-box area', 'point_vertex_deletion': 'not applicable; deleting a point is feature deletion'}
    write_outputs(details, args.output, protocol)

def merge(args: argparse.Namespace) -> None:
    files = sorted(args.shard_root.glob('shard*/structural_loss_by_file.csv'))
    if not files:
        files = sorted(args.shard_root.glob('*/structural_loss_by_file.csv'))
    if not files:
        raise FileNotFoundError('No shard structural_loss_by_file.csv files were found')
    args.output.mkdir(parents=True, exist_ok=False)
    details = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    protocol_paths = [path.parent / 'protocol.json' for path in files]
    source_protocol = json.loads(protocol_paths[0].read_text(encoding='utf-8'))
    source_protocol.update({'merged_shards': len(files), 'merged_rows': len(details)})
    write_outputs(details, args.output, source_protocol)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    run_parser = subparsers.add_parser('run', help='Run one complete experiment or one shard')
    run_parser.add_argument('--layers', type=Path, required=True)
    run_parser.add_argument('--mamba-checkpoint', type=Path, required=True)
    run_parser.add_argument('--fusion-checkpoint', type=Path, required=True)
    run_parser.add_argument('--image-checkpoint', type=Path, required=True)
    run_parser.add_argument('--copyright-image', type=Path, required=True)
    run_parser.add_argument('--output', type=Path, required=True)
    run_parser.add_argument('--attacks', nargs='+', choices=ATTACKS, default=list(ATTACKS))
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--threshold', type=float, default=0.9)
    run_parser.add_argument('--vertex-feature-fraction', type=float, default=0.2)
    run_parser.add_argument('--shard-index', type=int, default=0)
    run_parser.add_argument('--shard-count', type=int, default=1)
    merge_parser = subparsers.add_parser('merge', help='Merge parallel shard outputs')
    merge_parser.add_argument('--shard-root', type=Path, required=True)
    merge_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        if not 0 <= args.shard_index < args.shard_count:
            parser.error('--shard-index must be in [0, --shard-count)')
        if not 0 < args.vertex_feature_fraction <= 1:
            parser.error('--vertex-feature-fraction must be in (0, 1]')
        run(args)
    else:
        merge(args)
if __name__ == '__main__':
    main()
