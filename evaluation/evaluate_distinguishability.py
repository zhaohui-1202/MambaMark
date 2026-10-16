from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOSTS = ['gis_osm_buildings_a_free_1.shp', 'gis_osm_water_a_free_1.shp', 'gis_osm_roads_free_1.shp', 'gis_osm_pois_free_1.shp']
LABELS = ['buildings', 'water', 'roads', 'pois']
IMAGES = ['1.jpg', '10.jpg', '100.jpg', '101.jpg', '102.jpg']

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def nc_matrix(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError('Fingerprints must be finite and nonzero')
    unit = vectors / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)

def summarise(matrix, threshold):
    values = matrix[np.triu_indices(len(matrix), 1)]
    return {'pairs': len(values), 'mean_nonmatch_nc': float(values.mean()), 'max_nonmatch_nc': float(values.max()), 'false_accepts': int(np.sum(values >= threshold)), 'nc_far_percent': float(100 * np.mean(values >= threshold)), 'threshold': threshold}

def export(matrix, labels, output, stem, image_gallery):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    with (output / (stem + '.csv')).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['registered / candidate'] + labels)
        writer.writerows(([label] + row.tolist() for label, row in zip(labels, matrix)))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(8, 6) if image_gallery else (7.2, 5.8))
    kwargs = {'annot_kws': {'fontsize': 11, 'weight': 'bold'}, 'cbar_kws': {'label': 'Similarity (NC)'}} if image_gallery else {'square': True, 'linewidths': 0.8, 'linecolor': 'white', 'annot_kws': {'fontsize': 11}, 'cbar_kws': {'label': 'Normalized Correlation (NC)', 'shrink': 0.88}}
    sns.heatmap(matrix, annot=True, fmt='.4f', vmin=0, vmax=1, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, **kwargs)
    ax.set_ylim(len(labels), 0)
    if image_gallery:
        ax.set_title('Confusion Matrix: Different Copyright Images', pad=20, fontsize=15, weight='bold')
        ax.set_xlabel('Tested Data (Verification)', fontsize=13)
        ax.set_ylabel('Registered Data (Original)', fontsize=13)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    else:
        ax.set_title('Layer-level Distinguishability', pad=12, weight='bold')
        ax.set_xlabel('Candidate layer')
        ax.set_ylabel('Registered layer')
        ax.tick_params(axis='both', rotation=0)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / (stem + '.' + ext), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=Path, default=ROOT / 'data/geographic_layers')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/copyright')
    parser.add_argument('--mamba-checkpoint', type=Path, default=ROOT / 'checkpoints/mamba_encoder_final.pth')
    parser.add_argument('--fusion-checkpoint', type=Path, default=ROOT / 'checkpoints/fusion_final.pth')
    parser.add_argument('--image-checkpoint', type=Path, default=ROOT / 'checkpoints/image_encoder_final.pth')
    parser.add_argument('--image-names', nargs='+', default=IMAGES)
    parser.add_argument('--fixed-host', default=HOSTS[0])
    parser.add_argument('--fixed-image', default=IMAGES[0])
    parser.add_argument('--gallery', choices=['both', 'host', 'copyright'], default='both')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/distinguishability_structural_aug')
    parser.add_argument('--reference-summary', type=Path, help='Optional matrix_preview/summary.json for numerical verification')
    args = parser.parse_args(argv)
    if len(args.image_names) < 2 or len(set(args.image_names)) != len(args.image_names):
        parser.error('Provide at least two distinct image names')
    if not -1 <= args.threshold <= 1:
        parser.error('NC threshold must lie in [-1, 1]')
    from system import MambaMarkSystem
    import torch
    torch.set_num_threads(4)
    manager = MambaMarkSystem(args.mamba_checkpoint, args.image_checkpoint, args.fusion_checkpoint, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'metric': 'Cosine NC of final fused fingerprints', 'feature_count': 64, 'vertex_count': 64, 'weights': {name: {'path': str(path), 'sha256': digest(path)} for name, path in [('encoder', args.mamba_checkpoint), ('fusion', args.fusion_checkpoint), ('image', args.image_checkpoint)]}, 'note': 'Small galleries, not the extended 17-host FAR. Default five images were seen in training.'}
    host_cache, image_cache = ({}, {})
    from PIL import Image
    image_names = list(dict.fromkeys([args.fixed_image, *args.image_names]))
    pixels = np.stack([np.asarray(Image.open(args.images / name).convert('RGB').resize((224, 224)), dtype=np.float32) / 255 for name in image_names])
    bank = manager.encode_image(torch.from_numpy(pixels).permute(0, 3, 1, 2))
    image_cache.update({name: bank[i:i + 1] for i, name in enumerate(image_names)})
    summary['image_batch'] = image_names
    summary['image_sha256'] = {name: digest(args.images / name) for name in image_names}

    def fingerprint(host_name, image_name):
        if host_name not in host_cache:
            print('Encoding host:', host_name, flush=True)
            host_cache[host_name] = manager.encode_host(args.layers / host_name)
        if image_name not in image_cache:
            image_cache[image_name] = manager.encode_image(args.images / image_name)
        with torch.inference_mode():
            return manager.fusion(host_cache[host_name], image_cache[image_name]).cpu().numpy()[0]
    if args.gallery in ('host', 'both'):
        matrix = nc_matrix([fingerprint(n, args.fixed_image) for n in HOSTS])
        summary['host'] = dict(summarise(matrix, args.threshold), matrix=matrix.tolist(), labels=LABELS, files=HOSTS, fixed_image=args.fixed_image)
        export(matrix, LABELS, args.output, 'four_layer_nc', False)
    if args.gallery in ('both', 'copyright'):
        labels = [Path(n).stem for n in args.image_names]
        matrix = nc_matrix([fingerprint(args.fixed_host, n) for n in args.image_names])
        summary['image'] = dict(summarise(matrix, args.threshold), matrix=matrix.tolist(), labels=labels, files=args.image_names, fixed_host=args.fixed_host)
        export(matrix, labels, args.output, 'five_image_nc', True)
    if args.reference_summary:
        reference = json.loads(args.reference_summary.read_text(encoding='utf-8'))
        for key in ('host', 'image'):
            if key in summary:
                if key == 'host':
                    assert HOSTS == reference['host_names'] and args.fixed_image == reference['host_fixed_image']
                else:
                    assert args.image_names == reference['image_names'] and args.fixed_host == reference['image_fixed_host']
                np.testing.assert_allclose(summary[key]['matrix'], reference[key + '_nc'], atol=2e-05, rtol=0)
        summary['reference_matrix_check'] = 'passed, absolute tolerance 2e-5'
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'matrix'} for k, v in summary.items() if k in ('host', 'image')}, indent=2))
if __name__ == '__main__':
    main()
