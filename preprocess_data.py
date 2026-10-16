import argparse
import hashlib
import json
import time
from pathlib import Path
import geopandas as gpd
import numpy as np

def geometry_kind(geometry):
    if geometry.geom_type in ('Point', 'MultiPoint'):
        return 'Point'
    if geometry.geom_type in ('LineString', 'MultiLineString'):
        return 'LineString'
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return 'Polygon'
    return None

def coordinates(geometry, kind):
    if kind == 'Point':
        return np.asarray([geometry.coords[0]] if geometry.geom_type == 'Point' else [point.coords[0] for point in geometry.geoms], dtype=np.float64)
    if kind == 'LineString':
        line = geometry if geometry.geom_type == 'LineString' else max(geometry.geoms, key=lambda x: x.length)
        return np.asarray(line.coords, dtype=np.float64)
    polygon = geometry if geometry.geom_type == 'Polygon' else max(geometry.geoms, key=lambda x: x.area)
    return np.asarray(polygon.exterior.coords, dtype=np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, help='Folder containing *_free_2.shp and *_free_3.shp')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    array_dir = args.output / 'arrays'
    array_dir.mkdir()
    started, rows = (time.time(), [])
    paths = sorted(list(args.input.glob('*_free_2.shp')) + list(args.input.glob('*_free_3.shp')))
    for path in paths:
        layer = gpd.read_file(path)
        valid = [g for g in layer.geometry if g is not None and (not g.is_empty) and geometry_kind(g)]
        if not valid:
            continue
        kinds = [geometry_kind(g) for g in valid]
        kind = max(set(kinds), key=kinds.count)
        sequences = [coordinates(g, kind) for g in valid if geometry_kind(g) == kind]
        sequences = [x for x in sequences if len(x) and np.isfinite(x).all()]
        if not sequences:
            continue
        merged = np.concatenate(sequences).astype(np.float64)
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(x) for x in sequences])
        target = array_dir / f'{path.stem}.npz'
        np.savez_compressed(target, coords=merged, offsets=offsets)
        rows.append({'source': str(path), 'output': str(Path('arrays') / target.name), 'geometry_type': kind, 'features': len(sequences), 'vertices': len(merged), 'crs': str(layer.crs), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps({'layers': len(rows), 'seconds': time.time() - started}))
if __name__ == '__main__':
    main()
