import shutil
from pathlib import Path
import geopandas as gpd
import numpy as np
import shapely
from shapely.affinity import rotate, scale, skew
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import transform

def geometry_family(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in ('Point', 'MultiPoint'):
        return 'Point'
    if geometry.geom_type in ('LineString', 'MultiLineString'):
        return 'LineString'
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return 'Polygon'
    return None

def vertex_count(layer):
    return int(sum((shapely.get_num_coordinates(geometry) for geometry in layer.geometry if geometry is not None and (not geometry.is_empty))))

def layer_statistics(layer):
    bounds = [float(value) for value in layer.total_bounds] if len(layer) else [None] * 4
    return {'features': int(len(layer)), 'vertices': vertex_count(layer), 'bounds': bounds}

def _insert_in_sequence(coords, ratio, rng, closed=False):
    array = np.asarray(coords, dtype=np.float64)
    if len(array) < 2 or ratio <= 0:
        return array
    segment_count = len(array) - 1
    additions = min(segment_count, max(1, int(round(segment_count * ratio))))
    selected = set(rng.choice(segment_count, size=additions, replace=False).tolist())
    output = []
    for index in range(segment_count):
        output.append(array[index])
        if index in selected:
            output.append((array[index] + array[index + 1]) / 2.0)
    output.append(array[-1])
    result = np.asarray(output, dtype=np.float64)
    if closed:
        result[-1] = result[0]
    return result

def _insert_vertices(geometry, ratio, rng):
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type == 'LineString':
        return LineString(_insert_in_sequence(geometry.coords, ratio, rng))
    if geometry.geom_type == 'MultiLineString':
        return MultiLineString([_insert_vertices(part, ratio, rng) for part in geometry.geoms])
    if geometry.geom_type == 'Polygon':
        exterior = _insert_in_sequence(geometry.exterior.coords, ratio, rng, closed=True)
        holes = [_insert_in_sequence(ring.coords, ratio, rng, closed=True) for ring in geometry.interiors]
        return Polygon(exterior, holes)
    if geometry.geom_type == 'MultiPolygon':
        return MultiPolygon([_insert_vertices(part, ratio, rng) for part in geometry.geoms])
    return geometry

class ShapefileAttacker:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(f'Attack output already exists: {self.output_dir}')
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def save(self, layer, name):
        path = self.output_dir / f'{name}.shp'
        layer.to_file(path, encoding='utf-8')
        return str(path)

    def attack_noise(self, layer, intensity=0.001, seed=42):
        if intensity == 0:
            return self.save(layer, 'no_attack')
        rng = np.random.default_rng(seed)

        def perturb(geometry):
            if geometry is None or geometry.is_empty:
                return geometry

            def noisy(x, y, z=None):
                x_array = np.asarray(x, dtype=np.float64)
                y_array = np.asarray(y, dtype=np.float64)
                changed = (x_array + rng.normal(0, intensity, x_array.shape), y_array + rng.normal(0, intensity, y_array.shape))
                return changed if z is None else (*changed, z)
            return transform(noisy, geometry)
        attacked = layer.copy()
        attacked['geometry'] = attacked.geometry.apply(perturb)
        return self.save(attacked, f'noise_{intensity}')

    def attack_vertex_add(self, layer, ratio=0.3, seed=42):
        rng = np.random.default_rng(seed)
        attacked = layer.copy()
        attacked['geometry'] = attacked.geometry.apply(lambda geometry: _insert_vertices(geometry, ratio, rng))
        return self.save(attacked, f'vertex_add_{ratio}')

    def attack_feature_add(self, layer, ratio=0.3, donor=None, seed=42):
        if donor is None:
            raise ValueError('feature addition requires a separate donor layer')
        target_families = [geometry_family(geometry) for geometry in layer.geometry]
        target_families = [kind for kind in target_families if kind]
        if not target_families:
            raise ValueError('target layer has no supported geometry')
        target_family = max(set(target_families), key=target_families.count)
        if layer.crs is not None and donor.crs is not None and (layer.crs != donor.crs):
            donor = donor.to_crs(layer.crs)
        donor_geometries = [geometry for geometry in donor.geometry if geometry_family(geometry) == target_family]
        if not donor_geometries:
            raise ValueError(f'donor has no {target_family} geometry')
        count = max(1, int(round(len(layer) * ratio)))
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(donor_geometries), size=count, replace=count > len(donor_geometries))
        geometries = list(layer.geometry) + [donor_geometries[index] for index in ids]
        attacked = gpd.GeoDataFrame({'geometry': geometries}, crs=layer.crs)
        return self.save(attacked, f'feature_add_{ratio}')

    def attack_clip(self, layer, ratio=0.3):
        if not 0 <= ratio < 1:
            raise ValueError('clip ratio must be in [0, 1)')
        min_x, min_y, max_x, max_y = layer.total_bounds
        scale_factor = np.sqrt(1.0 - ratio)
        half_width = (max_x - min_x) * scale_factor / 2.0
        half_height = (max_y - min_y) * scale_factor / 2.0
        centre_x, centre_y = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        window = box(centre_x - half_width, centre_y - half_height, centre_x + half_width, centre_y + half_height)
        attacked = layer.copy()
        attacked['geometry'] = attacked.geometry.intersection(window)
        target_families = [geometry_family(geometry) for geometry in layer.geometry]
        target_families = [kind for kind in target_families if kind]
        target_family = max(set(target_families), key=target_families.count)
        attacked = attacked[~attacked.geometry.is_empty & attacked.geometry.notna() & attacked.geometry.apply(lambda geometry: geometry_family(geometry) == target_family)].copy()
        if attacked.empty:
            raise ValueError('spatial clipping removed every supported feature')
        return self.save(attacked, f'clip_{ratio}')

    def attack_delete(self, layer, ratio=0.3):
        retained = max(1, int(len(layer) * (1 - ratio)))
        attacked = layer.sample(n=retained) if retained < len(layer) else layer.copy()
        return self.save(attacked, f'delete_{ratio}')

    def attack_rotate(self, layer, angle=45):
        attacked = layer.copy()
        try:
            origin = attacked.dissolve().centroid.iloc[0]
        except Exception:
            origin = 'centroid'
        attacked['geometry'] = attacked.geometry.apply(lambda geometry: rotate(geometry, angle, origin=origin))
        return self.save(attacked, f'rotate_{angle}')

    def attack_scale(self, layer, xfact=1.0, yfact=1.0):
        attacked = layer.copy()
        try:
            origin = attacked.dissolve().centroid.iloc[0]
        except Exception:
            origin = 'centroid'
        attacked['geometry'] = attacked.geometry.apply(lambda geometry: scale(geometry, xfact=xfact, yfact=yfact, origin=origin))
        return self.save(attacked, f'scale_{xfact}_{yfact}')

    def attack_shear(self, layer, xs=0.0, ys=0.0):
        attacked = layer.copy()
        attacked['geometry'] = attacked.geometry.apply(lambda geometry: skew(geometry, xs=xs, ys=ys, origin='centroid'))
        return self.save(attacked, f'shear_{xs}_{ys}')

    def attack_simplify(self, layer, tolerance=0.001):
        attacked = layer.copy()
        attacked['geometry'] = attacked.geometry.simplify(tolerance)
        return self.save(attacked, f'simplify_{tolerance}')

def bit_correct_rate(reference, candidate):
    if reference is None or candidate is None:
        return 0.0
    reference = np.asarray(reference).ravel()
    candidate = np.asarray(candidate).ravel()
    length = min(len(reference), len(candidate))
    if length == 0:
        return 0.0
    return float(((reference[:length] > 0) == (candidate[:length] > 0)).mean())
