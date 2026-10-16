from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union
import geopandas as gpd
import numpy as np
from PIL import Image
from preprocess_data import coordinates, geometry_kind
from system import MambaMarkSystem, bit_correct_rate, normalized_correlation
PathLike = Union[str, Path]

class WatermarkManager:

    def __init__(self, mamba_checkpoint: PathLike | None=None, image_checkpoint: PathLike | None=None, fusion_checkpoint: PathLike | None=None, device: str | None=None, feature_count: int=64, vertex_count: int=64, nc_threshold: float=0.9) -> None:
        root = Path(__file__).resolve().parent
        checkpoint_dir = root / 'checkpoints'
        self.paths = {'mamba': Path(mamba_checkpoint or checkpoint_dir / 'mamba_encoder_final.pth'), 'image': Path(image_checkpoint or checkpoint_dir / 'image_encoder_final.pth'), 'fusion': Path(fusion_checkpoint or checkpoint_dir / 'fusion_final.pth')}
        self.feature_count = int(feature_count)
        self.vertex_count = int(vertex_count)
        self.nc_threshold = float(nc_threshold)
        self.system = MambaMarkSystem(self.paths['mamba'], self.paths['image'], self.paths['fusion'], device=device, feature_count=feature_count, vertex_count=vertex_count)

    @staticmethod
    def image_to_payload(image_path: PathLike, length: int=256) -> np.ndarray:
        side = int(np.ceil(np.sqrt(length)))
        image = Image.open(image_path).convert('L').resize((side, side), Image.Resampling.LANCZOS)
        values = np.asarray(image, dtype=np.float32).reshape(-1)[:length]
        return (values >= 127.5).astype(np.uint8)

    @staticmethod
    def text_to_payload(text: str, length: int=256) -> np.ndarray:
        if not text:
            raise ValueError('Payload text must not be empty')
        source = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))
        return np.resize(source, length).astype(np.uint8)

    @staticmethod
    def _payload_hash(payload: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(payload, dtype=np.uint8).tobytes()).hexdigest()

    @staticmethod
    def _registration_paths(path: PathLike) -> tuple[Path, Path]:
        array_path = Path(path)
        if array_path.suffix.lower() != '.npz':
            array_path = array_path.with_suffix('.npz')
        return (array_path, array_path.with_suffix('.json'))

    @staticmethod
    def layer_statistics(layer: PathLike) -> Dict[str, Any]:
        frame = gpd.read_file(layer)
        kinds, vertices = ([], 0)
        for geometry in frame.geometry:
            if geometry is None or geometry.is_empty:
                continue
            kind = geometry_kind(geometry)
            if kind:
                kinds.append(kind)
                vertices += len(coordinates(geometry, kind))
        dominant = max(set(kinds), key=kinds.count) if kinds else None
        return {'features': int(len(frame)), 'valid_features': int(len(kinds)), 'vertices': int(vertices), 'dominant_geometry': dominant, 'crs': str(frame.crs)}

    def _resolve_payload(self, payload_bits: np.ndarray | None, payload_image: PathLike | None, payload_text: str | None, default_image: PathLike) -> np.ndarray:
        supplied = sum((value is not None for value in (payload_bits, payload_image, payload_text)))
        if supplied > 1:
            raise ValueError('Provide only one of payload_bits, payload_image, or payload_text')
        if payload_bits is not None:
            payload = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
        elif payload_text is not None:
            payload = self.text_to_payload(payload_text, self.system.fingerprint_length)
        else:
            payload = self.image_to_payload(payload_image or default_image, self.system.fingerprint_length)
        if len(payload) != self.system.fingerprint_length or not np.isin(payload, [0, 1]).all():
            raise ValueError(f'Payload must contain {self.system.fingerprint_length} binary bits')
        return payload

    def register_layer(self, layer_path: PathLike, copyright_image: PathLike, registration_path: PathLike, *, payload_bits: np.ndarray | None=None, payload_image: PathLike | None=None, payload_text: str | None=None) -> Dict[str, Any]:
        payload = self._resolve_payload(payload_bits, payload_image, payload_text, copyright_image)
        record = self.system.register(layer_path, copyright_image, payload)
        array_path, metadata_path = self._registration_paths(registration_path)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        if array_path.exists() or metadata_path.exists():
            raise FileExistsError(f'Registration output already exists: {array_path}')
        np.savez_compressed(array_path, zero_watermark=record['zero_watermark'].astype(np.uint8), reference_fingerprint=record['reference_fingerprint'].astype(np.float32))
        metadata = {'method': 'MambaMark', 'source_layer': str(Path(layer_path).resolve()), 'copyright_image': str(Path(copyright_image).resolve()), 'payload_sha256': self._payload_hash(payload), 'fingerprint_length': self.system.fingerprint_length, 'feature_count': self.feature_count, 'vertex_count': self.vertex_count, 'nc_threshold': self.nc_threshold, 'registration_seconds': record['seconds'], 'layer': self.layer_statistics(layer_path), 'checkpoints': {name: path.name for name, path in self.paths.items()}}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'registration': str(array_path), 'metadata': str(metadata_path), **metadata}

    def verify_layer(self, candidate_layer: PathLike, copyright_image: PathLike, registration_path: PathLike, *, payload_bits: np.ndarray | None=None, payload_image: PathLike | None=None, payload_text: str | None=None) -> Dict[str, Any]:
        array_path, metadata_path = self._registration_paths(registration_path)
        if not array_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f'Incomplete registration: {array_path} / {metadata_path}')
        stored = np.load(array_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        payload = self._resolve_payload(payload_bits, payload_image, payload_text, copyright_image)
        detected = self.system.detect(candidate_layer, copyright_image, stored['zero_watermark'])
        nc = normalized_correlation(stored['reference_fingerprint'], detected['candidate_fingerprint'])
        bcr = bit_correct_rate(payload, detected['recovered_payload'])
        return {'method': 'MambaMark', 'candidate_layer': str(Path(candidate_layer).resolve()), 'continuous_nc': nc, 'payload_bcr': bcr, 'accepted': bool(nc >= self.nc_threshold), 'nc_threshold': self.nc_threshold, 'payload_hash_matches_registration': self._payload_hash(payload) == metadata['payload_sha256'], 'verification_seconds': detected['seconds'], 'candidate': self.layer_statistics(candidate_layer), 'recovered_payload': detected['recovered_payload']}

def _manager_from_args(args: argparse.Namespace) -> WatermarkManager:
    return WatermarkManager(mamba_checkpoint=args.mamba_checkpoint, image_checkpoint=args.image_checkpoint, fusion_checkpoint=args.fusion_checkpoint, device=args.device)

def main() -> None:
    parser = argparse.ArgumentParser(description='MambaMark zero-watermark registration and verification')
    parser.add_argument('--mamba-checkpoint', type=Path)
    parser.add_argument('--image-checkpoint', type=Path)
    parser.add_argument('--fusion-checkpoint', type=Path)
    parser.add_argument('--device')
    subparsers = parser.add_subparsers(dest='command', required=True)
    register = subparsers.add_parser('register')
    register.add_argument('--layer', type=Path, required=True)
    register.add_argument('--copyright-image', type=Path, required=True)
    register.add_argument('--registration', type=Path, required=True)
    register.add_argument('--payload-image', type=Path)
    register.add_argument('--payload-text')
    verify = subparsers.add_parser('verify')
    verify.add_argument('--layer', type=Path, required=True)
    verify.add_argument('--copyright-image', type=Path, required=True)
    verify.add_argument('--registration', type=Path, required=True)
    verify.add_argument('--payload-image', type=Path)
    verify.add_argument('--payload-text')
    args = parser.parse_args()
    manager = _manager_from_args(args)
    if args.command == 'register':
        result = manager.register_layer(args.layer, args.copyright_image, args.registration, payload_image=args.payload_image, payload_text=args.payload_text)
    else:
        result = manager.verify_layer(args.layer, args.copyright_image, args.registration, payload_image=args.payload_image, payload_text=args.payload_text)
        result.pop('recovered_payload')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
