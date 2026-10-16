from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Union
import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from torch import nn
from models.fusion import HostDominantGatedFusion, load_state
from models.image_encoder import ImageEncoder
from models.mamba_layer_encoder import NativeMultiGeometryEncoder
from preprocess_data import coordinates, geometry_kind
from training.data import GEOMETRY_TYPES
LayerInput = Union[str, Path, gpd.GeoDataFrame]
ImageInput = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]

def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else 0.0

def bit_correct_rate(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.uint8).reshape(-1)
    second = np.asarray(second, dtype=np.uint8).reshape(-1)
    if first.shape != second.shape:
        raise ValueError(f'Bit arrays have different shapes: {first.shape} and {second.shape}')
    return float(np.mean(first == second))

class MambaMarkSystem(nn.Module):
    fingerprint_length = 256

    def __init__(self, mamba_checkpoint: Union[str, Path], image_checkpoint: Union[str, Path], fusion_checkpoint: Union[str, Path], device: str | None=None, feature_count: int=64, vertex_count: int=64) -> None:
        super().__init__()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.feature_count = int(feature_count)
        self.vertex_count = int(vertex_count)
        if self.feature_count <= 0 or self.vertex_count <= 0:
            raise ValueError('feature_count and vertex_count must be positive')
        checkpoints = [Path(mamba_checkpoint), Path(image_checkpoint), Path(fusion_checkpoint)]
        missing = [str(path) for path in checkpoints if not path.is_file()]
        if missing:
            raise FileNotFoundError('Missing checkpoint(s): ' + ', '.join(missing))
        self.mamba = NativeMultiGeometryEncoder().to(self.device)
        self.mamba.load_state_dict(load_state(checkpoints[0], 'encoder'), strict=True)
        self.image_encoder = ImageEncoder(pretrained=False).to(self.device)
        self.image_encoder.load_state_dict(load_state(checkpoints[1]), strict=True)
        self.fusion = HostDominantGatedFusion().to(self.device)
        self.fusion.load_state_dict(load_state(checkpoints[2], 'fusion'), strict=True)
        self.eval()

    @staticmethod
    def _read_layer(layer: LayerInput) -> gpd.GeoDataFrame:
        frame = gpd.read_file(layer) if isinstance(layer, (str, Path)) else layer.copy()
        if 'geometry' not in frame or frame.empty:
            raise ValueError('The layer is empty or has no geometry column')
        return frame

    @staticmethod
    def _image_tensor(image: ImageInput, device: torch.device) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().to(device=device, dtype=torch.float32)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise ValueError('Image tensor must have shape [C,H,W] or [B,C,H,W]')
            return tensor
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            array = np.asarray(image)
            if array.dtype != np.uint8:
                array = np.clip(array * 255.0 if array.max(initial=0) <= 1 else array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array).convert('RGB')
        array = np.asarray(pil_image.resize((224, 224)), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    def encode_image(self, image: ImageInput) -> torch.Tensor:
        tensor = self._image_tensor(image, self.device)
        with torch.inference_mode():
            encoded = self.image_encoder(tensor)
            return encoded.mean(1) if encoded.dim() == 3 else encoded

    def encode_host(self, layer: LayerInput) -> torch.Tensor:
        frame = self._read_layer(layer)
        valid = [geometry for geometry in frame.geometry if geometry is not None and (not geometry.is_empty) and geometry_kind(geometry)]
        if not valid:
            raise ValueError('The layer contains no supported point, line, or polygon geometry')
        kinds = [geometry_kind(geometry) for geometry in valid]
        kind = max(set(kinds), key=kinds.count)
        valid = [geometry for geometry in valid if geometry_kind(geometry) == kind]
        if len(valid) < self.feature_count:
            feature_ids = np.arange(self.feature_count) % len(valid)
        else:
            feature_ids = np.linspace(0, len(valid) - 1, self.feature_count).round().astype(int)
        sequences, centroids = ([], [])
        for index in feature_ids:
            sequence = coordinates(valid[index], kind).astype(np.float32)
            if not len(sequence) or not np.isfinite(sequence).all():
                raise ValueError('A selected geometry has invalid coordinates')
            centroids.append(sequence.mean(0))
            sequence = sequence - sequence.mean(0)
            radius = np.linalg.norm(sequence, axis=1).max() if len(sequence) > 1 else 1.0
            sequence = sequence / max(float(radius), 1e-08)
            if kind == 'Point':
                sequence = np.zeros((1, 2), dtype=np.float32)
            positions = np.linspace(0, len(sequence) - 1, self.vertex_count).round().astype(int)
            sequences.append(sequence[positions])
        centroid_array = np.asarray(centroids, dtype=np.float32)
        centroid_array -= centroid_array.mean(0)
        centroid_radius = np.linalg.norm(centroid_array, axis=1).max()
        centroid_array /= max(float(centroid_radius), 1e-08)
        vertices = torch.tensor(np.stack(sequences), device=self.device)[None]
        vertex_mask = torch.ones(1, self.feature_count, self.vertex_count, dtype=torch.bool, device=self.device)
        centroid_tensor = torch.tensor(centroid_array, device=self.device)[None]
        geometry_types = torch.full((1, self.feature_count), GEOMETRY_TYPES[kind], dtype=torch.long, device=self.device)
        feature_mask = torch.ones(1, self.feature_count, dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            return self.mamba(vertices, vertex_mask, centroid_tensor, geometry_types, feature_mask)

    def get_feature_hash(self, layer: LayerInput, copyright_image: ImageInput) -> np.ndarray:
        host = self.encode_host(layer)
        image = self.encode_image(copyright_image)
        with torch.inference_mode():
            fingerprint = self.fusion(host, image)
        return fingerprint.detach().cpu().numpy()[0].astype(np.float32)

    @staticmethod
    def binarize(fingerprint: np.ndarray) -> np.ndarray:
        return (np.asarray(fingerprint).reshape(-1) >= 0).astype(np.uint8)

    def register(self, layer: LayerInput, copyright_image: ImageInput, payload_bits: np.ndarray) -> Dict[str, Any]:
        started = time.perf_counter()
        fingerprint = self.get_feature_hash(layer, copyright_image)
        feature_bits = self.binarize(fingerprint)
        payload = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
        if payload.shape != feature_bits.shape:
            raise ValueError(f'Payload must contain {len(feature_bits)} bits, received {len(payload)}')
        if not np.isin(payload, [0, 1]).all():
            raise ValueError('Payload must be binary')
        return {'zero_watermark': np.bitwise_xor(feature_bits, payload), 'reference_fingerprint': fingerprint, 'feature_bits': feature_bits, 'seconds': time.perf_counter() - started}

    def detect(self, layer: LayerInput, copyright_image: ImageInput, zero_watermark: np.ndarray) -> Dict[str, Any]:
        started = time.perf_counter()
        fingerprint = self.get_feature_hash(layer, copyright_image)
        feature_bits = self.binarize(fingerprint)
        watermark = np.asarray(zero_watermark, dtype=np.uint8).reshape(-1)
        if watermark.shape != feature_bits.shape:
            raise ValueError('The stored zero watermark has an incompatible length')
        return {'recovered_payload': np.bitwise_xor(feature_bits, watermark), 'candidate_fingerprint': fingerprint, 'feature_bits': feature_bits, 'seconds': time.perf_counter() - started}
ZeroWatermarkSystem = MambaMarkSystem
