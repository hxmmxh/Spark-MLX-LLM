"""Still-image preprocessing and placeholder expansion, without PyTorch."""

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Qwen2-VL's aspect-preserving patch-aligned resize (Apache-2.0)."""
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must be at most 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


class Spark3Processor:
    """One prompt, one or more still images, native MLX tensors.

    Accepts one placeholder per image or already expanded contiguous image
    placeholders. Image offsets are inclusive and preserve adjacent boundaries.
    """

    def __init__(self, tokenizer, args, image_config=None, root_config=None):
        if args.vision_config is None:
            raise ValueError("Spark3Processor requires vision_config")
        self.tokenizer = tokenizer
        self._tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        self.args = args
        self.image_config = dict(image_config or {})
        self.root_config = dict(root_config or {})
        self.image_token = args.image_token
        if self.image_token is None:
            if args.image_token_id < 0:
                raise ValueError("negative image_token_id requires image_token")
            self.image_token = tokenizer.convert_ids_to_tokens(args.image_token_id)
            if (
                not isinstance(self.image_token, str)
                or not self.image_token
                or tokenizer.convert_tokens_to_ids(self.image_token)
                != args.image_token_id
            ):
                raise ValueError("image_token_id must round-trip through the tokenizer")
        if not isinstance(self.image_token, str) or not self.image_token:
            raise ValueError("image_token must be a nonempty string")
        vision = args.vision_config
        for name, value in (
            ("patch_size", vision.patch_size),
            ("temporal_patch_size", vision.temporal_patch_size),
            ("merge_size", vision.spatial_merge_size),
        ):
            if self.image_config.get(name, value) != value:
                raise ValueError(f"image processor {name} must match vision_config")

    @classmethod
    def from_pretrained(cls, model_path, tokenizer, args=None):
        from .spark3 import ModelArgs

        path = Path(model_path)
        config = json.loads((path / "config.json").read_text())
        args = args or ModelArgs.from_dict(config)
        preprocessor = path / "preprocessor_config.json"
        image_config = (
            json.loads(preprocessor.read_text()) if preprocessor.exists() else {}
        )
        return cls(tokenizer, args, image_config, config)

    @property
    def image_placeholder(self):
        prefix, suffix = "", ""
        for key in ("vision_start_token_id", "vision_end_token_id"):
            token_id = self.root_config.get(key)
            token = (
                self.tokenizer.convert_ids_to_tokens(token_id)
                if token_id is not None
                else ""
            )
            if key == "vision_start_token_id":
                prefix = token
            else:
                suffix = token
        return prefix + self.image_token + suffix

    def process_images(self, images, **overrides):
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise ImportError(
                "Install spark-mlx-llm[vision] to process images"
            ) from exc

        config = self.image_config | overrides
        vision = self.args.vision_config
        patch, merge = vision.patch_size, vision.spatial_merge_size
        temporal = vision.temporal_patch_size
        size = config.get("size", {})
        min_pixels = config.get("min_pixels", size.get("shortest_edge", 56 * 56))
        max_pixels = config.get("max_pixels", size.get("longest_edge", 28 * 28 * 1280))
        if not 0 < min_pixels <= max_pixels:
            raise ValueError(
                "image pixel bounds must satisfy 0 < min_pixels <= max_pixels"
            )
        mean = np.asarray(config.get("image_mean", [0.5] * 3), dtype=np.float32)
        std = np.asarray(config.get("image_std", [0.5] * 3), dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
            raise ValueError(
                "image normalization requires three RGB means and positive stds"
            )
        patches, grids = [], []
        for source in images:
            if isinstance(source, (str, Path)):
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            elif isinstance(source, Image.Image):
                image = ImageOps.exif_transpose(source).convert("RGB")
            else:
                image = Image.fromarray(np.asarray(source)).convert("RGB")
            if config.get("do_resize", True):
                h, w = smart_resize(
                    image.height, image.width, patch * merge, min_pixels, max_pixels
                )
                image = image.resize(
                    (w, h), resample=config.get("resample", Image.Resampling.BICUBIC)
                )
            values = np.asarray(image, dtype=np.float32)
            h, w = values.shape[:2]
            if h % (patch * merge) or w % (patch * merge):
                raise ValueError(
                    "image dimensions must be divisible by patch_size * merge_size"
                )
            if config.get("do_rescale", True):
                values = values * np.float32(config.get("rescale_factor", 1 / 255))
            if config.get("do_normalize", True):
                values = (values - mean) / std
            gh, gw = h // patch, w // patch
            # HWC -> CTHW：在 temporal 个时间位置广播同一张静态图。
            values = np.broadcast_to(
                values.transpose(2, 0, 1)[:, None], (3, temporal, h, w)
            )
            # 将 H/W 分解为「分组位置、组内 patch 位置、patch 内像素位置」。
            # reshape 后各轴编号：
            #   0:C, 1:T, 2:group_h, 3:inner_h, 4:pixel_h,
            #             5:group_w, 6:inner_w, 7:pixel_w。
            # group_h/w 的大小为 gh/gw // merge，inner_h/w 的大小为 merge，
            # pixel_h/w 的大小为 patch。
            values = values.reshape(
                3, temporal, gh // merge, merge, patch, gw // merge, merge, patch
            )
            # (2,5,3,6,0,1,4,7) ->
            # [group_h, group_w, inner_h, inner_w, C, T, pixel_h, pixel_w]。
            # 前四轴展平成 patch 序列，后四轴展平成每个 patch 的像素向量。
            # 因此同一 merge x merge 区域内的 patch 会连续存放。例如 merge=2：
            #   A B | C D
            #   E F | G H
            #   I J | K L
            #   M N | O P
            # 排列为 [A B E F] [C D G H] [I J M N] [K L O P]。
            # 这里仅重排；ViT 编码后的 SpatialMerger 才将每组特征拼接并投影。
            values = values.transpose(2, 5, 3, 6, 0, 1, 4, 7).reshape(gh * gw, -1)
            patches.append(values)
            grids.append([1, gh, gw])
        return mx.array(np.concatenate(patches)), grids

    def __call__(
        self,
        text,
        images=None,
        *,
        add_special_tokens=True,
        images_kwargs=None,
        videos=None,
        audios=None,
    ):
        if videos is not None or audios is not None:
            raise ValueError("Spark3 currently supports still images only")
        if not isinstance(text, str):
            raise TypeError("Spark3Processor expects one string prompt")
        if images is not None and not isinstance(images, (list, tuple)):
            images = [images]
        pieces = text.split(self.image_token)
        if not images:
            if len(pieces) != 1:
                raise ValueError("image placeholders require supplied images")
            return {
                "input_ids": mx.array(
                    [self.tokenizer.encode(text, add_special_tokens=add_special_tokens)]
                )
            }
        pixels, grids = self.process_images(images, **(images_kwargs or {}))
        counts = [h * w // 4 for _, h, w in grids]
        if len(pieces) - 1 == len(images):
            widths = [1] * len(images)
        elif len(pieces) - 1 == sum(counts):
            widths = counts
        else:
            raise ValueError(
                "image placeholder count must match images or expanded visual tokens"
            )
        ids, offsets, index = [], [], 0
        for count, width in zip(counts, widths):
            if any(piece.strip() for piece in pieces[index + 1 : index + width]):
                raise ValueError(
                    "expanded placeholders must be contiguous within each image"
                )
            ids.extend(self.tokenizer.encode(pieces[index], add_special_tokens=False))
            start = len(ids)
            ids.extend([self.args.image_token_id] * count)
            offsets.append((start, len(ids) - 1))
            index += width
        ids.extend(self.tokenizer.encode(pieces[-1], add_special_tokens=False))
        if add_special_tokens:
            builder = getattr(self.tokenizer, "build_inputs_with_special_tokens", None)
            if builder is not None:
                prefix = builder([-1]).index(-1)
                ids = builder(ids)
            else:
                probe = self._tokenizer(
                    "x", add_special_tokens=True, return_special_tokens_mask=True
                )
                body = [
                    i
                    for i, special in enumerate(probe["special_tokens_mask"])
                    if not special
                ]
                if not body or body != list(range(body[0], body[-1] + 1)):
                    raise ValueError(
                        "tokenizer must use prefix/suffix single-sequence special tokens"
                    )
                prefix = body[0]
                ids = (
                    probe["input_ids"][:prefix]
                    + ids
                    + probe["input_ids"][body[-1] + 1 :]
                )
            offsets = [(start + prefix, end + prefix) for start, end in offsets]
        return {
            "input_ids": mx.array([ids]),
            "pixel_values": pixels,
            "image_grid_thw": grids,
            "image_offsets": offsets,
        }
