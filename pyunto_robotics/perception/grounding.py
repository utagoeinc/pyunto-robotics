"""Finding a named thing in an image.

The robot is told "open the office door" and has to work out which pixels are the door. Two
implementations of the same interface:

  ColorGrounder  Finds objects by their appearance in the scene. Fast, deterministic, and has
                 no model to download - which makes it the right thing to develop the
                 navigation loop against, and a working fallback if the VLM is unavailable.

  VLMGrounder    Asks a local vision-language model where the object is. Open-vocabulary, so it
                 handles instructions the scene was never annotated for.

Both return normalised 0-1 coordinates, matching how modern VLMs report grounding (Qwen3-VL
switched to relative coordinates precisely so results survive resizing).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Something found in an image, in normalised coordinates."""

    label: str
    x: float  # 0-1, centre
    y: float  # 0-1, centre
    confidence: float
    box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1, normalised

    def pixel(self, width: int, height: int) -> tuple[float, float]:
        """Centre in pixels."""
        return self.x * width, self.y * height

    @property
    def area(self) -> float:
        if self.box is None:
            return 0.0
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class Grounder(Protocol):
    """Locates a described object in an image."""

    def find(self, rgb: np.ndarray, description: str) -> list[Detection]:
        """Return detections, most confident first (empty if not found)."""
        ...


# --------------------------------------------------------------------------------------
# Colour-based grounding
# --------------------------------------------------------------------------------------

# Objects in the office are deliberately colour-coded, so hue is a reliable cue here.
# These RGB values were sampled from rendered frames, not taken from the material definitions --
# lighting shifts everything, so the material's nominal colour is not what the camera sees.
#
# Note the whiteboard and the fridge: near-white and near-grey are close to the office walls
# and floor, so they are weak targets for this grounder. That is a real limitation of matching
# on colour, and the reason VLMGrounder exists.
_PALETTE: dict[str, tuple[tuple[int, int, int], int]] = {
    "door": ((205, 145, 60), 55),        # orange frame; overlaps the leaf under office light
    "door_panel": ((190, 135, 86), 40),  # the wooden leaf itself
    "whiteboard": ((250, 250, 244), 8),  # tight: office walls render near (255,255,248)
    "monitor": ((26, 31, 41), 22),
    "plant": ((51, 128, 64), 40),
    "table": ((89, 76, 66), 30),
    "desk": ((140, 102, 71), 30),
    "fridge": ((178, 184, 191), 15),
}

# What a user might say, mapped to what the scene calls it.
_SYNONYMS: dict[str, str] = {
    "door": "door", "doorway": "door", "entrance": "door", "exit": "door",
    "ドア": "door", "扉": "door", "入口": "door", "出口": "door",
    "whiteboard": "whiteboard", "board": "whiteboard", "ホワイトボード": "whiteboard",
    "monitor": "monitor", "screen": "monitor", "display": "monitor", "モニタ": "monitor",
    "plant": "plant", "植物": "plant", "観葉植物": "plant",
    "table": "table", "meeting table": "table", "テーブル": "table",
    "desk": "desk", "机": "desk", "デスク": "desk",
    "fridge": "fridge", "refrigerator": "fridge", "冷蔵庫": "fridge",
}


def _canonical(description: str) -> str | None:
    """Map a free-text description onto a known object name."""
    text = description.lower().strip()
    if text in _SYNONYMS:
        return _SYNONYMS[text]
    # Longest match wins, so "meeting table" beats "table".
    matches = [(len(k), v) for k, v in _SYNONYMS.items() if k in text]
    if matches:
        return max(matches)[1]
    return None


class ColorGrounder:
    """Locates objects by their known colour in the simulated office.

    This is not a perception research contribution; it is a dependable stand-in that lets the
    navigation loop be built and tested without a model in the way. It also stays useful as a
    fallback when the VLM is unavailable or wrong.
    """

    def __init__(self, min_pixels: int = 60):
        self.min_pixels = min_pixels

    def find(self, rgb: np.ndarray, description: str) -> list[Detection]:
        name = _canonical(description)
        if name is None or name not in _PALETTE:
            return []

        target, tol = _PALETTE[name]
        mask = self._mask(rgb, target, tol)
        if name == "door":
            # The frame is a ring; include the leaf so the centroid lands on the doorway.
            panel, ptol = _PALETTE["door_panel"]
            mask = mask | self._mask(rgb, panel, ptol)

        return self._blobs(mask, name, rgb.shape[1], rgb.shape[0])

    @staticmethod
    def _mask(rgb: np.ndarray, target: tuple[int, int, int], tol: int) -> np.ndarray:
        diff = np.abs(rgb.astype(np.int16) - np.array(target, dtype=np.int16))
        return (diff.max(axis=2) <= tol)

    def _blobs(self, mask: np.ndarray, label: str, width: int, height: int) -> list[Detection]:
        """Split a mask into connected components via a column-run scan.

        Deliberately dependency-free: scipy.ndimage.label would do this, but the office has a
        handful of large, well-separated blobs and this keeps the install lean.
        """
        cols = mask.any(axis=0)
        detections: list[Detection] = []
        start: int | None = None

        for x in range(width + 1):
            filled = bool(cols[x]) if x < width else False
            if filled and start is None:
                start = x
            elif not filled and start is not None:
                det = self._blob_at(mask, label, start, x, width, height)
                if det is not None:
                    detections.append(det)
                start = None

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _blob_at(
        self, mask: np.ndarray, label: str, x0: int, x1: int, width: int, height: int
    ) -> Detection | None:
        region = mask[:, x0:x1]
        count = int(region.sum())
        if count < self.min_pixels:
            return None

        ys, xs = np.nonzero(region)
        cx = float(xs.mean() + x0)
        cy = float(ys.mean())
        # Confidence from apparent size: a bigger blob is a nearer, more certain target.
        confidence = float(min(1.0, count / (width * height * 0.02)))
        return Detection(
            label=label,
            x=cx / width,
            y=cy / height,
            confidence=confidence,
            box=(
                float(x0) / width,
                float(ys.min()) / height,
                float(x1) / width,
                float(ys.max()) / height,
            ),
        )


# --------------------------------------------------------------------------------------
# VLM grounding
# --------------------------------------------------------------------------------------

_GROUNDING_PROMPT = (
    "Look at this image from a robot's camera in an office.\n"
    'Find: "{description}".\n\n'
    "Reply with ONLY a JSON array, no other text. Each entry:\n"
    '  {{"label": "<what it is>", "x": <0-1>, "y": <0-1>, "confidence": <0-1>}}\n'
    "x and y are the CENTRE of the object as a fraction of image width and height, where "
    "(0,0) is top-left and (1,1) is bottom-right.\n"
    "If the object is not visible, reply with []."
)


class VLMGrounder:
    """Asks a local vision-language model to locate the object.

    The model is loaded lazily so importing this module stays cheap and the simulator can run
    without any weights on disk.
    """

    def __init__(self, model_id: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                 max_tokens: int = 256):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._model = None
        self._processor = None
        self._config = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_vlm import load  # noqa: PLC0415 - optional heavy dependency
            from mlx_vlm.utils import load_config  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "mlx-vlm is not installed. Install the extra: uv pip install -e '.[llm]'"
            ) from e
        log.info("loading grounding model %s (first run downloads weights)", self.model_id)
        self._model, self._processor = load(self.model_id)
        self._config = load_config(self.model_id)

    def find(self, rgb: np.ndarray, description: str) -> list[Detection]:
        self._load()
        from mlx_vlm import generate  # noqa: PLC0415
        from mlx_vlm.prompt_utils import apply_chat_template  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        image = Image.fromarray(rgb)
        prompt = apply_chat_template(
            self._processor, self._config, _GROUNDING_PROMPT.format(description=description),
            num_images=1,
        )
        reply = generate(
            self._model, self._processor, prompt, [image],
            max_tokens=self.max_tokens, verbose=False,
        )
        text = reply if isinstance(reply, str) else getattr(reply, "text", str(reply))
        return _parse_detections(text)


def _parse_detections(text: str) -> list[Detection]:
    """Pull detections out of a model reply.

    Models wrap JSON in prose or code fences often enough that locating the array by bracket
    matching is more reliable than trusting the whole reply to parse.
    """
    match = re.search(r"\[.*]", text, re.DOTALL)
    if not match:
        return []
    try:
        entries = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("grounding model returned unparseable JSON: %s", text[:200])
        return []

    out: list[Detection] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            x = float(entry["x"])
            y = float(entry["y"])
        except (KeyError, TypeError, ValueError):
            continue
        # Some models emit 0-1000 or pixel coordinates despite being asked for 0-1.
        if x > 1.0 or y > 1.0:
            scale = 1000.0 if max(x, y) <= 1000.0 else max(x, y)
            x, y = x / scale, y / scale
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        out.append(
            Detection(
                label=str(entry.get("label", "object")),
                x=x,
                y=y,
                confidence=float(entry.get("confidence", 0.5)),
            )
        )
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out
