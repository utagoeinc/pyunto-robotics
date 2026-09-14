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

    # --- residential street (assets/delivery.xml) ---
    # Front doors are the landmarks here, and each is a different colour because that is how
    # a person gives the address: "the house with the red door". The four are far apart in
    # RGB and far from the roofs, hedges and grass, so a door is never confused for scenery.
    # Tolerances are tight (30) because the doors differ from each other, not from the walls,
    # and a loose match would let red bleed into the brick-coloured roofs.
    # Measured from a rendered frame, not copied from the material's rgba: emission, the
    # sun angle and MuJoCo's tone mapping all move the final pixel, and a palette written
    # from the XML matched nothing. If a door is retinted, re-measure rather than guess.
    "red door": ((255, 56, 48), 45),
    "blue door": ((48, 141, 255), 45),
    "green door": ((36, 214, 84), 45),
    "yellow door": ((249, 250, 26), 45),
    "postbox": ((191, 64, 51), 26),

    # --- Mars (assets/mars.xml) ---
    # Measured from rendered frames. Tolerances are tight because the ground here is itself a
    # red-brown that a loose red match swallows whole: a first attempt at the beacon caught
    # 14,000 pixels of regolith. The emissive materials keep the hardware separable from it.
    "cache": ((74, 251, 248), 40),
    "beacon": ((36, 235, 100), 45),
    "lander": ((237, 66, 158), 45),

    # --- orchard (assets/orchard.xml) ---
    # Measured from rendered frames. An orchard is green and brown everywhere, so the two
    # landmarks are the colours it does not contain; the crates read bright red and the shed
    # door a strong blue, neither of which appears in bark, leaf, soil or sky.
    "crates": ((250, 74, 62), 45),
    "shed": ((46, 117, 255), 45),

    # --- home / laundry scene (assets/home.xml) ---
    # The washer is white against a white wall, which colour matching cannot see at all -- so
    # the cue is its blue trim ring, exactly as the office doors are found by their orange
    # frames rather than by the leaf.
    "washer": ((51, 115, 204), 45),
    "basket": ((242, 184, 56), 45),
    "counter": ((184, 140, 97), 35),
    "towel_blue": ((89, 168, 224), 45),
    "towel_pink": ((245, 184, 199), 35),

    # --- outdoor patrol scene (assets/campus.xml) ---
    # Foliage and hedges are both green and deliberately separate entries: a canopy is 3 m up
    # and a landmark, a hedge is at knee height and an obstacle, and telling them apart from
    # colour alone needs the two greens to be distinguishable.
    "tree": ((46, 115, 51), 38),
    "hedge": ((56, 107, 56), 30),
    "entrance": ((229, 115, 38), 45),
    "bollard": ((217, 77, 38), 40),
    "building": ((204, 199, 189), 22),
    "window": ((89, 140, 178), 35),

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

    # --- residential street ---
    # Longest-match wins in _canonical, so "red door" beats the bare "door" above and a
    # delivery instruction naming a colour reaches the right house.
    "red door": "red door", "赤いドア": "red door", "赤いドアの家": "red door",
    "red house": "red door", "赤い家": "red door",
    "blue door": "blue door", "青いドア": "blue door", "青いドアの家": "blue door",
    "blue house": "blue door", "青い家": "blue door",
    "green door": "green door", "緑のドア": "green door", "緑のドアの家": "green door",
    "green house": "green door", "緑の家": "green door",
    "yellow door": "yellow door", "黄色いドア": "yellow door", "黄色いドアの家": "yellow door",
    "yellow house": "yellow door", "黄色い家": "yellow door",
    "postbox": "postbox", "post box": "postbox", "mailbox": "postbox",
    "ポスト": "postbox", "郵便ポスト": "postbox",

    # --- Mars ---
    "cache": "cache", "sample cache": "cache", "sample": "cache",
    "サンプル": "cache", "採取装置": "cache", "キャッシュ": "cache",
    "beacon": "beacon", "marker": "beacon", "ビーコン": "beacon", "目印": "beacon",

    # --- orchard ---
    "crates": "crates", "crate": "crates", "apples": "crates", "fruit": "crates",
    "コンテナ": "crates", "かご": "crates", "リンゴ": "crates", "りんご": "crates",
    "収穫物": "crates", "箱": "crates",
    "shed": "shed", "packing shed": "shed", "barn": "shed",
    "小屋": "shed", "倉庫": "shed", "作業場": "shed", "選果場": "shed",
    "lander": "lander", "base": "lander", "着陸機": "lander", "着陸船": "lander",
    "基地": "lander", "ランダー": "lander",

    # --- home / laundry ---
    "washer": "washer", "washing machine": "washer", "drum": "washer",
    "洗濯機": "washer", "ドラム": "washer",
    "basket": "basket", "laundry basket": "basket", "洗濯かご": "basket",
    "洗濯カゴ": "basket", "かご": "basket", "カゴ": "basket",
    "counter": "counter", "vanity": "counter", "washstand": "counter",
    "洗面台": "counter", "カウンター": "counter", "台": "counter",
    "towel": "towel_blue", "タオル": "towel_blue",
    "blue towel": "towel_blue", "青いタオル": "towel_blue",
    "pink towel": "towel_pink", "ピンクのタオル": "towel_pink",

    # --- outdoor patrol ---
    "tree": "tree", "trees": "tree", "木": "tree", "樹木": "tree", "植木": "tree",
    "hedge": "hedge", "bush": "hedge", "生垣": "hedge", "植え込み": "hedge",
    # NOT "door"/"入口"/"出口": those already mean the office doors, and this table is shared
    # by every scene. Adding them here silently re-pointed 「オフィスのドアを開けて」 at the
    # campus building's entrance, because the longest-match rule has no idea which scene is
    # loaded. Only words that are unambiguous across all four scenes belong here.
    "entrance": "entrance", "玄関": "entrance", "エントランス": "entrance",
    "bollard": "bollard", "post": "bollard", "ポール": "bollard", "車止め": "bollard",
    "building": "building", "ビル": "building", "建物": "building", "建屋": "building",
    "window": "window", "窓": "window",

    # --- lunar ---
    "lander": "lander", "着陸機": "lander", "ランダー": "lander",
    "beacon": "beacon", "marker": "beacon", "ビーコン": "beacon", "目印": "beacon",
    "panel": "panel", "solar panel": "panel", "ソーラーパネル": "panel", "パネル": "panel",
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
