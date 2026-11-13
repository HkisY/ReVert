"""Sources of REAL photos for the builder.

    CocoSource    COCO 2017 photos + one annotated object per image (mask and
                  object name come from the annotations)
    FolderSource  any folder of photos: your own new photos (never seen by any
                  model), a high-resolution set, ... ; masks are free-form
                  brush strokes, there are no object names

Geometry: by default images are NOT upscaled. Large images are downscaled so
the long side is <= size, then everything is centre-cropped to multiples of
64 (what the editors need). Cropping keeps the original camera pixels; only
--allow-upscale resizes small images up (adds interpolation to x_r, which is
then shared by every condition, so it is fair, but avoidable).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from .masks import dilate, freeform_mask

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


@dataclass
class RealItem:
    sid: str
    image: object  # PIL.Image RGB
    mask: np.ndarray | None  # bool HxW, None for instruction editors on folder sources
    obj: str | None
    meta: dict = field(default_factory=dict)
    parts: list | None = None  # two-object samples (E4b): [{"mask", "obj", "ann_id"}, ...]; mask = their union


def fit_to_grid(img, mask=None, size: int = 1024, allow_upscale: bool = False, grid: int = 64):
    """Return (img, mask) with both sides multiples of `grid` and long side <= size."""
    from PIL import Image

    w, h = img.size
    s = size / max(w, h)
    if s < 1 or allow_upscale:
        nw, nh = max(grid, int(round(w * s))), max(grid, int(round(h * s)))
        img = img.resize((nw, nh), Image.LANCZOS if s < 1 else Image.BICUBIC)
        if mask is not None:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 127
        w, h = nw, nh
    cw, ch = (w // grid) * grid, (h // grid) * grid
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    if mask is not None:
        mask = mask[y0 : y0 + ch, x0 : x0 + cw]
    return img, mask


class CocoSource:
    def __init__(self, root: str, split: str = "val2017", size: int = 1024, seed: int = 0, min_area: float = 0.03,
                 max_area: float = 0.25, dilate_px: int = 7, allow_upscale: bool = False, pairs: bool = False,
                 min_gap: int = 24, max_area_ratio: float = 2.0, max_union: float = 0.4,
                 exclude_image_ids: set | None = None):
        from pycocotools.coco import COCO

        ann = os.path.join(root, "annotations", f"instances_{split}.json")
        if not os.path.exists(ann):
            raise FileNotFoundError(f"{ann} not found. Run: python -m revert_audit.data.download --root {root}")
        self.root, self.split, self.size, self.dilate_px, self.allow_upscale = root, split, size, dilate_px, allow_upscale
        self.coco = COCO(ann)
        self.cats = {c["id"]: c["name"] for c in self.coco.loadCats(self.coco.getCatIds())}
        self.min_area, self.max_area = min_area, max_area
        self.pairs, self.min_gap, self.max_area_ratio, self.max_union = pairs, min_gap, max_area_ratio, max_union
        self.exclude = set(exclude_image_ids or ())
        self.order = [i for i in np.random.default_rng(seed).permutation(self.coco.getImgIds()).tolist()
                      if i not in self.exclude]
        self.rng = np.random.default_rng(seed + 1)

    def other_category(self, cat_name: str) -> str:
        names = [n for n in self.cats.values() if n != cat_name]
        return str(self.rng.choice(names))

    def __iter__(self):
        from PIL import Image

        for iid in self.order:
            info = self.coco.loadImgs(iid)[0]
            area_img = info["width"] * info["height"]
            anns = [x for x in self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid, iscrowd=False))
                    if self.min_area <= x["area"] / area_img <= self.max_area]
            if not anns:
                continue
            path = os.path.join(self.root, self.split, info["file_name"])
            if not os.path.exists(path):
                continue
            if self.pairs:
                item = self._pair_item(iid, info, anns, path)
                if item is not None:
                    yield item
                continue
            ann = anns[int(self.rng.integers(len(anns)))]
            img = Image.open(path).convert("RGB")
            m = self.coco.annToMask(ann).astype(bool)
            img, m = fit_to_grid(img, m, self.size, self.allow_upscale)
            m = dilate(m, self.dilate_px)
            if not m.any():
                continue
            yield RealItem(sid=f"coco{iid}", image=img, mask=m, obj=self.cats[ann["category_id"]],
                           meta={"source": "coco", "coco_split": self.split, "coco_image_id": iid, "coco_ann_id": ann["id"]})


    def _pair_item(self, iid, info, anns, path):
        """Two objects whose dilated masks are >= min_gap px apart, areas within
        max_area_ratio of each other, union <= max_union of the image. Each object
        is edited separately from the same real photo (independent edits), so
        reverting one object entirely leaves exactly a one-object forgery: no new
        real/forged boundary and no forged rim (E4b)."""
        from PIL import Image

        if len(anns) < 2:
            return None
        img0 = Image.open(path).convert("RGB")
        cands = []
        for a in anns:
            img, m = fit_to_grid(img0, self.coco.annToMask(a).astype(bool), self.size, self.allow_upscale)
            m = dilate(m, self.dilate_px)
            if m.any():
                cands.append((a, m))
        pairs = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                (a1, m1), (a2, m2) = cands[i], cands[j]
                s1, s2 = int(m1.sum()), int(m2.sum())
                if max(s1, s2) > self.max_area_ratio * min(s1, s2):
                    continue
                if (m1 | m2).mean() > self.max_union:
                    continue
                if (dilate(m1, self.min_gap) & m2).any():
                    continue
                pairs.append((i, j))
        if not pairs:
            return None
        i, j = pairs[int(self.rng.integers(len(pairs)))]
        (a1, m1), (a2, m2) = cands[i], cands[j]
        img, _ = fit_to_grid(img0, None, self.size, self.allow_upscale)
        parts = [{"mask": m1, "obj": self.cats[a1["category_id"]], "ann_id": a1["id"]},
                 {"mask": m2, "obj": self.cats[a2["category_id"]], "ann_id": a2["id"]}]
        return RealItem(sid=f"coco{iid}p", image=img, mask=m1 | m2, obj=None, parts=parts,
                        meta={"source": "coco", "coco_split": self.split, "coco_image_id": iid,
                              "coco_ann_ids": [a1["id"], a2["id"]], "pair": True})


class FolderSource:
    def __init__(self, folder: str, size: int = 1024, seed: int = 0, min_frac: float = 0.03, max_frac: float = 0.25,
                 name: str = "photos", allow_upscale: bool = False):
        files = [p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(IMG_EXT)]
        if not files:
            raise FileNotFoundError(f"no images found under {folder}")
        self.files = [files[i] for i in np.random.default_rng(seed).permutation(len(files))]
        self.size, self.name, self.allow_upscale = size, name, allow_upscale
        self.min_frac, self.max_frac = min_frac, max_frac
        self.rng = np.random.default_rng(seed + 1)

    def __iter__(self):
        import re

        from PIL import Image, ImageOps

        for p in self.files:
            try:
                img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            except OSError:
                continue
            img, _ = fit_to_grid(img, None, self.size, self.allow_upscale)
            w, h = img.size
            m = freeform_mask(h, w, self.rng, self.min_frac, self.max_frac)
            stem = re.sub(r"[^A-Za-z0-9_\-]", "_", os.path.splitext(os.path.basename(p))[0])
            yield RealItem(sid=f"{self.name}_{stem}", image=img, mask=m, obj=None,
                           meta={"source": self.name, "file": os.path.abspath(p)})
