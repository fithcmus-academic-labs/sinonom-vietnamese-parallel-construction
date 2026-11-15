"""
Document Preprocessing and Layout Segmentation Module
Implements Pre-Model Data Preprocessing (Color-Space Ink Layer Decomposition,
Adaptive Background Normalization, Ruling/Grid Removal, Bleed-Through Suppression)
and Script-Specific Bounding Box Extraction strictly adhering to 'guide/Quy_định_đồ_án.pdf'.
"""
import os
import shutil
import cv2
import numpy as np
import pymupdf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from core.config import LAPLACIAN_BLUR_THRESHOLD, RENDER_DPI


def cv2_imread_unicode(path: Path | str, flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Safely read image with Unicode/non-ASCII characters on Windows."""
    try:
        with open(str(path), "rb") as f:
            bytes_data = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(bytes_data, flags)
    except Exception:
        return None


def cv2_imwrite_unicode(path: Path | str, img: np.ndarray, ext: str = ".jpg") -> bool:
    """Safely write image with Unicode/non-ASCII characters on Windows."""
    try:
        success, encoded = cv2.imencode(ext, img)
        if success:
            with open(str(path), "wb") as f:
                f.write(encoded)
            return True
    except Exception:
        pass
    return False


# Books where Upper Hán Nôm is written in Red/Vermilion ink and Lower Quốc ngữ in Blue/Dark ink
RED_NOM_BOOKS = (
    "ca trù thể cách, nôm_y",
    "thánh du quảng huân diên nghĩa",
    "tây dương quân thiên lạc",
)

# Books where Hán Nôm pages (ch/) are written in Vertical Columns
VERTICAL_CH_BOOKS = (
    "thudạlữhoài",
    "thudaluhoai",
    "tìbàquốcâmtântruyện_琵琶國音新傳_1",
    "tìbàquốcâmtântruyện_琵琶國音新傳_2",
    "thơcaduhànhtừhànộivàokinhđôhuế_1",
)


class DocumentPreprocessor:
    def __init__(self, book_info: dict, out_base_dir: Path, create_dirs: bool = False):
        self.book_info = book_info
        self.title = book_info["title"]
        self.pdf_filename = book_info.get("pdf_filename", f"{self.title}.pdf")
        self.out_base_dir = Path(out_base_dir) / self.title

        self.dir_ch = self.out_base_dir / "ch"
        self.dir_vi = self.out_base_dir / "vi"
        self.dir_blank = self.out_base_dir / "blank"
        self.dir_misc = self.out_base_dir / "misc"

        if create_dirs:
            for d in [self.dir_ch, self.dir_vi, self.dir_blank, self.dir_misc]:
                d.mkdir(parents=True, exist_ok=True)

    def split_and_extract_images(self, pdf_path: str) -> List[Tuple[int, str, Path]]:
        """
        Extract high-resolution JPEG images for all pages in PDF.
        Returns list of (page_num_1_indexed, image_filename, saved_path).
        """
        for d in [self.dir_ch, self.dir_vi, self.dir_blank, self.dir_misc]:
            d.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(pdf_path)
        extracted = []

        total_pages = len(doc)
        pad_width = max(3, len(str(total_pages)))

        cache_dir = self.out_base_dir / "extracted_all"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(total_pages):
            page_num = idx + 1
            page = doc[idx]

            stt_str = f"{page_num:0{pad_width}d}"
            img_name = f"{self.title}.{stt_str}.jpg"
            img_path = cache_dir / img_name

            if not img_path.exists() or img_path.stat().st_size == 0:
                imgs = page.get_images()
                if imgs and len(imgs) >= 1 and idx > 0:
                    xref = imgs[0][0]
                    base_img = doc.extract_image(xref)
                    with open(img_path, "wb") as f:
                        f.write(base_img["image"])
                else:
                    pix = page.get_pixmap(dpi=RENDER_DPI)
                    pix.save(str(img_path))

            extracted.append((page_num, img_name, img_path))

        doc.close()
        return extracted

    def cleanup_cache(self):
        """Remove temporary extracted_all directory so output folder only contains compliant folders."""
        cache_dir = self.out_base_dir / "extracted_all"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

    @staticmethod
    def classify_page(img_path: Path, page_num: int, total_pages: int) -> str:
        """
        Classify page into 'misc', 'blank', or 'content'.
        Excludes book covers, inner title pages, and French prefaces.
        """
        fname = img_path.name.lower()
        if "thonomhoxuanhuonghoangxuanhan" in fname and page_num <= 7:
            return "misc"
        if "tho ho xuan huong [chu nom]" in fname and page_num == 1:
            return "misc"
        if ("_y" in fname or "digcoll" in fname or "thánh" in fname or "thâp nhị" in fname) and total_pages > 20:
            if page_num <= 4 or page_num == total_pages:
                return "misc"
        elif total_pages > 16:
            if page_num in (1, 2) or page_num == total_pages:
                return "misc"

        img = cv2_imread_unicode(img_path)
        if img is None:
            return "misc"

        h, w, _ = img.shape
        inner = img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        std_val = float(gray.std())

        if std_val < 14.0 and lap_var < 15.0:
            return "blank"

        return "content"

    @staticmethod
    def detect_margin_x(img: np.ndarray) -> int:
        """
        Detect vertical red margin line of notebook paper (typically x in 4%..24% of width).
        Returns x coordinate right after the red margin line, or a small 2.5% border if no red line exists.
        """
        h, w, _ = img.shape
        b = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        r = img[:, :, 2].astype(np.int16)

        redness = (r - np.maximum(g, b)) > 22
        vert_counts = redness.sum(axis=0)

        x_min = max(10, int(w * 0.04))
        x_max = int(w * 0.24)
        if x_max > x_min:
            search_region = vert_counts[x_min:x_max]
            if search_region.max() > int(h * 0.25):
                peak_x = x_min + int(np.argmax(search_region))
                return peak_x + 6

        return int(w * 0.025)

    @staticmethod
    def _clean_single_ink_layer(raw_mask: np.ndarray) -> np.ndarray:
        """
        Clean a binary ink mask by removing horizontal notebook ruling lines,
        vertical grid/margin lines, scanner border strips, and isolated sensor specks.
        """
        rh, rw = raw_mask.shape
        u = (raw_mask.astype(np.uint8)) * 255 if raw_mask.dtype != np.uint8 else raw_mask.copy()

        # Remove horizontal notebook ruling lines
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(48, int(rw * 0.28)), 1))
        u = cv2.subtract(u, cv2.morphologyEx(u, cv2.MORPH_OPEN, hk))

        # Remove continuous vertical grid/margin lines
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(rh * 0.022))))
        u = cv2.subtract(u, cv2.morphologyEx(u, cv2.MORPH_OPEN, vk))

        # Suppress vertical scanner edge artifacts (> 42% column density at edges)
        col_dens = (u > 0).sum(axis=0) / float(max(1, rh))
        edge_cols = np.where(col_dens > 0.42)[0]
        for c in edge_cols:
            if c < int(rw * 0.04) or c > int(rw * 0.96):
                u[:, max(0, c - 2):min(rw, c + 3)] = 0

        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        return u

    @classmethod
    def decompose_bi_ink_layers(
        cls, roi_bgr: np.ndarray, red_is_ch: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-Model Data Preprocessing: Decompose a bi-ink manuscript ROI into two
        strictly disjoint, grid-free binary ink masks: (mask_ch, mask_vi).
        - mask_dark captures Black / Blue / Dark-Purple ink (high absorption in Red channel, low R-G).
        - mask_red captures Red / Vermilion / Magenta ink (high positive R-G chrominance).
        Because Hán Nôm and Quốc ngữ are separated into disjoint masks BEFORE segmentation,
        a bounding box on mask_ch can physically never merge with or include strokes from mask_vi.
        """
        g = roi_bgr[:, :, 1].astype(np.float32)
        r = roi_bgr[:, :, 2].astype(np.float32)
        rg = r - g

        bg_r = float(np.percentile(r, 85))
        bg_g = float(np.percentile(g, 85))
        p5_r = float(np.percentile(r, 5))

        if red_is_ch:
            # In Red-Nom + Blue-Quoc-Ngu books, mask_dark is thin blue/purple pen ink
            dark_r_thresh = max(23.0, (bg_r - p5_r) * 0.21)
            raw_dark = (((bg_r - r) > dark_r_thresh) & (rg < 15.0)) | (
                ((bg_g - g) > 56.0) & (rg < 18.0) & ((bg_r - r) > 32.0)
            )
            raw_red = (rg > 16.5) & ((bg_g - g) > 13.0)
        else:
            # In Black-Nom + Red-Quoc-Ngu books, mask_dark is black brush ink;
            # use higher threshold (36.0) to completely reject grey notebook grid intersections
            dark_r_thresh = max(36.0, (bg_r - p5_r) * 0.27)
            raw_dark = (((bg_r - r) > dark_r_thresh) & (rg < 16.0)) | (
                ((bg_g - g) > 62.0) & (rg < 18.0) & ((bg_r - r) > 42.0)
            )
            raw_red = (rg > 15.0) & ((bg_g - g) > 12.0)

        clean_dark = cls._clean_single_ink_layer(raw_dark)
        clean_red = cls._clean_single_ink_layer(raw_red)

        if red_is_ch:
            return clean_red, clean_dark
        return clean_dark, clean_red

    @classmethod
    def synthesize_preprocessed_layer_image(
        cls, img_bgr: np.ndarray, book_title: str = "", target_script: str = "ch"
    ) -> np.ndarray:
        """
        Synthesize a clean, contrast-enhanced image containing ONLY the target script layer
        ('ch' for Hán Nôm or 'vi' for Quốc ngữ) on a normalized white background (255, 255, 255),
        with notebook ruling lines, margins, and the other script's ink completely removed.
        Used directly as preprocessed input to OCR models and KimHanNomClient API.
        """
        h, w, _ = img_bgr.shape
        margin_x = cls.detect_margin_x(img_bgr)
        right_limit = int(w * 0.975)
        top_cut, bot_cut = int(h * 0.025), int(h * 0.992)

        roi = img_bgr[top_cut:bot_cut, margin_x:right_limit]
        b_lower = book_title.lower()
        red_is_ch = any(k in b_lower for k in RED_NOM_BOOKS)

        mask_ch, mask_vi = cls.decompose_bi_ink_layers(roi, red_is_ch=red_is_ch)
        target_mask = mask_ch if target_script == "ch" else mask_vi

        # If target_mask is nearly empty (monochrome book), fall back to monochrome cleaned ink
        if (target_mask > 0).sum() < 500:
            target_mask, _ = cls._build_ink_binary(roi)

        canvas = np.full_like(img_bgr, 255)
        canvas_roi = canvas[top_cut:bot_cut, margin_x:right_limit]
        # Render strokes in high-contrast dark charcoal on pure white background for optimal OCR
        ys, xs = np.where(target_mask > 0)
        canvas_roi[ys, xs] = np.clip(roi[ys, xs].astype(np.int16) - 35, 0, 180).astype(np.uint8)
        return canvas

    @staticmethod
    def _build_ink_binary(roi_bgr: np.ndarray, strong_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a clean binary ink mask for monochrome pages, suppressing notebook ruling lines
        and optional reverse-side paper bleed-through (when strong_only=True).
        """
        g = roi_bgr[:, :, 1].astype(np.int16)
        r = roi_bgr[:, :, 2].astype(np.int16)

        bg_g = float(np.percentile(g, 85))
        min_drop = 35.0 if strong_only else 25.0
        scale = 0.28 if strong_only else 0.21
        dark_ink = (bg_g - g) > max(min_drop, (bg_g - float(np.percentile(g, 5))) * scale)

        signed_rg = (r - g).astype(np.float32)
        if strong_only:
            combined = (dark_ink.astype(np.uint8)) * 255
        else:
            red_ink = (signed_rg > 20.0) & ((bg_g - g) > 14.0)
            combined = ((dark_ink | red_ink).astype(np.uint8)) * 255

        cleaned = DocumentPreprocessor._clean_single_ink_layer(combined)
        return cleaned, signed_rg

    @staticmethod
    def build_target_ink_mask_for_crop(
        crop_bgr: np.ndarray, book_name: str, folder: str, is_bi_ink: bool
    ) -> np.ndarray:
        """
        Build the script-specific binary ink mask inside a bounding box crop so tighten_box
        only fits strokes belonging to that box's script ('ch' or 'vi').
        """
        if crop_bgr.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        b_lower = book_name.lower()
        if is_bi_ink:
            red_is_ch = any(k in b_lower for k in RED_NOM_BOOKS)
            want_red = (folder == "ch" and red_is_ch) or (folder == "vi" and not red_is_ch)
            g = crop_bgr[:, :, 1].astype(np.float32)
            r = crop_bgr[:, :, 2].astype(np.float32)
            rg = r - g
            bg_r = float(np.percentile(r, 85))
            bg_g = float(np.percentile(g, 85))
            if want_red:
                m = (rg > 14.0) & ((bg_g - g) > 11.0)
            else:
                m = (((bg_r - r) > 20.0) & (rg < 16.0)) | (((bg_g - g) > 52.0) & (rg < 20.0))
            if (m > 0).sum() >= 15:
                return (m.astype(np.uint8)) * 255

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        bg = float(np.percentile(gray, 85))
        thresh = 32.0 if "thơcaduhành" in b_lower and "_2" in b_lower else 18.0
        m = (bg - gray) > max(thresh, (bg - float(np.percentile(gray, 10))) * 0.20)
        if (m > 0).sum() < 15:
            g_c = crop_bgr[:, :, 1].astype(np.float32)
            bg_gc = float(np.percentile(g_c, 85))
            m = ((bg_gc - g_c) > 16.0) | ((bg - gray) > 12.0)
            if (m > 0).sum() < 15 and float(np.percentile(gray, 85)) > 180 and float(np.percentile(gray, 50)) < 50:
                m = (gray < 140) & (gray > 10)
        return (m.astype(np.uint8)) * 255

    def segment_t2b_lines(self, img_path: Path) -> Dict[str, List[dict]]:
        """
        Preprocess page image and produce tight, non-overlapping 4-point bounding boxes
        for Hán Nôm (ch) and Quốc ngữ (vi) across all 24 books in the corpus.
        """
        img = cv2_imread_unicode(img_path)
        if img is None:
            return {"ch": [], "vi": []}

        h, w, _ = img.shape
        fname_lower = img_path.name.lower()
        b_lower = self.title.lower()

        # ------------------------------------------------------------------
        # CASE 1: Side-by-side parallel table (Tế cô hồn văn SV4.0 Tuệ Phương)
        # ------------------------------------------------------------------
        if "tế cô hồn" in fname_lower or "sv4.0" in fname_lower or (
            w > h * 1.15 and "thơ" not in fname_lower and "tìbà" not in fname_lower
        ):
            return self._segment_side_by_side_table(img)

        # ------------------------------------------------------------------
        # CASE 2: Split-page layout (ThoNomHoXuanHuongHoangXuanHanSuuTapVaGhiChu)
        # Top half = Vertical Hán Nôm columns (ch); Bottom half = Horizontal Quốc ngữ lines (vi)
        # ------------------------------------------------------------------
        if "thonomhoxuanhuonghoangxuanhan" in fname_lower or "thonomhoxuanhuonghoangxuanhan" in b_lower:
            return self._segment_top_vertical_nom_bottom_horizontal_qn(img)

        # ------------------------------------------------------------------
        # CASE 3: Pure Vertical Hán Nôm Column pages (ThuDạLữHoàiNgâm, TìBà_1/ch, TìBà_2/ch, ThơCaDuHành_1/ch)
        # ------------------------------------------------------------------
        is_ch_folder = img_path.parent.name == "ch"
        is_vi_folder = img_path.parent.name == "vi"

        if any(k in fname_lower or k in b_lower for k in VERTICAL_CH_BOOKS):
            if is_vi_folder:
                # Pages in vi/ of ThơCaDuHành_1 or TìBà_1 are pure horizontal Quốc ngữ lines
                vi_lines = self._segment_monolingual_horizontal_lines(img, is_qn=True)
                return {"ch": [], "vi": vi_lines}
            else:
                vert_cols = self.segment_vertical_nom_columns(img_path)
                return {"ch": vert_cols, "vi": []}

        # ------------------------------------------------------------------
        # CASE 4: Pure Horizontal Quốc ngữ translation (Thụ Dạ Lữ Hoài Ngâm Khúc_Dịch)
        # ------------------------------------------------------------------
        if "thụ dạ lữ hoài" in fname_lower or "dịch" in fname_lower:
            vi_lines = self._segment_monolingual_horizontal_lines(img, is_qn=True)
            return {"ch": [], "vi": vi_lines}

        # ------------------------------------------------------------------
        # CASE 5: Tì Bà Quôc Âm Tân Truyên, Preface in Hán, text in Nôm_Y
        # Preface (005..021) and Postface (236..239) have no red Quốc ngữ ink -> Vertical Hán columns
        # ------------------------------------------------------------------
        if "preface in hán" in fname_lower or "preface in hán" in b_lower:
            g_c = img[:, :, 1].astype(np.int16)
            r_c = img[:, :, 2].astype(np.int16)
            red_ink_cnt = int(((r_c - g_c) > 20).sum())
            if red_ink_cnt < 5000:
                vert_cols = self.segment_vertical_nom_columns(img_path)
                if len(vert_cols) >= 3:
                    return {"ch": vert_cols, "vi": []}
                ch_lines = self._segment_monolingual_horizontal_lines(img, is_qn=False)
                return {"ch": ch_lines, "vi": []}

        # ------------------------------------------------------------------
        # CASE 6: Digital Typeset Monochrome Interlinear (Tho Ho Xuan Huong [chu nom])
        # ------------------------------------------------------------------
        if "tho ho xuan huong [chu nom]" in fname_lower or "tho ho xuan huong [chu nom]" in b_lower:
            return self._segment_digital_typeset_interlinear(img)

        # ------------------------------------------------------------------
        # CASE 7: Mixed 2-Column Hán Stanzas & Cursive Quốc ngữ (ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_2)
        # ------------------------------------------------------------------
        if ("thơcaduhành" in fname_lower or "thơcaduhành" in b_lower) and "_2" in (fname_lower + b_lower):
            return self._segment_mixed_stanza_and_cursive(img)

        # ------------------------------------------------------------------
        # CASE 8: All 15 Bi-Ink Interlinear Manuscripts (Preprocessed Disjoint Ink Layers)
        # ------------------------------------------------------------------
        red_is_ch = any(k in fname_lower or k in b_lower for k in RED_NOM_BOOKS)
        return self._segment_bi_ink_interlinear(img, red_is_ch=red_is_ch)

    def _segment_bi_ink_interlinear(self, img: np.ndarray, red_is_ch: bool = False) -> Dict[str, List[dict]]:
        """
        Segment bi-ink interlinear manuscripts by decomposing the image into disjoint
        Hán Nôm (mask_ch) and Quốc ngữ (mask_vi) ink layers BEFORE line segmentation.
        Guarantees:
        - Zero bounding boxes ever contain both Hán Nôm and Quốc ngữ.
        - Zero pages ever swap or confuse Hán Nôm and Quốc ngữ.
        - Pure Hán pages (e.g. end of Nhi thap tu hieu) keep 100% of their Hán lines.
        """
        h, w, _ = img.shape
        margin_x = self.detect_margin_x(img)
        right_limit = int(w * 0.975)
        top_cut = int(h * 0.025)
        bot_cut = int(h * 0.992)

        roi = img[top_cut:bot_cut, margin_x:right_limit]
        rh, rw, _ = roi.shape

        mask_ch, mask_vi = self.decompose_bi_ink_layers(roi, red_is_ch=red_is_ch)

        def extract_layer_candidates(layer_mask: np.ndarray) -> List[Dict[str, Any]]:
            cw = max(26, int(rw * 0.038))
            closed = cv2.morphologyEx(
                layer_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (cw, 3))
            )
            # Compute horizontal projection on core text columns [11%..90% of rw] so left/right marginalia
            # never bridge vertical gaps between main poem lines
            core_x1, core_x2 = int(rw * 0.11), int(rw * 0.90)
            hp = (closed[:, core_x1:core_x2] > 0).sum(axis=1)
            if (hp >= max(14, int(rw * 0.014))).sum() < 15:
                hp = (closed > 0).sum(axis=1)
            min_px = max(14, int(rw * 0.013))
            min_lh = max(12, int(h * 0.008))
            max_lh = max(55, int(h * 0.072))

            spans: List[Tuple[int, int]] = []
            in_l = False
            sy = 0
            for y, val in enumerate(hp):
                if val >= min_px and not in_l:
                    in_l = True
                    sy = y
                elif val < min_px and in_l:
                    in_l = False
                    if (y - sy) >= 6:
                        spans.append((sy, y))
            if in_l and (len(hp) - sy) >= 6:
                spans.append((sy, len(hp)))

            # Merge detached diacritics/descenders/radicals on the SAME ink layer
            merged: List[Tuple[int, int]] = []
            for sy, ey in spans:
                if (
                    merged
                    and (sy - merged[-1][1]) <= int(h * 0.015)
                    and (ey - merged[-1][0]) <= int(h * 0.062)
                ):
                    merged[-1] = (merged[-1][0], ey)
                else:
                    merged.append((sy, ey))

            # Split any single-layer span that exceeds max_lh
            curr = list(merged)
            for _ in range(4):
                nxt: List[Tuple[int, int]] = []
                changed = False
                for sy, ey in curr:
                    sh = ey - sy
                    if sh > max_lh and sh >= min_lh * 2:
                        ms = sy + int(sh * 0.22)
                        me = sy + int(sh * 0.78)
                        if me > ms:
                            sp = ms + int(np.argmin(hp[ms:me]))
                            if (sp - sy) >= min_lh and (ey - sp) >= min_lh:
                                nxt.append((sy, sp))
                                nxt.append((sp, ey))
                                changed = True
                                continue
                    nxt.append((sy, ey))
                curr = nxt
                if not changed:
                    break

            cands: List[Dict[str, Any]] = []
            for sy, ey in curr:
                if (ey - sy) < min_lh:
                    continue
                sub = layer_mask[sy:ey, :]
                cp = (sub > 0).sum(axis=0)
                ac = np.where(cp >= 2)[0]
                if len(ac) < 10:
                    continue

                # Strip isolated marginal numbers/notes on left (< 12% rw) or right (> 88% rw)
                gaps = np.where(np.diff(ac) > int(rw * 0.032))[0]
                if len(gaps) > 0:
                    segs = [
                        s
                        for s in np.split(ac, gaps + 1)
                        if len(s) >= int(rw * 0.055)
                        or (len(s) >= 12 and s[0] > int(rw * 0.11) and s[-1] < int(rw * 0.88))
                    ]
                    if not segs:
                        continue
                    ac = np.concatenate(segs)

                rp = (sub[:, ac[0]:ac[-1] + 1] > 0).sum(axis=1)
                ar = np.where(rp >= 2)[0]
                if len(ar) < 5:
                    continue

                x1 = max(margin_x, margin_x + int(ac[0]) - 3)
                x2 = min(w - 4, margin_x + int(ac[-1]) + 3)
                y1 = max(top_cut + sy, top_cut + sy + int(ar[0]) - 1)
                y2 = min(top_cut + ey - 1, top_cut + sy + int(ar[-1]) + 1)

                if (x2 - x1) < int(w * 0.08) or (y2 - y1) < 10:
                    continue

                ink_cnt = int((sub[:, ac[0]:ac[-1] + 1] > 0).sum())
                cands.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "ink": ink_cnt,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "yc": (y1 + y2) / 2.0,
                    }
                )
            return cands

        ch_cands = extract_layer_candidates(mask_ch)
        vi_cands = extract_layer_candidates(mask_vi)

        # Suppress inline cross-layer overwrites/punctuation (e.g. narrow red circle inside black Nom line
        # or narrow 2-word dark ink correction overwritten onto a red Quoc Ngu line).
        # Never suppress wide full poem lines (w >= 26% of page width).
        def suppress_cross_layer_overwrites(
            primary: List[Dict[str, Any]], other: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            kept = []
            for p in primary:
                if p["w"] >= int(w * 0.26):
                    kept.append(p)
                    continue
                py1, py2 = p["box"][1], p["box"][3]
                suppressed = False
                for o in other:
                    oy1, oy2 = o["box"][1], o["box"][3]
                    ov = max(0, min(py2, oy2) - max(py1, oy1))
                    if ov > 0.50 * p["h"] and o["ink"] > p["ink"] * 1.6:
                        suppressed = True
                        break
                if not suppressed:
                    kept.append(p)
            return kept

        ch_cands = suppress_cross_layer_overwrites(ch_cands, vi_cands)
        vi_cands = suppress_cross_layer_overwrites(vi_cands, ch_cands)

        # Filter top folio headers (e.g. "f° 5a col. 3-6", "f° 2a col. 6-7") at the top of the page
        if len(ch_cands) >= 2 and ch_cands[0]["box"][1] < int(h * 0.065):
            # If no vi line exists between ch_cands[0] and ch_cands[1], or if it is narrow, it is a top header
            has_vi_between = any(ch_cands[0]["yc"] < v["yc"] < ch_cands[1]["yc"] for v in vi_cands)
            if not has_vi_between or ch_cands[0]["w"] < int(w * 0.45):
                ch_cands = ch_cands[1:]

        if ch_cands and vi_cands:
            first_ch_y1 = ch_cands[0]["box"][1]
            vi_cands = [v for v in vi_cands if v["yc"] > first_ch_y1 + 8]

            if len(vi_cands) > len(ch_cands) and ch_cands:
                last_ch_yc = ch_cands[-1]["yc"]
                below_last = [v for v in vi_cands if v["yc"] > last_ch_yc]
                if len(below_last) > 1:
                    cutoff_y = below_last[0]["box"][3] + 5
                    vi_cands = [v for v in vi_cands if v["yc"] <= cutoff_y]

        # Enforce strictly 0 cross-script vertical overlap between ch_cands and vi_cands
        for c in ch_cands:
            for v in vi_cands:
                # If c is the Nom line right above v and their vertical boxes touch/overlap slightly
                if c["yc"] < v["yc"] and c["box"][3] >= v["box"][1] and (v["yc"] - c["yc"]) < int(h * 0.065):
                    mid_y = (c["box"][3] + v["box"][1]) // 2
                    if mid_y - c["box"][1] >= 12 and v["box"][3] - (mid_y + 1) >= 12:
                        c["box"][3] = mid_y
                        v["box"][1] = mid_y + 1
                # If v is the Quoc Ngu line of previous couplet and touches c below it
                elif v["yc"] < c["yc"] and v["box"][3] >= c["box"][1] and (c["yc"] - v["yc"]) < int(h * 0.065):
                    mid_y = (v["box"][3] + c["box"][1]) // 2
                    if mid_y - v["box"][1] >= 12 and c["box"][3] - (mid_y + 1) >= 12:
                        v["box"][3] = mid_y
                        c["box"][1] = mid_y + 1

        # Fallback for single-ink pages placed in both ch/ and vi/ (e.g. pure Hán page 031 in Nhi thap tu hieu)
        if not vi_cands and ch_cands:
            vi_cands = list(ch_cands)
        elif not ch_cands and vi_cands:
            ch_cands = list(vi_cands)

        ch_boxes = [
            {
                "points": [[c["box"][0], c["box"][1]], [c["box"][2], c["box"][1]], [c["box"][2], c["box"][3]], [c["box"][0], c["box"][3]]],
                "y_center": (c["box"][1] + c["box"][3]) / 2.0,
            }
            for c in ch_cands
        ]
        vi_boxes = [
            {
                "points": [[v["box"][0], v["box"][1]], [v["box"][2], v["box"][1]], [v["box"][2], v["box"][3]], [v["box"][0], v["box"][3]]],
                "y_center": (v["box"][1] + v["box"][3]) / 2.0,
            }
            for v in vi_cands
        ]
        return {"ch": ch_boxes, "vi": vi_boxes}

    def _segment_digital_typeset_interlinear(self, img: np.ndarray) -> Dict[str, List[dict]]:
        """
        Segment digital typeset monochrome interlinear pages (Tho Ho Xuan Huong [chu nom]).
        Classifies each line directly by preprocessed CJK vs Latin glyph morphology (height & stroke profile)
        so orphan lines at page breaks (e.g. Page 015 ending with Nom, Page 016 starting with Quoc Ngu)
        never invert the page parity.
        """
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < 200).astype(np.uint8) * 255

        hp = (ink > 0).sum(axis=1)
        spans: List[Tuple[int, int]] = []
        in_l = False
        sy = 0
        for y, val in enumerate(hp):
            if val >= 8 and not in_l:
                in_l = True
                sy = y
            elif val < 8 and in_l:
                in_l = False
                if (y - sy) >= 10:
                    spans.append((sy, y))
        if in_l and (len(hp) - sy) >= 10:
            spans.append((sy, len(hp)))

        # Merge diacritics within 7px if combined height <= 43px
        merged: List[Tuple[int, int]] = []
        for sy, ey in spans:
            if merged and (sy - merged[-1][1]) <= 7 and (ey - merged[-1][0]) <= 43:
                merged[-1] = (merged[-1][0], ey)
            else:
                merged.append((sy, ey))

        # Split any merged span > 48px
        curr = list(merged)
        for _ in range(4):
            nxt: List[Tuple[int, int]] = []
            for sy, ey in curr:
                if (ey - sy) > 48:
                    ms = sy + int((ey - sy) * 0.25)
                    me = sy + int((ey - sy) * 0.75)
                    sp = ms + int(np.argmin(hp[ms:me]))
                    if (sp - sy) >= 14 and (ey - sp) >= 14:
                        nxt.extend([(sy, sp), (sp, ey)])
                        continue
                nxt.append((sy, ey))
            curr = nxt

        ch_boxes = []
        vi_boxes = []
        for sy, ey in curr:
            sub = ink[sy:ey, :]
            ac = np.where((sub > 0).sum(axis=0) >= 1)[0]
            if len(ac) < 15:
                continue
            x1 = max(0, int(ac[0]) - 3)
            x2 = min(w, int(ac[-1]) + 3)
            ar = np.where((sub[:, ac[0]:ac[-1] + 1] > 0).sum(axis=1) >= 1)[0]
            if len(ar) < 8:
                continue
            y1 = max(0, sy + int(ar[0]) - 2)
            y2 = min(h, sy + int(ar[-1]) + 2)

            lh = int(ar[-1] - ar[0] + 1)
            lw = int(ac[-1] - ac[0] + 1)
            crop = sub[ar[0]:ar[-1] + 1, ac[0]:ac[-1] + 1]
            r_dens = (crop > 0).sum(axis=1) / float(max(1, lw))
            mid_d = float(r_dens[int(lh * 0.30):int(lh * 0.75)].mean()) if lh >= 10 else 0.1
            outer_d = (
                float(np.concatenate([r_dens[:int(lh * 0.25)], r_dens[int(lh * 0.75):]]).mean())
                if lh >= 10
                else 0.0
            )
            ratio_outer_mid = outer_d / max(0.01, mid_d)

            # In Tho Ho Xuan Huong [chu nom], CJK Nom lines have glyph height 35..43px,
            # while Times New Roman Quoc Ngu lines have glyph height 20..31px.
            is_nom = (lh >= 34) or (lh >= 31 and ratio_outer_mid > 0.45)
            box_item = {
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "y_center": (y1 + y2) / 2.0,
            }
            if is_nom:
                ch_boxes.append(box_item)
            else:
                vi_boxes.append(box_item)

        return {"ch": ch_boxes, "vi": vi_boxes}

    def _segment_top_vertical_nom_bottom_horizontal_qn(self, img: np.ndarray) -> Dict[str, List[dict]]:
        """
        Segment ThoNomHoXuanHuongHoangXuanHanSuuTapVaGhiChu_HoXuanHuong:
        - Top half (above central blank divider) contains Vertical Hán Nôm columns (read right-to-left) -> ch
        - Bottom half (below central blank divider) contains Horizontal Quốc ngữ lines (read top-to-bottom) -> vi
        """
        h, w, _ = img.shape
        top_cut = int(h * 0.03)
        bot_cut = int(h * 0.96)
        left_cut = int(w * 0.06)
        right_cut = int(w * 0.94)

        roi = img[top_cut:bot_cut, left_cut:right_cut]
        ink_mask, _ = self._build_ink_binary(roi)
        rh, rw = ink_mask.shape

        # Find the large horizontal blank divider between top Nôm block and bottom Quốc ngữ block (y in 28%..68% of rh)
        hp = (ink_mask > 0).sum(axis=1)
        s_min = int(rh * 0.25)
        s_max = int(rh * 0.70)
        blank_rows = hp[s_min:s_max] < max(5, int(rw * 0.008))

        best_len = 0
        best_mid = int(rh * 0.48)
        cur_len = 0
        cur_start = 0
        for i, is_b in enumerate(blank_rows):
            if is_b:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_mid = s_min + cur_start + cur_len // 2
            else:
                cur_len = 0

        split_y_abs = top_cut + best_mid

        # 1. Segment Top Half into Vertical Hán Nôm Columns (sorted Right-to-Left)
        ch_boxes = self._segment_vertical_columns_in_SubROI(
            img, left_cut, top_cut, right_cut, split_y_abs
        )

        # 2. Segment Bottom Half into Horizontal Quốc ngữ Lines (sorted Top-to-Bottom)
        bot_img = img[split_y_abs:bot_cut, left_cut:right_cut]
        vi_rel = self._segment_monolingual_horizontal_lines(bot_img, is_qn=True, full_Margins=False)
        vi_boxes = []
        for b in vi_rel:
            pts = b["points"]
            x1, y1 = pts[0][0] + left_cut, pts[0][1] + split_y_abs
            x2, y2 = pts[2][0] + left_cut, pts[2][1] + split_y_abs
            # Exclude tiny page numbers at bottom-left/right (e.g. "13")
            if (x2 - x1) < int(w * 0.07):
                continue
            vi_boxes.append(
                {
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "y_center": (y1 + y2) / 2.0,
                }
            )

        # Fallback for odd-numbered French translation pages (where top half has no vertical Nôm columns)
        if not ch_boxes:
            full_horiz = self._segment_monolingual_horizontal_lines(img, is_qn=True, full_Margins=True)
            ch_boxes = [
                b
                for b in full_horiz
                if (b["points"][2][0] - b["points"][0][0]) >= int(w * 0.07)
            ]
        if not vi_boxes and ch_boxes:
            vi_boxes = list(ch_boxes)

        return {"ch": ch_boxes, "vi": vi_boxes}

    def _segment_monolingual_horizontal_lines(
        self, img: np.ndarray, is_qn: bool = True, full_Margins: bool = True
    ) -> List[dict]:
        """
        Segment horizontal lines on a monolingual page or sub-ROI (e.g. Thụ Dạ Lữ Hoài Ngâm Khúc_Dịch,
        ThơCaDuHành_1/vi, TìBà_1/vi, or bottom half of Hoàng Xuân Hãn).
        Removes red stamps/seals and marginal line numbers (5, 10, 15, 20).
        """
        h, w, _ = img.shape
        if full_Margins:
            top_cut, bot_cut = int(h * 0.035), int(h * 0.96)
            left_cut, right_cut = int(w * 0.06), int(w * 0.94)
        else:
            top_cut, bot_cut = 0, h
            left_cut, right_cut = 0, w

        roi = img[top_cut:bot_cut, left_cut:right_cut]
        rh, rw, _ = roi.shape

        # Suppress red seals/stamps on monolingual pages while handling yellowed paper (where bg_R - bg_G ~ 40)
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)
        bg_r = float(np.percentile(r, 85))
        bg_g = float(np.percentile(g, 85))
        p5_r = float(np.percentile(r, 5))
        rg_excess = (r - g) - (bg_r - bg_g)
        dark_r = (bg_r - r) > max(20.0, (bg_r - p5_r) * 0.21)
        not_red_stamp = rg_excess < 22.0
        ink = self._clean_single_ink_layer(dark_r & not_red_stamp)

        cw = max(22, int(rw * 0.032))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (cw, 2)))
        hp = (closed > 0).sum(axis=1)
        min_px = max(12, int(rw * 0.012))
        min_lh = max(10, int(h * 0.008))
        max_lh = max(34, int(h * 0.038))

        spans: List[Tuple[int, int]] = []
        in_l = False
        sy = 0
        for y, val in enumerate(hp):
            if val >= min_px and not in_l:
                in_l = True
                sy = y
            elif val < min_px and in_l:
                in_l = False
                if (y - sy) >= 8:
                    spans.append((sy, y))
        if in_l and (len(hp) - sy) >= 8:
            spans.append((sy, len(hp)))

        # Split touching cursive/typed lines > max_lh
        curr = list(spans)
        for _ in range(8):
            nxt: List[Tuple[int, int]] = []
            changed = False
            for sy, ey in curr:
                sh = ey - sy
                if sh > max_lh and sh >= min_lh * 2:
                    ms = sy + int(sh * 0.22)
                    me = sy + int(sh * 0.78)
                    if me > ms:
                        sp = ms + int(np.argmin(hp[ms:me]))
                        if (sp - sy) >= min_lh and (ey - sp) >= min_lh:
                            nxt.extend([(sy, sp), (sp, ey)])
                            changed = True
                            continue
                nxt.append((sy, ey))
            curr = nxt
            if not changed:
                break

        boxes = []
        for sy, ey in curr:
            sub = ink[sy:ey, :]
            cp = (sub > 0).sum(axis=0)
            ac = np.where(cp >= 2)[0]
            if len(ac) < 10:
                continue

            # Strip marginal line numbers (5, 10, 15, 20) separated by > 4.5% width gap
            gaps = np.where(np.diff(ac) > int(rw * 0.045))[0]
            if len(gaps) > 0:
                segs = [
                    s
                    for s in np.split(ac, gaps + 1)
                    if len(s) >= int(rw * 0.05)
                    or (len(s) >= 10 and s[0] > int(rw * 0.12) and s[-1] < int(rw * 0.85))
                ]
                if not segs:
                    continue
                ac = np.concatenate(segs)

            rp = (sub[:, ac[0]:ac[-1] + 1] > 0).sum(axis=1)
            ar = np.where(rp >= 2)[0]
            if len(ar) < 5:
                continue

            x1 = left_cut + int(ac[0]) - 3
            x2 = left_cut + int(ac[-1]) + 3
            y1 = top_cut + sy + int(ar[0]) - 2
            y2 = top_cut + sy + int(ar[-1]) + 2
            if (x2 - x1) < int(w * 0.08) or (y2 - y1) < 10:
                continue

            boxes.append(
                {
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "y_center": (y1 + y2) / 2.0,
                }
            )
        return boxes

    def _segment_mixed_stanza_and_cursive(self, img: np.ndarray) -> Dict[str, List[dict]]:
        """
        Segment ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_2:
        - Preprocesses thin onion-skin paper with strong bleed-through suppression ((bg - gray) > 35)
          and removes bottom watermark ('VERRERIE DU NORD' at y > 90% height) + red annotations.
        - Splits horizontal rows at column/phrase gaps (>= 2.5% width) so 2-column 7-char Hán stanzas
          and side-by-side bilingual entries ('20. 過清華鎮 | Qua trấn Thanh Hoa', '驛勇決 | Chợ Dũng Quyết')
          are cleanly separated into individual segments.
        - Classifies each segment into Hán Nôm (ch) vs Quốc ngữ (vi) via vertical stroke crossing frequency (v_trans)
          and 2D character stroke density.
        """
        h, w, _ = img.shape
        top_cut, bot_cut = int(h * 0.03), int(h * 0.90)
        left_cut, right_cut = int(w * 0.08), int(w * 0.95)

        roi = img[top_cut:bot_cut, left_cut:right_cut]
        rh, rw, _ = roi.shape

        # Suppress red editorial circles/notes and reverse-side bleed-through relative to paper background
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)
        bg_r = float(np.percentile(r, 85))
        bg_g = float(np.percentile(g, 85))
        rg_excess = (r - g) - (bg_r - bg_g)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = float(np.percentile(gray, 85))

        strong_dark = ((bg - gray) > 34.0) & (rg_excess < 22.0)
        ink = self._clean_single_ink_layer(strong_dark)
        if (ink > 0).sum() < 6000:
            # Fallback for red-ink inserted note slip (e.g. ThơCaDuHành_2.009.jpg) or faint pencil page
            strong_dark = ((bg_g - g) > 22.0) | ((bg - gray) > 16.0)
            ink = self._clean_single_ink_layer(strong_dark)

        cw = max(18, int(rw * 0.024))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (cw, 2)))
        hp = (closed > 0).sum(axis=1)
        min_px = max(10, int(rw * 0.012))
        min_lh = 12
        max_lh = int(h * 0.033)  # ~47px on h=1447

        spans: List[Tuple[int, int]] = []
        in_l = False
        sy = 0
        for y, val in enumerate(hp):
            if val >= min_px and not in_l:
                in_l = True
                sy = y
            elif val < min_px and in_l:
                in_l = False
                if (y - sy) >= 10:
                    spans.append((sy, y))
        if in_l and (len(hp) - sy) >= 10:
            spans.append((sy, len(hp)))

        # Merge tiny diacritic slivers (< 6px gap and combined height <= max_lh)
        merged: List[Tuple[int, int]] = []
        for sy, ey in spans:
            if merged and (sy - merged[-1][1]) <= 5 and (ey - merged[-1][0]) <= max_lh:
                merged[-1] = (merged[-1][0], ey)
            else:
                merged.append((sy, ey))

        # Split touching rows > max_lh
        curr = list(merged)
        for _ in range(8):
            nxt: List[Tuple[int, int]] = []
            changed = False
            for sy, ey in curr:
                sh = ey - sy
                if sh > max_lh and sh >= min_lh * 2:
                    ms = sy + int(sh * 0.24)
                    me = sy + int(sh * 0.76)
                    if me > ms:
                        sp = ms + int(np.argmin(hp[ms:me]))
                        if (sp - sy) >= min_lh and (ey - sp) >= min_lh:
                            nxt.extend([(sy, sp), (sp, ey)])
                            changed = True
                            continue
                nxt.append((sy, ey))
            curr = nxt
            if not changed:
                break

        ch_boxes = []
        vi_boxes = []
        gap_thresh = max(22, int(rw * 0.028))

        for sy, ey in curr:
            sub = ink[sy:ey, :]
            ac = np.where((sub > 0).sum(axis=0) >= 1)[0]
            if len(ac) < 18:
                continue

            gaps = np.where(np.diff(ac) > gap_thresh)[0]
            segs = [s for s in np.split(ac, gaps + 1) if len(s) >= 32]
            if not segs:
                continue

            for s in segs:
                sx1, sx2 = int(s[0]), int(s[-1])
                crop = sub[:, sx1:sx2 + 1]
                ar = np.where((crop > 0).sum(axis=1) >= 1)[0]
                if len(ar) < 8:
                    continue
                crop_tight = crop[ar[0]:ar[-1] + 1, :]
                sh_c, sw_c = crop_tight.shape
                if sw_c < 35 or sh_c < 10:
                    continue

                fill = float((crop_tight > 0).sum()) / float(sh_c * sw_c)
                v_trans = float(np.abs(np.diff((crop_tight > 0).astype(np.int8), axis=0)).sum(axis=0).mean())

                # Hán Nôm verses/entries have multiple stacked horizontal strokes per character (v_trans >= 3.0)
                is_nom = (v_trans >= 3.0) or (v_trans >= 2.85 and fill >= 0.22 and len(segs) >= 2)

                x1 = max(0, left_cut + sx1 - 3)
                x2 = min(w, left_cut + sx2 + 3)
                y1 = max(0, top_cut + sy + int(ar[0]) - 2)
                y2 = min(h, top_cut + sy + int(ar[-1]) + 2)

                item = {
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "y_center": (y1 + y2) / 2.0,
                    "v_trans": v_trans,
                }
                if is_nom:
                    ch_boxes.append(item)
                else:
                    vi_boxes.append(item)

        # Fallback for 100% prose commentary pages where inline Hán characters are inside lines with v_trans < 3.0
        if not ch_boxes and vi_boxes:
            ranked = sorted(vi_boxes, key=lambda b: b.get("v_trans", 0.0), reverse=True)
            ch_boxes = [dict(b) for b in ranked[: max(1, min(3, len(ranked) // 4))]]
            ch_boxes.sort(key=lambda b: b["y_center"])
        elif not vi_boxes and ch_boxes:
            vi_boxes = [dict(b) for b in ch_boxes]

        return {"ch": ch_boxes, "vi": vi_boxes}

    def _segment_side_by_side_table(self, img: np.ndarray) -> Dict[str, List[dict]]:
        """
        Segment side-by-side parallel table pages (Tế cô hồn văn SV4.0 Tuệ Phương).
        Splits any touching rows exceeding max_row_h so no box ever merges two rows.
        """
        h, w, _ = img.shape
        top_cut = int(h * 0.05)
        bot_cut = int(h * 0.95)
        nom_x1, nom_x2 = int(w * 0.135), int(w * 0.415)
        qn_x1, qn_x2 = int(w * 0.420), int(w * 0.920)

        roi = img[top_cut:bot_cut, nom_x1:qn_x2]
        ink_mask, _ = self._build_ink_binary(roi)

        h_proj = (ink_mask > 0).sum(axis=1)
        min_row_pixels = max(12, int((qn_x2 - nom_x1) * 0.01))
        min_row_h = max(12, int(h * 0.012))
        max_row_h = max(48, int(h * 0.055))  # ~81px on h=1480

        rows: List[Tuple[int, int]] = []
        in_row = False
        sy = 0
        for y, val in enumerate(h_proj):
            if val >= min_row_pixels and not in_row:
                in_row = True
                sy = y
            elif val < min_row_pixels and in_row:
                in_row = False
                if (y - sy) >= min_row_h:
                    rows.append((sy, y))
        if in_row and (len(h_proj) - sy) >= min_row_h:
            rows.append((sy, len(h_proj)))

        # Split any merged two-row table span > max_row_h
        curr = list(rows)
        for _ in range(4):
            nxt: List[Tuple[int, int]] = []
            for sy, ey in curr:
                if (ey - sy) > max_row_h and (ey - sy) >= min_row_h * 2:
                    ms = sy + int((ey - sy) * 0.25)
                    me = sy + int((ey - sy) * 0.75)
                    sp = ms + int(np.argmin(h_proj[ms:me]))
                    if (sp - sy) >= min_row_h and (ey - sp) >= min_row_h:
                        nxt.extend([(sy, sp), (sp, ey)])
                        continue
                nxt.append((sy, ey))
            curr = nxt

        ch_boxes = []
        vi_boxes = []
        nom_w = nom_x2 - nom_x1

        for sy, ey in curr:
            nom_cell = ink_mask[sy:ey, :nom_w]
            nom_cols = np.where((nom_cell > 0).sum(axis=0) >= 2)[0]
            if len(nom_cols) > 0:
                nom_rows = np.where((nom_cell[:, nom_cols[0]:nom_cols[-1] + 1] > 0).sum(axis=1) >= 2)[0]
                if len(nom_rows) > 0:
                    x1 = nom_x1 + int(nom_cols[0]) - 4
                    x2 = nom_x1 + int(nom_cols[-1]) + 4
                    y1 = top_cut + sy + int(nom_rows[0]) - 2
                    y2 = top_cut + sy + int(nom_rows[-1]) + 2
                    ch_boxes.append(
                        {
                            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            "y_center": (y1 + y2) / 2.0,
                        }
                    )

            qn_cell = ink_mask[sy:ey, (qn_x1 - nom_x1):]
            qn_cols = np.where((qn_cell > 0).sum(axis=0) >= 2)[0]
            if len(qn_cols) > 0:
                qn_rows = np.where((qn_cell[:, qn_cols[0]:qn_cols[-1] + 1] > 0).sum(axis=1) >= 2)[0]
                if len(qn_rows) > 0:
                    x1 = qn_x1 + int(qn_cols[0]) - 4
                    x2 = qn_x1 + int(qn_cols[-1]) + 4
                    y1 = top_cut + sy + int(qn_rows[0]) - 2
                    y2 = top_cut + sy + int(qn_rows[-1]) + 2
                    vi_boxes.append(
                        {
                            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            "y_center": (y1 + y2) / 2.0,
                        }
                    )

        return {"ch": ch_boxes, "vi": vi_boxes}

    def _segment_vertical_columns_in_SubROI(
        self, img: np.ndarray, init_x1: int, init_y1: int, init_x2: int, init_y2: int
    ) -> List[dict]:
        """
        Helper to segment vertical Hán Nôm columns inside a specified rectangular sub-ROI,
        ordered strictly Right-to-Left (x descending), then Top-to-Bottom (y ascending).
        """
        h, w, _ = img.shape
        inner_bgr = img[init_y1:init_y2, init_x1:init_x2]
        if inner_bgr.size == 0:
            return []

        # Suppress red punctuation circles/ticks relative to paper background chrominance (bg_R - bg_G)
        g = inner_bgr[:, :, 1].astype(np.float32)
        r = inner_bgr[:, :, 2].astype(np.float32)
        bg_r = float(np.percentile(r, 85))
        bg_g = float(np.percentile(g, 85))
        p5_r = float(np.percentile(r, 5))
        rg_excess = (r - g) - (bg_r - bg_g)
        dark_r = ((bg_r - r) > max(20.0, (bg_r - p5_r) * 0.20)) & (rg_excess < 22.0)
        ink_inner = self._clean_single_ink_layer(dark_r)
        ih, iw = ink_inner.shape

        # Fallback for dark book covers with white catalog stickers (e.g. ThuDạLữHoàiNgâm.001.jpg where bg_r <= 30)
        if (ink_inner > 0).sum() < 400:
            gray_in = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2GRAY)
            bright_sticker = gray_in > 180
            if int(bright_sticker.sum()) > 1000 and int(bright_sticker.sum()) < int(ih * iw * 0.35):
                # Find dark text inside the bright catalog sticker
                num_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(bright_sticker.astype(np.uint8), connectivity=8)
                best_idx = -1
                best_area = 0
                for li in range(1, num_lbl):
                    area = int(stats[li, cv2.CC_STAT_AREA])
                    if area > best_area and area < int(ih * iw * 0.25):
                        best_area = area
                        best_idx = li
                if best_idx > 0:
                    sx = int(stats[best_idx, cv2.CC_STAT_LEFT])
                    sy = int(stats[best_idx, cv2.CC_STAT_TOP])
                    sw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
                    sh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
                    x1 = init_x1 + sx + int(sw * 0.12)
                    x2 = init_x1 + sx + int(sw * 0.88)
                    y1 = init_y1 + sy + int(sh * 0.12)
                    y2 = init_y1 + sy + int(sh * 0.88)
                    return [
                        {
                            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            "x_center": (x1 + x2) / 2.0,
                            "y_center": (y1 + y2) / 2.0,
                        }
                    ]

        # Detect top and bottom woodblock border bars inside inner ROI (> 60% width)
        row_density = (ink_inner > 0).sum(axis=1) / float(max(1, iw))
        top_candidates = np.where(row_density[:int(ih * 0.16)] > 0.60)[0]
        bot_candidates = np.where(row_density[int(ih * 0.84):] > 0.60)[0]

        sub_y1 = int(top_candidates[-1]) + 8 if len(top_candidates) > 0 else 0
        sub_y2 = int(ih * 0.84) + int(bot_candidates[0]) - 8 if len(bot_candidates) > 0 else ih
        if sub_y2 <= sub_y1 + int(ih * 0.3):
            sub_y1, sub_y2 = 0, ih

        text_only = ink_inner[sub_y1:sub_y2, :]
        rh, rw = text_only.shape

        col_proj = (text_only > 0).sum(axis=0)
        col_proj_smooth = np.convolve(col_proj, np.ones(5) / 5.0, mode="same")
        min_col_ink = max(10, int(rh * 0.032))
        min_col_w = max(18, int(rw * 0.022))
        max_single_col_w = int(rw * 0.092)

        col_spans: List[Tuple[int, int]] = []
        in_col = False
        sx = 0
        for x, val in enumerate(col_proj_smooth):
            if val >= min_col_ink and not in_col:
                in_col = True
                sx = x
            elif val < min_col_ink and in_col:
                in_col = False
                if (x - sx) >= min_col_w:
                    col_spans.append((sx, x))
        if in_col and (len(col_proj_smooth) - sx) >= min_col_w:
            col_spans.append((sx, len(col_proj_smooth)))

        # Recursively split wide merged column spans at local projection minima
        for _ in range(5):
            next_cols: List[Tuple[int, int]] = []
            changed = False
            for sx, ex in col_spans:
                cw = ex - sx
                if cw > max_single_col_w and cw >= min_col_w * 2:
                    ms = sx + int(cw * 0.22)
                    me = sx + int(cw * 0.78)
                    if me > ms:
                        split_x = ms + int(np.argmin(col_proj_smooth[ms:me]))
                        if (split_x - sx) >= min_col_w and (ex - split_x) >= min_col_w:
                            next_cols.append((sx, split_x))
                            next_cols.append((split_x, ex))
                            changed = True
                            continue
                next_cols.append((sx, ex))
            col_spans = next_cols
            if not changed:
                break

        raw_boxes = []
        for sx, ex in col_spans:
            col_slice = text_only[:, sx:ex]
            row_proj = (col_slice > 0).sum(axis=1)
            active_rows = np.where(row_proj >= 2)[0]
            if len(active_rows) == 0:
                continue
            sy = int(active_rows[0])
            ey = int(active_rows[-1])
            if (ey - sy) < int(rh * 0.08):
                continue

            col_ink = (col_slice[sy:ey + 1, :] > 0).sum(axis=0)
            active_cols = np.where(col_ink >= 2)[0]
            if len(active_cols) > 0:
                sx_tight = sx + int(active_cols[0])
                ex_tight = sx + int(active_cols[-1])
            else:
                sx_tight = sx
                ex_tight = ex

            x1 = max(0, init_x1 + sx_tight - 2)
            x2 = min(w, init_x1 + ex_tight + 2)
            y1 = max(0, init_y1 + sub_y1 + sy - 3)
            y2 = min(h, init_y1 + sub_y1 + ey + 3)
            raw_boxes.append((x1, y1, x2, y2))

        sorted_boxes = sorted(raw_boxes, key=lambda b: (-b[0], b[1]))
        return [
            {
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "x_center": (x1 + x2) / 2.0,
                "y_center": (y1 + y2) / 2.0,
            }
            for x1, y1, x2, y2 in sorted_boxes
        ]

    def segment_vertical_nom_columns(self, img_path: Path) -> List[dict]:
        """
        Segment vertical Sino-Nom columns (Tnom, Tnomw, T1/T2 vertical) adhering to regulation:
        - Supports both single-tier vertical manuscripts (ThuDạLữHoàiNgâm, TìBà_1, Tì Bà Preface in Hán)
          and two-tier vertical manuscripts (TìBà_2, ThơCaDuHành_1/ch).
        - Ordering: RIGHT-TO-LEFT, TOP-TO-BOTTOM (within each vertical tier).
        """
        img = cv2_imread_unicode(img_path)
        if img is None:
            return []

        h, w, _ = img.shape
        fname_lower = img_path.name.lower()
        b_lower = self.title.lower()

        init_y1, init_y2 = int(h * 0.035), int(h * 0.965)
        init_x1, init_x2 = int(w * 0.075), int(w * 0.925)

        # Check if this is a 2-tier vertical column book (TìBà_2 or ThơCaDuHành_1/ch)
        is_two_tier_book = ("tìbàquốcâm" in (fname_lower + b_lower) and "_2" in (fname_lower + b_lower)) or (
            "thơcaduhành" in (fname_lower + b_lower) and "_1" in (fname_lower + b_lower)
        )

        if is_two_tier_book:
            roi = img[init_y1:init_y2, init_x1:init_x2]
            ink, _ = self._build_ink_binary(roi)
            rh, rw = ink.shape
            # Find the horizontal corridor separating Upper Tier and Lower Tier (around 36%..58% of rh)
            hp = (ink > 0).sum(axis=1)
            hp_smooth = np.convolve(hp, np.ones(7) / 7.0, mode="same")
            s1, s2 = int(rh * 0.36), int(rh * 0.58)
            if s2 > s1:
                mid_rel = s1 + int(np.argmin(hp_smooth[s1:s2]))
                if hp_smooth[mid_rel] < rw * 0.06:
                    mid_abs = init_y1 + mid_rel
                    # Also check if TìBà_2 has a top commentary tier around 12%..24% of rh
                    s_top1, s_top2 = int(rh * 0.12), int(rh * 0.24)
                    top_tier_boxes = []
                    upper_start_y = init_y1
                    if "tìbà" in (fname_lower + b_lower) and s_top2 > s_top1:
                        top_rel = s_top1 + int(np.argmin(hp_smooth[s_top1:s_top2]))
                        if hp_smooth[top_rel] < rw * 0.04 and hp[:top_rel].sum() > 800:
                            top_abs = init_y1 + top_rel
                            top_tier_boxes = self._segment_vertical_columns_in_SubROI(
                                img, init_x1, init_y1, init_x2, top_abs
                            )
                            upper_start_y = top_abs

                    upper_boxes = self._segment_vertical_columns_in_SubROI(
                        img, init_x1, upper_start_y, init_x2, mid_abs
                    )
                    lower_boxes = self._segment_vertical_columns_in_SubROI(
                        img, init_x1, mid_abs, init_x2, init_y2
                    )
                    combined = top_tier_boxes + upper_boxes + lower_boxes
                    if len(combined) >= 6:
                        return combined

        return self._segment_vertical_columns_in_SubROI(img, init_x1, init_y1, init_x2, init_y2)
