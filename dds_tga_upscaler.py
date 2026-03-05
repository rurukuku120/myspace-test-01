#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDS → TGA 업스케일러
- DDS 파일을 TGA로 변환 (4K 텍스처 기준)
- Real-ESRGAN (Python) / ncnn-vulkan / Waifu2x / Bicubic 엔진 지원
- RGBA 채널 분리 기능
- 폴더 또는 개별 파일 처리
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import subprocess
import sys
import time
import queue
import io
import json
import uuid
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ──────────────────────────────────────────
# 상수
# ──────────────────────────────────────────
APP_TITLE   = "DDS → TGA 업스케일러"
APP_VERSION = "1.1.0"
DEFAULT_TARGET = 4096

ESRGAN_MODELS = {
    "RealESRGAN_x4plus  (범용)":           ("RealESRGAN_x4plus",           4),
    "RealESRGAN_x4plus_anime (애니)":      ("RealESRGAN_x4plus_anime_6B",  4),
    "RealESRNet_x4plus  (빠름)":           ("RealESRNet_x4plus",           4),
    "RealESRGAN_x2plus  (x2)":             ("RealESRGAN_x2plus",           2),
}

WAIFU2X_MODELS = {
    "models-cunet  (고품질)":                       "models-cunet",
    "models-upconv_7_anime  (애니)":                "models-upconv_7_anime_style_art_rgb",
    "models-upconv_7_photo  (사진)":                "models-upconv_7_photo",
}

REALESRGAN_NCNN_MODELS = {
    "realesr-animevideov3  (애니/영상)":   "realesr-animevideov3",
    "realesrgan-x4plus  (범용)":           "realesrgan-x4plus",
    "realesrgan-x4plus-anime  (애니)":     "realesrgan-x4plus-anime",
    "realesrnet-x4plus  (빠름)":           "realesrnet-x4plus",
}

COMFYUI_UPSCALE_MODELS = [
    "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "4x-UltraSharp.pth",
    "4x_NMKD-Siax_200k.pth",
    "8x_NMKD-Superscale.pth",
]

# ──────────────────────────────────────────
# 업스케일 엔진
# ──────────────────────────────────────────
class BicubicEngine:
    """PIL Lanczos 업스케일 (항상 사용 가능)"""
    name = "bicubic"

    def is_available(self):
        return True

    def upscale(self, img, scale, **kw):
        from PIL import Image
        w, h = img.size
        return img.resize((w * scale, h * scale), Image.LANCZOS)


class RealESRGANPythonEngine:
    """Real-ESRGAN Python 패키지 엔진"""
    name = "realesrgan_python"

    def is_available(self):
        try:
            import torch                                    # noqa
            from realesrgan import RealESRGANer            # noqa
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa
            return True
        except ImportError:
            return False

    def upscale(self, img, scale, model_name="RealESRGAN_x4plus",
                gpu_id=0, tile=0, **kw):
        import torch
        import numpy as np
        from PIL import Image
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        num_block = 6 if "anime_6B" in model_name else 23
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=num_block, num_grow_ch=32, scale=scale)

        model_url = (
            f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
            f"v0.1.0/{model_name}.pth"
        )
        device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0
            else "cpu"
        )
        upsampler = RealESRGANer(
            scale=scale, model_path=model_url, model=model,
            tile=tile, tile_pad=10, pre_pad=0,
            half=device.type == "cuda", device=device,
        )
        rgb = np.array(img.convert("RGB"))
        out, _ = upsampler.enhance(rgb, outscale=scale)
        result = Image.fromarray(out)

        # 알파 채널 복원
        if img.mode == "RGBA":
            alpha = img.split()[3].resize(result.size, Image.LANCZOS)
            result = result.convert("RGBA")
            result.putalpha(alpha)
        return result


class NCNNEngine:
    """realesrgan-ncnn-vulkan / waifu2x-ncnn-vulkan 외부 실행파일 엔진"""
    name = "ncnn"

    def __init__(self, exe_path=""):
        self.exe_path = exe_path

    def is_available(self):
        if self.exe_path and os.path.isfile(self.exe_path):
            return True
        for name in ["realesrgan-ncnn-vulkan", "waifu2x-ncnn-vulkan"]:
            r = subprocess.run(
                ["where", name], capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            if r.returncode == 0:
                return True
        return False

    def upscale(self, img, scale, exe_path="", model="",
                gpu_id=0, denoise=2, tile=0, **kw):
        import tempfile
        from PIL import Image

        exe = exe_path or self.exe_path
        if not exe:
            raise RuntimeError("ncnn 실행파일 경로를 지정하세요.")

        has_alpha = img.mode in ("RGBA", "LA")

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "input.png")
            out_path = os.path.join(tmpdir, "output.png")

            rgb = img.convert("RGB")
            if has_alpha:
                alpha_ch = img.split()[-1]
            rgb.save(in_path)

            cmd = [exe, "-i", in_path, "-o", out_path,
                   "-s", str(scale), "-g", str(gpu_id)]
            if model:
                cmd += ["-n", model]
            if tile > 0:
                cmd += ["-t", str(tile)]
            if denoise >= 0:
                cmd += ["-d", str(denoise)]

            r = subprocess.run(
                cmd, capture_output=True, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            if r.returncode != 0:
                raise RuntimeError(f"ncnn 실패:\n{r.stderr.strip()}")

            result = Image.open(out_path).copy()

        if has_alpha:
            alpha_up = alpha_ch.resize(result.size, Image.LANCZOS)
            result = result.convert("RGBA")
            result.putalpha(alpha_up)
        return result


class ComfyUIEngine:
    """ComfyUI REST API 업스케일 엔진 (로컬 서버 http://host:port)"""
    name = "comfyui"

    def is_available(self):
        return True  # 런타임에 연결 시도

    def upscale(self, img, scale,
                comfyui_host="127.0.0.1", comfyui_port=8188,
                comfyui_model="RealESRGAN_x4plus.pth", **kw):
        from PIL import Image

        base_url = f"http://{comfyui_host}:{comfyui_port}"

        # ── 1. 이미지 업로드 ──────────────────────
        has_alpha = img.mode in ("RGBA", "LA")
        alpha_ch  = img.split()[-1] if has_alpha else None
        rgb_img   = img.convert("RGB")

        buf = io.BytesIO()
        rgb_img.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()

        upload_filename = f"dds_upscaler_{uuid.uuid4().hex}.png"
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{upload_filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{base_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                upload_result = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(f"ComfyUI 이미지 업로드 실패: {e}\n서버가 실행 중인지 확인하세요.")

        uploaded_name = upload_result.get("name", upload_filename)

        # ── 2. 워크플로우(프롬프트) 전송 ──────────
        client_id = uuid.uuid4().hex
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_name},
            },
            "2": {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": comfyui_model},
            },
            "3": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["2", 0],
                    "image":         ["1", 0],
                },
            },
            "4": {
                "class_type": "SaveImage",
                "inputs": {
                    "images":        ["3", 0],
                    "filename_prefix": f"dds_up_{uuid.uuid4().hex[:8]}",
                },
            },
        }
        payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        req2 = urllib.request.Request(
            f"{base_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req2, timeout=30) as resp:
                prompt_result = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(f"ComfyUI 프롬프트 전송 실패: {e}")

        prompt_id = prompt_result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI prompt_id 없음: {prompt_result}")

        # ── 3. 완료 폴링 (최대 300초, 0.2초 간격) ──
        out_info = None
        for _ in range(1500):
            time.sleep(0.2)
            try:
                with urllib.request.urlopen(
                    f"{base_url}/history/{prompt_id}", timeout=10
                ) as resp:
                    history = json.loads(resp.read())
            except Exception:
                continue

            if prompt_id not in history:
                continue

            outputs = history[prompt_id].get("outputs", {})
            for node_out in outputs.values():
                imgs = node_out.get("images", [])
                if imgs:
                    out_info = imgs[0]
                    break
            if out_info is not None:
                break
        else:
            raise RuntimeError("ComfyUI 처리 타임아웃 (300초 초과)")

        # ── 4. 결과 이미지 다운로드 ──────────────
        params = urllib.parse.urlencode({
            "filename": out_info["filename"],
            "subfolder": out_info.get("subfolder", ""),
            "type": out_info.get("type", "output"),
        })
        try:
            with urllib.request.urlopen(
                f"{base_url}/view?{params}", timeout=30
            ) as resp:
                result_bytes = resp.read()
        except Exception as e:
            raise RuntimeError(f"ComfyUI 결과 다운로드 실패: {e}")

        result = Image.open(io.BytesIO(result_bytes)).convert("RGB")

        # ── 알파 채널 복원 ────────────────────────
        if has_alpha:
            alpha_up = alpha_ch.resize(result.size, Image.LANCZOS)
            result = result.convert("RGBA")
            result.putalpha(alpha_up)

        return result


# ──────────────────────────────────────────
# DDS 변환 유틸
# ──────────────────────────────────────────
def read_dds(path: str):
    from PIL import Image
    try:
        img = Image.open(path)
        img.load()  # 파일 핸들 즉시 해제 (lazy load 방지)
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA")
        return img
    except Exception as e:
        raise RuntimeError(f"DDS 읽기 실패: {e}")


def write_tga(img, path: str):
    img.save(path, format="TGA")


def split_rgba_channels(img, channels: list, tga_path: str) -> list:
    import numpy as np
    from PIL import Image

    arr = np.array(img.convert("RGBA"))
    ch_map = {"R": 0, "G": 1, "B": 2, "A": 3}
    base = Path(tga_path)

    def _save_channel(ch):
        idx = ch_map.get(ch.upper())
        if idx is None:
            return None
        out = base.parent / f"{base.stem}_{ch.upper()}.tga"
        Image.fromarray(arr[:, :, idx], mode="L").save(str(out), format="TGA")
        return str(out)

    with ThreadPoolExecutor(max_workers=min(len(channels), 4)) as ex:
        results = list(ex.map(_save_channel, channels))
    return [r for r in results if r is not None]


# ──────────────────────────────────────────
# 처리 워커 (별도 스레드)
# ──────────────────────────────────────────
class ProcessWorker(threading.Thread):

    def __init__(self, files: list, settings: dict, q: queue.Queue):
        super().__init__(daemon=True)
        self.files    = files
        self.settings = settings
        self.q        = q
        self._stop    = threading.Event()

    def stop(self):
        self._stop.set()

    def _log(self, msg, level="INFO"):
        self.q.put({"type": "log", "msg": msg, "level": level})

    def _prog(self, cur, tot, fname=""):
        self.q.put({"type": "progress", "cur": cur, "tot": tot, "fname": fname})

    def run(self):
        s = self.settings
        total = len(self.files)

        # 엔진 선택
        ename = s.get("engine", "bicubic")
        if ename == "realesrgan_python":
            engine = RealESRGANPythonEngine()
        elif ename in ("realesrgan_ncnn", "waifu2x_ncnn"):
            engine = NCNNEngine(s.get("ncnn_exe", ""))
        elif ename == "comfyui":
            engine = ComfyUIEngine()
        else:
            engine = BicubicEngine()

        if not engine.is_available():
            self._log(f"엔진 '{ename}' 를 사용할 수 없습니다. Bicubic으로 대체합니다.", "WARN")
            engine = BicubicEngine()

        scale  = s.get("scale", 4)
        target = s.get("target_size", DEFAULT_TARGET)

        # 다음 파일 미리 읽기용 executor
        _prefetch_ex = ThreadPoolExecutor(max_workers=1)
        _prefetch_future = None

        def _prefetch(path):
            try:
                return read_dds(path)
            except Exception:
                return None

        # 첫 번째 파일 미리 읽기 시작
        if self.files:
            _prefetch_future = _prefetch_ex.submit(_prefetch, self.files[0])

        for i, dds_path in enumerate(self.files):
            if self._stop.is_set():
                self._log("⛔ 처리 중단됨", "WARN")
                break

            fname = os.path.basename(dds_path)
            self._prog(i, total, fname)
            self._log(f"▶ {fname}")

            try:
                # 출력 경로
                out_dir = s.get("output_dir", "")
                if s.get("keep_structure") and s.get("input_base"):
                    rel = os.path.relpath(os.path.dirname(dds_path), s["input_base"])
                    if rel != ".":
                        out_dir = os.path.join(out_dir, rel)
                os.makedirs(out_dir, exist_ok=True)

                stem     = Path(dds_path).stem
                tga_path = os.path.join(out_dir, f"{stem}.tga")

                if os.path.exists(tga_path) and not s.get("overwrite", True):
                    self._log(f"  건너뜀 (이미 존재): {stem}.tga", "WARN")
                    # 건너뛰어도 다음 파일 prefetch
                    if i + 1 < len(self.files):
                        _prefetch_future = _prefetch_ex.submit(_prefetch, self.files[i + 1])
                    continue

                # DDS 읽기 (prefetch 결과 우선 사용)
                if _prefetch_future is not None:
                    img = _prefetch_future.result()
                    _prefetch_future = None
                    if img is None:
                        img = read_dds(dds_path)
                else:
                    img = read_dds(dds_path)

                # 다음 파일 미리 읽기 시작 (현재 업스케일 중에 병렬 실행)
                if i + 1 < len(self.files):
                    _prefetch_future = _prefetch_ex.submit(_prefetch, self.files[i + 1])
                self._log(f"  읽기 완료: {img.size[0]}x{img.size[1]} {img.mode}")

                # 업스케일 필요 여부 판단
                w, h = img.size
                if max(w, h) >= target:
                    # 이미 4K 이상 → 리사이즈만
                    from PIL import Image
                    img_up = img.resize((target, target), Image.LANCZOS)
                    self._log(f"  리사이즈(이미 큼): {img_up.size[0]}x{img_up.size[1]}")
                else:
                    img_up = engine.upscale(
                        img, scale,
                        model_name     = s.get("model_name", "RealESRGAN_x4plus"),
                        gpu_id         = s.get("gpu_id", 0),
                        tile           = s.get("tile_size", 0),
                        denoise        = s.get("denoise", 2),
                        exe_path       = s.get("ncnn_exe", ""),
                        model          = s.get("ncnn_model", ""),
                        comfyui_host   = s.get("comfyui_host", "127.0.0.1"),
                        comfyui_port   = s.get("comfyui_port", 8188),
                        comfyui_model  = s.get("comfyui_model", "RealESRGAN_x4plus.pth"),
                    )
                    self._log(f"  업스케일 완료: {img_up.size[0]}x{img_up.size[1]}")

                    # 목표 크기 초과 시 크롭/리사이즈
                    if max(img_up.size) > target:
                        from PIL import Image
                        img_up = img_up.resize((target, target), Image.LANCZOS)

                # TGA 저장
                write_tga(img_up, tga_path)
                self._log(f"  저장: {os.path.basename(tga_path)}", "OK")

                # RGBA 채널 분리
                if s.get("split_channels") and s.get("channels"):
                    saved = split_rgba_channels(img_up, s["channels"], tga_path)
                    for sp in saved:
                        self._log(f"  채널: {os.path.basename(sp)}", "OK")

            except Exception as e:
                self._log(f"  오류: {e}", "ERROR")

        _prefetch_ex.shutdown(wait=False)
        self._prog(total, total)
        self.q.put({"type": "done"})


# ──────────────────────────────────────────
# GUI 메인 클래스
# ──────────────────────────────────────────
class App(tk.Tk):

    # ── 색상 팔레트 (Catppuccin Mocha) ──
    BG      = "#1e1e2e"
    BG2     = "#313244"
    BG3     = "#45475a"
    FG      = "#cdd6f4"
    ACCENT  = "#89b4fa"
    GREEN   = "#a6e3a1"
    RED     = "#f38ba8"
    YELLOW  = "#fab387"
    SUBTEXT = "#6c7086"

    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE}  v{APP_VERSION}")
        self.geometry("860x820")
        self.minsize(740, 660)
        self.configure(bg=self.BG)

        self.input_files: list[str] = []
        self.worker: ProcessWorker | None = None
        self.q: queue.Queue = queue.Queue()

        self._init_vars()
        self._apply_style()
        self._build_ui()
        self.after(50, self._poll_queue)
        self.after(200, self._check_deps_async)

    # ── 변수 초기화 ──────────────────────
    def _init_vars(self):
        self.var_engine         = tk.StringVar(value="realesrgan_python")
        self.var_ncnn_exe       = tk.StringVar()
        self.var_esrgan_model   = tk.StringVar(value=list(ESRGAN_MODELS.keys())[0])
        self.var_ncnn_re_model  = tk.StringVar(value=list(REALESRGAN_NCNN_MODELS.keys())[0])
        self.var_w2x_model      = tk.StringVar(value=list(WAIFU2X_MODELS.keys())[0])
        self.var_scale          = tk.StringVar(value="4x")
        self.var_target         = tk.StringVar(value=str(DEFAULT_TARGET))
        self.var_gpu_id         = tk.IntVar(value=0)
        self.var_tile           = tk.IntVar(value=0)
        self.var_denoise        = tk.IntVar(value=2)
        self.var_output_dir     = tk.StringVar()
        self.var_keep_structure = tk.BooleanVar(value=True)
        self.var_overwrite      = tk.BooleanVar(value=True)
        self.var_recursive      = tk.BooleanVar(value=True)
        self.var_split          = tk.BooleanVar(value=False)
        self.var_ch_r           = tk.BooleanVar(value=True)
        self.var_ch_g           = tk.BooleanVar(value=True)
        self.var_ch_b           = tk.BooleanVar(value=True)
        self.var_ch_a           = tk.BooleanVar(value=True)
        self.var_comfyui_host   = tk.StringVar(value="127.0.0.1")
        self.var_comfyui_port   = tk.StringVar(value="8188")
        self.var_comfyui_model  = tk.StringVar(value=COMFYUI_UPSCALE_MODELS[0])

    # ── ttk 스타일 ────────────────────────
    def _apply_style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        B, B2, B3, F, A = self.BG, self.BG2, self.BG3, self.FG, self.ACCENT
        st.configure(".",            background=B,  foreground=F,  font=("Segoe UI", 9))
        st.configure("TFrame",       background=B)
        st.configure("Card.TFrame",  background=B2)
        st.configure("TLabel",       background=B,  foreground=F)
        st.configure("Card.TLabel",  background=B2, foreground=F)
        st.configure("Sub.TLabel",   background=B2, foreground=self.SUBTEXT,
                     font=("Segoe UI", 8))
        st.configure("Head.TLabel",  background=B2, foreground=A,
                     font=("Segoe UI", 10, "bold"))
        st.configure("TButton",      background=B3, foreground=F, relief="flat", padding=6)
        st.configure("Start.TButton",background=A,  foreground=B,
                     font=("Segoe UI", 10, "bold"), padding=9)
        st.configure("TCheckbutton", background=B2, foreground=F)
        st.configure("TRadiobutton", background=B2, foreground=F)
        st.configure("TCombobox",    fieldbackground=B2, background=B2, foreground=F)
        st.configure("TSpinbox",     fieldbackground=B2, background=B2, foreground=F)
        st.configure("TEntry",       fieldbackground=B2, foreground=F)
        st.configure("TProgressbar", background=A,  troughcolor=B3)
        st.configure("TNotebook",    background=B)
        st.configure("TNotebook.Tab",background=B2, foreground=F, padding=[12, 5])
        st.map("TNotebook.Tab",
               background=[("selected", B3)],
               foreground=[("selected", A)])
        st.map("TButton",
               background=[("active", B3), ("disabled", B2)],
               foreground=[("disabled", self.SUBTEXT)])
        st.map("TCheckbutton", background=[("active", B2)])
        st.map("TRadiobutton", background=[("active", B2)])

        # 커스텀 체크마크(✓) 인디케이터 적용
        self._img_chk_on, self._img_chk_off = self._create_check_images()
        st.element_create(
            "Custom.Checkbutton.indicator", "image",
            self._img_chk_on,
            ("!selected", self._img_chk_off),
        )
        st.layout("TCheckbutton", [
            ("Checkbutton.padding", {"sticky": "nswe", "children": [
                ("Custom.Checkbutton.indicator", {"side": "left", "sticky": ""}),
                ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                    ("Checkbutton.label", {"sticky": "nswe"})
                ]})
            ]})
        ])

    def _create_check_images(self):
        """체크됨(✓) / 미체크 인디케이터 이미지 생성"""
        sz = 14

        # ── 체크됨: ACCENT 배경 + 흰색 ✓ ──
        ch = tk.PhotoImage(width=sz, height=sz)
        ch.put(self.ACCENT, to=(0, 0, sz, sz))
        # ✓ 획: 왼쪽 아래 방향 + 오른쪽 위 방향 (2px 두께)
        for x, y in [(2,8),(3,9),(4,10),(5,9),(6,8),(7,7),(8,6),(9,5),(10,4)]:
            if 0 <= x < sz and 0 <= y < sz:
                ch.put("#ffffff", to=(x, y,   x+1, y+1))
            if 0 <= x < sz and 0 <= y-1 < sz:
                ch.put("#ffffff", to=(x, y-1, x+1, y))

        # ── 미체크: 어두운 배경 + 테두리 ──
        uc = tk.PhotoImage(width=sz, height=sz)
        uc.put(self.BG2, to=(0, 0, sz, sz))
        uc.put(self.BG3, to=(0,    0,    sz,    1))     # top
        uc.put(self.BG3, to=(0,    sz-1, sz,    sz))    # bottom
        uc.put(self.BG3, to=(0,    0,    1,     sz))    # left
        uc.put(self.BG3, to=(sz-1, 0,    sz,    sz))    # right

        return ch, uc

    # ── UI 빌드 ───────────────────────────
    def _build_ui(self):
        # 타이틀 바
        tb = tk.Frame(self, bg="#11111b", height=48)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        tk.Label(tb, text=f"  {APP_TITLE}", bg="#11111b", fg=self.ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(side="left", pady=10)
        tk.Label(tb, text=f"v{APP_VERSION}  ", bg="#11111b", fg=self.SUBTEXT,
                 font=("Segoe UI", 8)).pack(side="right", pady=14)

        # 노트북 탭
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        t_main    = ttk.Frame(nb)
        t_engine  = ttk.Frame(nb)
        t_rgba    = ttk.Frame(nb)
        nb.add(t_main,   text="  파일 & 출력  ")
        nb.add(t_engine, text="  업스케일 엔진  ")
        nb.add(t_rgba,   text="  RGBA 채널  ")

        self._tab_main(t_main)
        self._tab_engine(t_engine)
        self._tab_rgba(t_rgba)

        # 하단 패널
        self._build_bottom()

    # ── 카드 헬퍼 ─────────────────────────
    def _card(self, parent, title: str) -> ttk.Frame:
        outer = ttk.Frame(parent, style="Card.TFrame")
        outer.pack(fill="x", padx=8, pady=5)
        ttk.Label(outer, text=title, style="Head.TLabel").pack(anchor="w", padx=10, pady=(8, 3))
        inner = ttk.Frame(outer, style="Card.TFrame")
        inner.pack(fill="x", padx=10, pady=(0, 10))
        return inner

    def _row(self, parent, label="", width=14) -> tuple:
        """라벨 + 콘텐츠 행 반환"""
        r = ttk.Frame(parent, style="Card.TFrame")
        r.pack(fill="x", pady=3)
        if label:
            ttk.Label(r, text=label, style="Card.TLabel", width=width).pack(side="left")
        return r

    # ── 탭 1: 파일 & 출력 ─────────────────
    def _tab_main(self, parent):
        # 입력 카드
        c = self._card(parent, "📂  입력 파일")

        lf = ttk.Frame(c, style="Card.TFrame")
        lf.pack(fill="x")
        self.listbox = tk.Listbox(
            lf, height=7, bg="#181825", fg=self.FG,
            selectbackground=self.BG3, selectforeground=self.FG,
            borderwidth=0, highlightthickness=1, highlightbackground=self.BG3,
            font=("Consolas", 8), activestyle="none"
        )
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        bf = ttk.Frame(c, style="Card.TFrame")
        bf.pack(fill="x", pady=(6, 2))
        ttk.Button(bf, text="파일 추가",   command=self._add_files).pack(side="left", padx=(0, 4))
        ttk.Button(bf, text="폴더 추가",   command=self._add_folder).pack(side="left", padx=(0, 4))
        ttk.Button(bf, text="선택 삭제",   command=self._remove_sel).pack(side="left", padx=(0, 4))
        ttk.Button(bf, text="전체 초기화", command=self._clear_files).pack(side="left")
        ttk.Checkbutton(bf, text="하위 폴더 포함", variable=self.var_recursive,
                        style="TCheckbutton").pack(side="right")

        self.lbl_count = ttk.Label(c, text="파일 0개 선택됨", style="Sub.TLabel")
        self.lbl_count.pack(anchor="w", pady=(2, 0))

        # 출력 카드
        c2 = self._card(parent, "💾  출력 설정")

        r1 = self._row(c2, "출력 폴더:")
        ttk.Entry(r1, textvariable=self.var_output_dir).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r1, text="찾아보기", command=self._browse_output).pack(side="right")

        r2 = self._row(c2)
        ttk.Checkbutton(r2, text="폴더 구조 유지", variable=self.var_keep_structure,
                        style="TCheckbutton").pack(side="left", padx=(0, 20))
        ttk.Checkbutton(r2, text="기존 파일 덮어쓰기", variable=self.var_overwrite,
                        style="TCheckbutton").pack(side="left")

        r3 = self._row(c2, "목표 해상도:")
        ttk.Entry(r3, textvariable=self.var_target, width=8).pack(side="left", padx=4)
        ttk.Label(r3, text="× px  (텍스처 4K = 4096)", style="Card.TLabel").pack(side="left")

    # ── 탭 2: 업스케일 엔진 ──────────────
    def _tab_engine(self, parent):
        # 엔진 선택
        c = self._card(parent, "🚀  엔진 선택")

        engines = [
            ("Real-ESRGAN  Python 패키지  (pip install realesrgan basicsr torch)",
             "realesrgan_python"),
            ("Real-ESRGAN  ncnn-vulkan  (외부 .exe, AMD/NVIDIA GPU)",
             "realesrgan_ncnn"),
            ("Waifu2x  ncnn-vulkan  (외부 .exe, 애니/만화 전용)",
             "waifu2x_ncnn"),
            ("ComfyUI  REST API  (로컬 ComfyUI 서버, http://127.0.0.1:8188)",
             "comfyui"),
            ("Bicubic Lanczos  (항상 사용 가능, 품질 낮음)",
             "bicubic"),
        ]
        for label, val in engines:
            ttk.Radiobutton(c, text=label, variable=self.var_engine, value=val,
                            style="TRadiobutton",
                            command=self._refresh_engine_ui).pack(anchor="w", pady=2)

        # ncnn 실행파일 경로
        self.f_ncnn_exe = ttk.Frame(c, style="Card.TFrame")
        self.f_ncnn_exe.pack(fill="x", pady=(8, 0))
        ttk.Label(self.f_ncnn_exe, text="실행파일 경로:", style="Card.TLabel",
                  width=14).pack(side="left")
        ttk.Entry(self.f_ncnn_exe, textvariable=self.var_ncnn_exe).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(self.f_ncnn_exe, text="찾기",
                   command=self._browse_ncnn).pack(side="right")

        # ComfyUI 서버 설정
        self.f_comfyui = ttk.Frame(c, style="Card.TFrame")
        r_host = self._row(self.f_comfyui, "서버 주소:", 12)
        ttk.Entry(r_host, textvariable=self.var_comfyui_host,
                  width=20).pack(side="left", padx=4)
        ttk.Label(r_host, text="포트:", style="Card.TLabel").pack(side="left", padx=(8, 0))
        ttk.Entry(r_host, textvariable=self.var_comfyui_port,
                  width=7).pack(side="left", padx=4)
        r_model = self._row(self.f_comfyui, "업스케일 모델:", 12)
        ttk.Combobox(r_model, textvariable=self.var_comfyui_model,
                     values=COMFYUI_UPSCALE_MODELS, width=36).pack(side="left", padx=4)
        ttk.Label(r_model, text="(models/upscale_models/ 에 배치)",
                  style="Sub.TLabel").pack(side="left")

        # 모델 선택
        c2 = self._card(parent, "🎯  모델")

        # Python ESRGAN 모델
        self.f_esrgan_model = ttk.Frame(c2, style="Card.TFrame")
        self.f_esrgan_model.pack(fill="x", pady=2)
        ttk.Label(self.f_esrgan_model, text="모델:", style="Card.TLabel",
                  width=8).pack(side="left")
        ttk.Combobox(self.f_esrgan_model, textvariable=self.var_esrgan_model,
                     values=list(ESRGAN_MODELS.keys()), state="readonly",
                     width=46).pack(side="left", padx=4)

        # ESRGAN ncnn 모델
        self.f_re_ncnn_model = ttk.Frame(c2, style="Card.TFrame")
        ttk.Label(self.f_re_ncnn_model, text="모델:", style="Card.TLabel",
                  width=8).pack(side="left")
        ttk.Combobox(self.f_re_ncnn_model, textvariable=self.var_ncnn_re_model,
                     values=list(REALESRGAN_NCNN_MODELS.keys()), state="readonly",
                     width=46).pack(side="left", padx=4)

        # Waifu2x 모델
        self.f_w2x_model = ttk.Frame(c2, style="Card.TFrame")
        ttk.Label(self.f_w2x_model, text="모델:", style="Card.TLabel",
                  width=8).pack(side="left")
        ttk.Combobox(self.f_w2x_model, textvariable=self.var_w2x_model,
                     values=list(WAIFU2X_MODELS.keys()), state="readonly",
                     width=46).pack(side="left", padx=4)

        # 세부 설정
        c3 = self._card(parent, "🔧  세부 설정")

        r1 = self._row(c3, "업스케일 배율:", 14)
        ttk.Combobox(r1, textvariable=self.var_scale,
                     values=["2x", "4x"], state="readonly", width=7).pack(side="left", padx=4)

        r2 = self._row(c3, "GPU ID:", 14)
        ttk.Spinbox(r2, from_=-1, to=7, textvariable=self.var_gpu_id,
                    width=5).pack(side="left", padx=4)
        ttk.Label(r2, text="(-1 = CPU 강제)", style="Card.TLabel").pack(side="left")

        r3 = self._row(c3, "타일 크기:", 14)
        ttk.Spinbox(r3, from_=0, to=2048, increment=64, textvariable=self.var_tile,
                    width=7).pack(side="left", padx=4)
        ttk.Label(r3, text="(0=자동  |  VRAM 부족 시 256~512)", style="Card.TLabel").pack(side="left")

        self.f_denoise = self._row(c3, "노이즈 제거:", 14)
        self.lbl_dn = ttk.Label(self.f_denoise, text=str(self.var_denoise.get()),
                                style="Card.TLabel", width=3)
        ttk.Scale(self.f_denoise, from_=-1, to=3, orient="horizontal",
                  variable=self.var_denoise, length=130,
                  command=lambda v: self.lbl_dn.configure(
                      text=str(int(float(v))))).pack(side="left", padx=4)
        self.lbl_dn.pack(side="left")
        ttk.Label(self.f_denoise, text="(-1=없음, Waifu2x 전용)",
                  style="Sub.TLabel").pack(side="left", padx=8)

        self._refresh_engine_ui()

    # ── 탭 3: RGBA 채널 ───────────────────
    def _tab_rgba(self, parent):
        c = self._card(parent, "🎨  RGBA 채널 분리")

        ttk.Checkbutton(c, text="채널 분리 활성화 (원본 TGA와 별도로 각 채널을 그레이스케일 TGA로 저장)",
                        variable=self.var_split, style="TCheckbutton",
                        command=self._refresh_rgba_ui).pack(anchor="w", pady=(0, 10))

        self.f_rgba_opts = ttk.Frame(c, style="Card.TFrame")
        self.f_rgba_opts.pack(fill="x")

        ttk.Label(self.f_rgba_opts, text="추출할 채널:", style="Card.TLabel").pack(anchor="w")
        ch_row = ttk.Frame(self.f_rgba_opts, style="Card.TFrame")
        ch_row.pack(anchor="w", pady=5)

        self.ch_checks = []
        for text, var in [("R  (Red)",  self.var_ch_r), ("G  (Green)", self.var_ch_g),
                          ("B  (Blue)", self.var_ch_b), ("A  (Alpha)", self.var_ch_a)]:
            cb = ttk.Checkbutton(ch_row, text=text, variable=var, style="TCheckbutton")
            cb.pack(side="left", padx=12)
            self.ch_checks.append(cb)

        # 설명
        c2 = self._card(parent, "ℹ  출력 예시")
        info = (
            "파일명이 'diffuse.dds' 인 경우:\n\n"
            "  diffuse.tga          ← 원본 RGBA 전체\n"
            "  diffuse_R.tga        ← R 채널 (그레이스케일)\n"
            "  diffuse_G.tga        ← G 채널 (그레이스케일)\n"
            "  diffuse_B.tga        ← B 채널 (그레이스케일)\n"
            "  diffuse_A.tga        ← A 채널 (투명도)\n\n"
            "활용 예:\n"
            "  PBR 텍스처의 Roughness(R) / Metallic(G) / AO(B) / Opacity(A) 분리\n"
            "  ORM 패킹 텍스처 분해"
        )
        tk.Text(c2, height=11, bg="#181825", fg=self.SUBTEXT,
                font=("Consolas", 8), relief="flat",
                borderwidth=0, wrap="word", state="normal").pack(fill="x")
        for w in c2.winfo_children():
            if isinstance(w, tk.Text):
                w.insert("1.0", info)
                w.configure(state="disabled")

        self._refresh_rgba_ui()

    # ── 하단 진행 패널 ────────────────────
    def _build_bottom(self):
        bot = ttk.Frame(self, style="Card.TFrame")
        bot.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(bot, text="로그", style="Head.TLabel").pack(anchor="w", padx=10, pady=(6, 3))

        lf = ttk.Frame(bot, style="Card.TFrame")
        lf.pack(fill="x", padx=10)
        self.log_text = tk.Text(
            lf, height=8, bg="#11111b", fg=self.FG,
            font=("Consolas", 8), relief="flat",
            borderwidth=0, wrap="none", state="disabled"
        )
        lsb = ttk.Scrollbar(lf, orient="vertical", command=self.log_text.yview)
        lsb_x = ttk.Scrollbar(lf, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=lsb.set, xscrollcommand=lsb_x.set)
        lsb.pack(side="right", fill="y")
        self.log_text.pack(side="top", fill="x", expand=False)
        lsb_x.pack(side="bottom", fill="x")

        self.log_text.tag_configure("OK",    foreground=self.GREEN)
        self.log_text.tag_configure("ERROR", foreground=self.RED)
        self.log_text.tag_configure("WARN",  foreground=self.YELLOW)
        self.log_text.tag_configure("INFO",  foreground=self.FG)

        # 진행 바
        ctrl = tk.Frame(self, bg=self.BG)
        ctrl.pack(fill="x", padx=8, pady=6)

        self.lbl_status = ttk.Label(ctrl, text="대기 중…", style="TLabel")
        self.lbl_status.pack(anchor="w")
        self.prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(ctrl, variable=self.prog_var, maximum=100).pack(fill="x", pady=3)

        btn_row = tk.Frame(ctrl, bg=self.BG)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="로그 지우기", command=self._clear_log).pack(side="left")
        self.btn_stop  = ttk.Button(btn_row, text="⬛  중단",  command=self._stop,
                                    state="disabled")
        self.btn_stop.pack(side="right", padx=(4, 0))
        self.btn_start = ttk.Button(btn_row, text="▶  변환 시작",
                                    style="Start.TButton", command=self._start)
        self.btn_start.pack(side="right")

    # ── UI 상태 갱신 ──────────────────────
    def _refresh_engine_ui(self):
        engine = self.var_engine.get()
        is_ncnn_re  = engine == "realesrgan_ncnn"
        is_ncnn_w2  = engine == "waifu2x_ncnn"
        is_python   = engine == "realesrgan_python"
        is_ncnn     = is_ncnn_re or is_ncnn_w2
        is_comfyui  = engine == "comfyui"

        # ncnn 실행파일 행
        if is_ncnn:
            self.f_ncnn_exe.pack(fill="x", pady=(8, 0))
        else:
            self.f_ncnn_exe.pack_forget()

        # ComfyUI 서버 설정 행
        if is_comfyui:
            self.f_comfyui.pack(fill="x", pady=(8, 0))
        else:
            self.f_comfyui.pack_forget()

        # 모델 행
        self.f_esrgan_model.pack_forget()
        self.f_re_ncnn_model.pack_forget()
        self.f_w2x_model.pack_forget()
        if is_python or is_ncnn_re:
            self.f_esrgan_model.pack(fill="x", pady=2)
        elif is_ncnn_w2:
            self.f_w2x_model.pack(fill="x", pady=2)

        # 노이즈 (waifu2x 전용)
        if is_ncnn_w2:
            self.f_denoise.pack(fill="x", pady=3)
        else:
            self.f_denoise.pack_forget()

    def _refresh_rgba_ui(self):
        enabled = self.var_split.get()
        for cb in self.ch_checks:
            cb.configure(state="normal" if enabled else "disabled")

    # ── 파일 목록 핸들러 ─────────────────
    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="DDS 파일 선택",
            filetypes=[("DDS 파일", "*.dds *.DDS"), ("모든 파일", "*.*")]
        )
        self._push_files(files)

    def _add_folder(self):
        folder = filedialog.askdirectory(title="폴더 선택")
        if not folder:
            return
        rec = self.var_recursive.get()
        patterns = ["**/*.dds", "**/*.DDS"] if rec else ["*.dds", "*.DDS"]
        seen: set[str] = set()
        files = []
        for pat in patterns:
            for p in Path(folder).glob(pat):
                key = str(p).lower()
                if key not in seen:
                    seen.add(key)
                    files.append(str(p))
        self._push_files(files)
        if not self.var_output_dir.get():
            self.var_output_dir.set(os.path.join(folder, "output_tga"))

    def _push_files(self, files):
        existing = set(self.input_files)
        new = [f for f in files if f not in existing]
        self.input_files.extend(new)
        for f in new:
            self.listbox.insert("end", os.path.basename(f))
        self._update_count()

    def _remove_sel(self):
        for idx in reversed(self.listbox.curselection()):
            self.listbox.delete(idx)
            self.input_files.pop(idx)
        self._update_count()

    def _clear_files(self):
        self.input_files.clear()
        self.listbox.delete(0, "end")
        self._update_count()

    def _update_count(self):
        n = len(self.input_files)
        self.lbl_count.configure(text=f"파일 {n}개 선택됨")

    def _browse_output(self):
        d = filedialog.askdirectory(title="출력 폴더")
        if d:
            self.var_output_dir.set(d)

    def _browse_ncnn(self):
        f = filedialog.askopenfilename(
            title="ncnn 실행파일 선택",
            filetypes=[("실행파일", "*.exe"), ("모든 파일", "*.*")]
        )
        if f:
            self.var_ncnn_exe.set(f)

    # ── 처리 시작 / 중단 ─────────────────
    def _start(self):
        if not self.input_files:
            messagebox.showwarning("경고", "DDS 파일을 추가하세요.")
            return
        out_dir = self.var_output_dir.get().strip()
        if not out_dir:
            messagebox.showwarning("경고", "출력 폴더를 지정하세요.")
            return

        channels = []
        if self.var_split.get():
            for ch, var in [("R", self.var_ch_r), ("G", self.var_ch_g),
                            ("B", self.var_ch_b), ("A", self.var_ch_a)]:
                if var.get():
                    channels.append(ch)

        engine = self.var_engine.get()
        model_name, _ = ESRGAN_MODELS.get(
            self.var_esrgan_model.get(), ("RealESRGAN_x4plus", 4))
        ncnn_re_model = REALESRGAN_NCNN_MODELS.get(self.var_ncnn_re_model.get(), "")
        w2x_model     = WAIFU2X_MODELS.get(self.var_w2x_model.get(), "")
        ncnn_model    = ncnn_re_model if engine == "realesrgan_ncnn" else w2x_model

        try:
            target = int(self.var_target.get())
        except ValueError:
            target = DEFAULT_TARGET

        settings = {
            "engine":         engine,
            "ncnn_exe":       self.var_ncnn_exe.get(),
            "model_name":     model_name,
            "ncnn_model":     ncnn_model,
            "scale":          int(self.var_scale.get().replace("x", "")),
            "target_size":    target,
            "gpu_id":         self.var_gpu_id.get(),
            "tile_size":      self.var_tile.get(),
            "denoise":        self.var_denoise.get(),
            "output_dir":     out_dir,
            "keep_structure": self.var_keep_structure.get(),
            "input_base":     str(Path(self.input_files[0]).parent),
            "overwrite":      self.var_overwrite.get(),
            "split_channels": bool(channels),
            "channels":       channels,
            "comfyui_host":   self.var_comfyui_host.get().strip(),
            "comfyui_port":   int(self.var_comfyui_port.get() or 8188),
            "comfyui_model":  self.var_comfyui_model.get(),
        }

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.prog_var.set(0)
        self.lbl_status.configure(text="처리 중…")

        self.worker = ProcessWorker(list(self.input_files), settings, self.q)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.btn_stop.configure(state="disabled")

    # ── 로그 ─────────────────────────────
    def _log(self, msg: str, level: str = "INFO"):
        self.log_text.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n", level)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ── 큐 폴링 ──────────────────────────
    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                t = item["type"]
                if t == "log":
                    self._log(item["msg"], item.get("level", "INFO"))
                elif t == "progress":
                    cur, tot = item["cur"], item["tot"]
                    pct = cur / tot * 100 if tot else 0
                    self.prog_var.set(pct)
                    fname = item.get("fname", "")
                    self.lbl_status.configure(
                        text=f"{cur} / {tot}  {fname}" if fname else f"{cur} / {tot}")
                elif t == "done":
                    self.prog_var.set(100)
                    self.lbl_status.configure(text="✅  완료!")
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self._log("모든 파일 처리 완료", "OK")
                    messagebox.showinfo("완료", "변환이 완료되었습니다!")
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    # ── 의존성 확인 ───────────────────────
    def _check_deps_async(self):
        def _check():
            missing = []
            try:
                from PIL import Image  # noqa
            except ImportError:
                missing.append("Pillow")
            try:
                import numpy  # noqa
            except ImportError:
                missing.append("numpy")
            if missing:
                self.q.put({
                    "type": "log",
                    "msg": f"⚠ 필수 패키지 없음: {', '.join(missing)}\n"
                           f"   → pip install {' '.join(missing)}",
                    "level": "WARN"
                })
            else:
                self.q.put({
                    "type": "log",
                    "msg": "의존성 확인 완료 (Pillow, numpy)",
                    "level": "OK"
                })
        threading.Thread(target=_check, daemon=True).start()


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────
if __name__ == "__main__":
    try:
        from PIL import Image    # noqa
        import numpy             # noqa
    except ImportError:
        print("필수 패키지를 설치합니다...")
        subprocess.run([sys.executable, "-m", "pip", "install", "Pillow", "numpy", "-q"],
                       check=False)

    app = App()
    app.mainloop()
