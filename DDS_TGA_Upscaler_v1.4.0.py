#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDS → TGA 업스케일러
- DDS 파일을 TGA로 변환 (4K 텍스처 기준)
- Real-ESRGAN (Python) / ComfyUI 엔진 지원
- RGBA 채널 분리 기능
- 폴더 또는 개별 파일 처리
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
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
APP_VERSION = "1.4.0"
COMFYUI_CLOUD_BASE_URL = "https://cloud.comfy.org"
DEFAULT_TARGET = 4096

ESRGAN_MODELS = {
    "RealESRGAN_x4plus  (범용)":           ("RealESRGAN_x4plus",           4),
    "RealESRGAN_x4plus_anime (애니)":      ("RealESRGAN_x4plus_anime_6B",  4),
    "RealESRNet_x4plus  (빠름)":           ("RealESRNet_x4plus",           4),
    "RealESRGAN_x2plus  (x2)":             ("RealESRGAN_x2plus",           2),
}


COMFYUI_UPSCALE_MODELS = [
    "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "4x-UltraSharp.pth",
    "4x_NMKD-Siax_200k.pth",
    "8x_NMKD-Superscale.pth",
]

COMFYUI_CONTROLNET_MODELS = [
    "control_v11f1e_sd15_tile.pth",
    "control_v11f1e_sd15_tile_fp8_e4m3fn.safetensors",
    "controlnet-tile-sdxl-1.0.safetensors",
]

COMFYUI_SD_CHECKPOINTS = [
    "v1-5-pruned-emaonly.safetensors",
    "v1-5-pruned-emaonly.ckpt",
    "dreamshaper_8.safetensors",
    "revAnimated_v122.safetensors",
    "realisticVisionV60B1_v51VAE.safetensors",
]

COMFYUI_SAMPLERS = [
    "euler_ancestral",
    "euler",
    "dpmpp_2m_sde",
    "dpm_2_ancestral",
    "dpmpp_2m",
    "dpmpp_sde",
]

GEMINI_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.0-flash-preview-image-generation",
]
GEMINI_RESOLUTIONS = ["auto", "1K", "2K", "4K", "8K"]
GEMINI_DEFAULT_PROMPT = (
    "upscale this. refine details. preserve text. retain composition."
)
GEMINI_SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively "
    "invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)

# ──────────────────────────────────────────
# 툴팁
# ──────────────────────────────────────────
class Tooltip:
    """마우스 호버 시 툴팁 표시"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text   = text
        self.tw     = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tw or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg="#45475a")
        lbl = tk.Label(tw, text=self.text, justify="left",
                       background="#1e1e2e", foreground="#cdd6f4",
                       relief="flat", borderwidth=0,
                       font=("Segoe UI", 9),
                       padx=8, pady=6, wraplength=340)
        lbl.pack(padx=1, pady=1)

    def _hide(self, _event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None


# ──────────────────────────────────────────
# 업스케일 엔진
# ──────────────────────────────────────────
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


class ComfyUIEngine:
    """ComfyUI REST API 업스케일 엔진 (로컬 서버 http://host:port)"""
    name = "comfyui"

    def is_available(self):
        return True  # 런타임에 연결 시도

    def upscale(self, img, scale,
                comfyui_host="127.0.0.1", comfyui_port=8188,
                comfyui_use_cloud=False, comfyui_api_key="",
                comfyui_model="RealESRGAN_x4plus.pth",
                comfyui_workflow="upscale",
                comfyui_sd_model="v1-5-pruned-emaonly.safetensors",
                comfyui_cn_model="control_v11f1e_sd15_tile.pth",
                comfyui_cn_strength=1.0,
                comfyui_denoise=0.35,
                comfyui_cfg=7.0,
                comfyui_steps=20,
                comfyui_sampler="euler_ancestral",
                comfyui_pos_prompt="high quality texture, detailed",
                comfyui_neg_prompt="blurry, low quality, artifacts",
                comfyui_timeout=600,
                gemini_model="gemini-3-pro-image-preview",
                gemini_prompt=GEMINI_DEFAULT_PROMPT,
                gemini_resolution="4K",
                gemini_seed=-1,
                src_png_path=None,
                log_fn=None,
                **kw):
        import random
        from PIL import Image

        def _log(msg, level="INFO"):
            if log_fn:
                log_fn(msg, level)

        # ── 서버 모드 분기 ────────────────────────
        if comfyui_use_cloud:
            base_url   = COMFYUI_CLOUD_BASE_URL
            api_prefix = "/api"
            auth_hdr   = {"X-API-Key": comfyui_api_key}
            _log(f"  [ComfyUI] 클라우드 모드 → {base_url}")
        else:
            base_url   = f"http://{comfyui_host}:{comfyui_port}"
            api_prefix = ""
            auth_hdr   = {}
            _log(f"  [ComfyUI] 로컬 모드 → {base_url}")

        # ── 1. 이미지 업로드 ──────────────────────
        has_alpha = img.mode in ("RGBA", "LA")
        alpha_ch  = img.split()[-1] if has_alpha else None

        # 저장된 PNG 파일이 있으면 그대로 읽어서 업로드 (변환 과정 없음)
        if src_png_path and os.path.exists(src_png_path):
            with open(src_png_path, "rb") as f:
                img_bytes = f.read()
            _log(f"  [ComfyUI] 저장된 PNG 사용: {os.path.basename(src_png_path)}")
        else:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            buf.seek(0)
            img_bytes = buf.read()

        upload_filename = f"dds_upscaler_{uuid.uuid4().hex}.png"
        boundary = uuid.uuid4().hex

        # 클라우드 모드는 type=input 필드 추가 필요
        if comfyui_use_cloud:
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{upload_filename}"\r\n'
                f"Content-Type: image/png\r\n\r\n"
            ).encode() + img_bytes + (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="type"\r\n\r\n'
                f"input\r\n--{boundary}--\r\n"
            ).encode()
        else:
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{upload_filename}"\r\n'
                f"Content-Type: image/png\r\n\r\n"
            ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

        kb = len(img_bytes) / 1024
        _log(f"  [ComfyUI] 이미지 업로드 중… ({kb:.0f} KB)")
        upload_headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        upload_headers.update(auth_hdr)
        req = urllib.request.Request(
            f"{base_url}{api_prefix}/upload/image",
            data=body,
            headers=upload_headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                upload_result = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(f"ComfyUI 이미지 업로드 실패: {e}\n{'API 키를 확인하세요.' if comfyui_use_cloud else '서버가 실행 중인지 확인하세요.'}")

        uploaded_name = upload_result.get("name", upload_filename)
        _log(f"  [ComfyUI] 업로드 완료 → {uploaded_name}")

        # ── 2. 워크플로우(프롬프트) 전송 ──────────
        client_id = uuid.uuid4().hex

        if comfyui_workflow == "controlnet_tile":
            # ControlNet Tile: ESRGAN 초기 업스케일 → SD img2img (구조 보존)
            w, h = img.size
            target_w = w * scale
            target_h = h * scale
            seed = random.randint(0, 2 ** 32 - 1)
            workflow = {
                "1":  {"class_type": "CheckpointLoaderSimple",
                       "inputs": {"ckpt_name": comfyui_sd_model}},
                "2":  {"class_type": "CLIPTextEncode",
                       "inputs": {"text": comfyui_pos_prompt, "clip": ["1", 1]}},
                "3":  {"class_type": "CLIPTextEncode",
                       "inputs": {"text": comfyui_neg_prompt, "clip": ["1", 1]}},
                "4":  {"class_type": "LoadImage",
                       "inputs": {"image": uploaded_name}},
                "5":  {"class_type": "UpscaleModelLoader",
                       "inputs": {"model_name": comfyui_model}},
                "6":  {"class_type": "ImageUpscaleWithModel",
                       "inputs": {"upscale_model": ["5", 0], "image": ["4", 0]}},
                "7":  {"class_type": "ImageScale",
                       "inputs": {"image": ["6", 0], "upscale_method": "bilinear",
                                  "width": target_w, "height": target_h, "crop": "disabled"}},
                "8":  {"class_type": "VAEEncode",
                       "inputs": {"pixels": ["7", 0], "vae": ["1", 2]}},
                "9":  {"class_type": "ControlNetLoader",
                       "inputs": {"control_net_name": comfyui_cn_model}},
                "10": {"class_type": "ControlNetApply",
                       "inputs": {"conditioning": ["2", 0], "control_net": ["9", 0],
                                  "image": ["7", 0], "strength": comfyui_cn_strength}},
                "11": {"class_type": "KSampler",
                       "inputs": {"model": ["1", 0], "positive": ["10", 0],
                                  "negative": ["3", 0], "latent_image": ["8", 0],
                                  "seed": seed, "steps": comfyui_steps,
                                  "cfg": comfyui_cfg, "sampler_name": comfyui_sampler,
                                  "scheduler": "karras", "denoise": comfyui_denoise}},
                "12": {"class_type": "VAEDecode",
                       "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
                "13": {"class_type": "SaveImage",
                       "inputs": {"images": ["12", 0],
                                  "filename_prefix": f"dds_cntile_{uuid.uuid4().hex[:8]}"}},
            }
        elif comfyui_workflow == "gemini_image":
            # Nano Banana Pro — GeminiImage2Node 워크플로우
            seed = gemini_seed if gemini_seed >= 0 else random.randint(0, 2 ** 32 - 1)
            workflow = {
                "2": {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_name},
                    "_meta": {"title": "이미지 로드"},
                },
                "4": {
                    "class_type": "GeminiImage2Node",
                    "inputs": {
                        "prompt":              gemini_prompt,
                        "model":               gemini_model,
                        "seed":                seed,
                        "aspect_ratio":        "auto",
                        "resolution":          gemini_resolution,
                        "response_modalities": "IMAGE+TEXT",
                        "system_prompt":       GEMINI_SYSTEM_PROMPT,
                        "images":              ["2", 0],
                    },
                    "_meta": {"title": "Nano Banana Pro (Google Gemini Image)"},
                },
                "5": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "filename_prefix": f"dds_gemini_{uuid.uuid4().hex[:8]}",
                        "images":          ["4", 0],
                    },
                },
            }
        else:
            # 기본 업스케일 모델 워크플로우
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
                        "images":          ["3", 0],
                        "filename_prefix": f"dds_up_{uuid.uuid4().hex[:8]}",
                    },
                },
            }
        wf_labels = {
            "controlnet_tile": "ControlNet Tile",
            "gemini_image":    "Gemini (Nano Banana Pro)",
        }
        wf_label = wf_labels.get(comfyui_workflow, "업스케일 모델")
        _log(f"  [ComfyUI] 워크플로우 전송 중… ({wf_label})")
        payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        prompt_headers = {"Content-Type": "application/json"}
        prompt_headers.update(auth_hdr)
        req2 = urllib.request.Request(
            f"{base_url}{api_prefix}/prompt",
            data=payload,
            headers=prompt_headers,
        )
        try:
            with urllib.request.urlopen(req2, timeout=30) as resp:
                prompt_result = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(f"ComfyUI 프롬프트 전송 실패: {e}")

        prompt_id = prompt_result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI prompt_id 없음: {prompt_result}")
        _log(f"  [ComfyUI] 작업 등록 완료 (ID: {prompt_id[:8]}…)")

        # ── 3. 완료 폴링 (타임아웃: comfyui_timeout초, 2초 간격) ──
        poll_interval = 2.0
        max_ticks = max(1, int(comfyui_timeout / poll_interval))
        _log(f"  [ComfyUI] 처리 대기 중… (최대 {comfyui_timeout}초)")
        out_info = None
        last_status = ""
        for tick in range(max_ticks):
            time.sleep(poll_interval)
            try:
                if comfyui_use_cloud:
                    hist_req = urllib.request.Request(
                        f"{base_url}/api/history_v2/{prompt_id}",
                        headers=auth_hdr,
                    )
                    with urllib.request.urlopen(hist_req, timeout=10) as resp:
                        history = json.loads(resp.read())
                    status = history.get("status", "")
                    if status != last_status:
                        _log(f"  [ComfyUI] 상태: {status}")
                        last_status = status
                    if status == "failed":
                        raise RuntimeError(f"ComfyUI 클라우드 작업 실패: {history.get('error', '')}")
                    if status != "completed":
                        continue
                    outputs = history.get("outputs", {})
                else:
                    hist_req = urllib.request.Request(
                        f"{base_url}/history/{prompt_id}",
                        headers=auth_hdr,
                    )
                    with urllib.request.urlopen(hist_req, timeout=10) as resp:
                        history = json.loads(resp.read())
                    if prompt_id not in history:
                        # 30초마다 대기 중 메시지
                        if tick % 15 == 0 and tick > 0:
                            _log(f"  [ComfyUI] 대기 중… ({tick * int(poll_interval)}초 경과)")
                        continue
                    outputs = history[prompt_id].get("outputs", {})
            except RuntimeError:
                raise
            except Exception:
                continue

            for node_out in outputs.values():
                imgs = node_out.get("images", [])
                if imgs:
                    out_info = imgs[0]
                    break
            if out_info is not None:
                break
        else:
            raise RuntimeError(f"ComfyUI 처리 타임아웃 ({comfyui_timeout}초 초과)")

        _log("  [ComfyUI] 처리 완료, 결과 다운로드 중…")

        # ── 4. 결과 이미지 다운로드 ──────────────
        params = urllib.parse.urlencode({
            "filename": out_info["filename"],
            "subfolder": out_info.get("subfolder", ""),
            "type": out_info.get("type", "output"),
        })
        view_req = urllib.request.Request(
            f"{base_url}{api_prefix}/view?{params}",
            headers=auth_hdr,
        )
        try:
            with urllib.request.urlopen(view_req, timeout=30) as resp:
                result_bytes = resp.read()
        except Exception as e:
            raise RuntimeError(f"ComfyUI 결과 다운로드 실패: {e}")

        result = Image.open(io.BytesIO(result_bytes)).convert("RGB")
        _log(f"  [ComfyUI] 다운로드 완료 → {result.size[0]}x{result.size[1]}")

        # ── 알파 채널 복원 ────────────────────────
        if has_alpha:
            alpha_up = alpha_ch.resize(result.size, Image.LANCZOS)
            result = result.convert("RGBA")
            result.putalpha(alpha_up)

        return result


# ──────────────────────────────────────────
# DDS 변환 유틸
# ──────────────────────────────────────────
def _decode_dds_t2d(path: str):
    """
    DDS 헤더를 직접 파싱하고 texture2ddecoder 로 픽셀 데이터를 디코딩합니다.
    DXT1/3/5, BC1~BC7, ATI1/ATI2 등 게임 텍스처 전 포맷 지원.
    pip install texture2ddecoder
    """
    import struct
    import numpy as np
    import texture2ddecoder
    from PIL import Image

    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] != b"DDS ":
        raise ValueError("DDS 매직 바이트 없음")

    height = struct.unpack_from("<I", raw, 12)[0]
    width  = struct.unpack_from("<I", raw, 16)[0]
    fourcc = raw[84:88]

    data_offset = 128
    dxgi_format = None
    if fourcc == b"DX10":
        dxgi_format = struct.unpack_from("<I", raw, 128)[0]
        data_offset = 148

    pdata = raw[data_offset:]

    FOURCC_MAP = {
        b"DXT1": texture2ddecoder.decode_bc1,
        b"DXT3": texture2ddecoder.decode_bc2,
        b"DXT5": texture2ddecoder.decode_bc3,
        b"ATI1": texture2ddecoder.decode_bc4,
        b"BC4U": texture2ddecoder.decode_bc4,
        b"BC4S": texture2ddecoder.decode_bc4,
        b"ATI2": texture2ddecoder.decode_bc5,
        b"BC5U": texture2ddecoder.decode_bc5,
        b"BC5S": texture2ddecoder.decode_bc5,
    }
    DXGI_MAP = {
        71: texture2ddecoder.decode_bc1,  72: texture2ddecoder.decode_bc1,
        74: texture2ddecoder.decode_bc2,  75: texture2ddecoder.decode_bc2,
        77: texture2ddecoder.decode_bc3,  78: texture2ddecoder.decode_bc3,
        80: texture2ddecoder.decode_bc4,  81: texture2ddecoder.decode_bc4,
        83: texture2ddecoder.decode_bc5,  84: texture2ddecoder.decode_bc5,
        98: texture2ddecoder.decode_bc7,  99: texture2ddecoder.decode_bc7,
    }

    if fourcc in FOURCC_MAP:
        bgra = FOURCC_MAP[fourcc](pdata, width, height)
    elif fourcc == b"DX10" and dxgi_format is not None:
        if dxgi_format in (95, 96):  # BC6H
            bgra = texture2ddecoder.decode_bc6(pdata, width, height)
        elif dxgi_format in DXGI_MAP:
            bgra = DXGI_MAP[dxgi_format](pdata, width, height)
        else:
            raise ValueError(f"미지원 DXGI 포맷: {dxgi_format}")
    else:
        raise ValueError(f"미지원 FourCC: {fourcc}")

    # texture2ddecoder 출력은 BGRA → RGBA 변환
    arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
    arr = arr[:, :, [2, 1, 0, 3]]
    return Image.fromarray(arr, mode="RGBA")


def read_dds(path: str):
    """
    이미지 파일을 PIL Image로 반환합니다.
    - PNG/TGA/JPEG/BMP 등: Pillow로 직접 읽기
    - DDS: 1차 Pillow → 2차 texture2ddecoder → 3차 imageio
    """
    from PIL import Image
    import numpy as np

    # DDS가 아닌 일반 이미지 포맷은 Pillow로 바로 읽기
    if Path(path).suffix.lower() != ".dds":
        try:
            img = Image.open(path)
            img.load()
            if img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGBA")
            return img
        except Exception as e:
            raise RuntimeError(f"이미지 읽기 실패: {e}")

    # ── 1차: Pillow 내장 ─────────────────────────────────────
    try:
        img = Image.open(path)
        img.load()
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA")
        if np.array(img.convert("RGB")).min() < 250:
            return img
    except Exception:
        pass

    # ── 2차: texture2ddecoder ────────────────────────────────
    try:
        return _decode_dds_t2d(path)
    except ImportError:
        pass
    except Exception:
        pass

    # ── 3차: imageio ─────────────────────────────────────────
    try:
        import imageio.v3 as iio
        arr = iio.imread(path)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        return Image.fromarray(arr[:, :, :3], mode="RGB")
    except Exception:
        pass

    raise RuntimeError(
        f"DDS 읽기 실패: {path}\n"
        "지원하지 않는 DDS 포맷입니다.\n"
        "해결: pip install texture2ddecoder"
    )


def write_image(img, path: str, fmt: str = "png"):
    """fmt: 'png' or 'tga'"""
    fmt = fmt.lower()
    if fmt == "tga":
        img.save(path, format="TGA")
    else:
        img.save(path, format="PNG")


def split_rgba_channels(img, channels: list, out_path: str, fmt: str = "png") -> list:
    import numpy as np
    from PIL import Image

    arr = np.array(img.convert("RGBA"))
    ch_map = {"R": 0, "G": 1, "B": 2, "A": 3}
    base = Path(out_path)

    def _save_channel(ch):
        idx = ch_map.get(ch.upper())
        if idx is None:
            return None
        ext = "tga" if fmt.lower() == "tga" else "png"
        out = base.parent / f"{base.stem}_{ch.upper()}.{ext}"
        save_fmt = "TGA" if fmt.lower() == "tga" else "PNG"
        Image.fromarray(arr[:, :, idx], mode="L").save(str(out), format=save_fmt)
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
        ename = s.get("engine", "realesrgan_python")
        if ename == "comfyui":
            engine = ComfyUIEngine()
        else:
            engine = RealESRGANPythonEngine()

        if not engine.is_available():
            self._log(f"엔진 '{ename}' 를 사용할 수 없습니다. 패키지 설치를 확인하세요.", "WARN")

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

                out_fmt   = s.get("output_format", "png").lower()
                ext       = "tga" if out_fmt == "tga" else "png"
                stem      = Path(dds_path).stem
                tga_path  = os.path.join(out_dir, f"{stem}.{ext}")

                if os.path.exists(tga_path) and not s.get("overwrite", True):
                    self._log(f"  건너뜀 (이미 존재): {stem}.{ext}", "WARN")
                    # 건너뛰어도 다음 파일 prefetch
                    if i + 1 < len(self.files):
                        _prefetch_future = _prefetch_ex.submit(_prefetch, self.files[i + 1])
                    continue

                # 이미지 읽기 (prefetch 결과 우선 사용)
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

                # 픽셀 통계로 디코딩 정상 여부 확인
                import numpy as np
                _arr = np.array(img.convert("RGB"))
                _mean = _arr.mean()
                _min, _max = int(_arr.min()), int(_arr.max())
                self._log(f"  읽기 완료: {img.size[0]}x{img.size[1]} {img.mode}  "
                          f"픽셀 범위={_min}~{_max}  평균={_mean:.1f}")
                if _max <= 5:
                    self._log("  ⚠ 이미지가 거의 검정 — 디코딩 실패 가능성", "WARN")
                elif _min >= 250:
                    self._log("  ⚠ 이미지가 거의 흰색 — 디코딩 실패 가능성 (pip install texture2ddecoder)", "WARN")

                # ── PNG 변환 저장 (ComfyUI 업로드 전 중간 파일) ──────────
                src_dir = os.path.join(s.get("output_dir", out_dir), "_source")
                os.makedirs(src_dir, exist_ok=True)
                src_png = os.path.join(src_dir, f"{stem}.png")
                img.save(src_png, format="PNG")
                self._log(f"  PNG 변환 저장: _source/{stem}.png")

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
                        src_png_path        = src_png if ename == "comfyui" else None,
                        log_fn              = self._log,
                        model_name          = s.get("model_name", "RealESRGAN_x4plus"),
                        gpu_id              = s.get("gpu_id", 0),
                        tile                = s.get("tile_size", 0),
                        comfyui_use_cloud   = s.get("comfyui_use_cloud", False),
                        comfyui_api_key     = s.get("comfyui_api_key", ""),
                        comfyui_host        = s.get("comfyui_host", "127.0.0.1"),
                        comfyui_port        = s.get("comfyui_port", 8188),
                        comfyui_model       = s.get("comfyui_model", "RealESRGAN_x4plus.pth"),
                        comfyui_workflow    = s.get("comfyui_workflow", "upscale"),
                        comfyui_sd_model    = s.get("comfyui_sd_model", "v1-5-pruned-emaonly.safetensors"),
                        comfyui_cn_model    = s.get("comfyui_cn_model", "control_v11f1e_sd15_tile.pth"),
                        comfyui_cn_strength = s.get("comfyui_cn_strength", 1.0),
                        comfyui_denoise     = s.get("comfyui_denoise", 0.35),
                        comfyui_cfg         = s.get("comfyui_cfg", 7.0),
                        comfyui_steps       = s.get("comfyui_steps", 20),
                        comfyui_sampler     = s.get("comfyui_sampler", "euler_ancestral"),
                        comfyui_timeout     = s.get("comfyui_timeout", 600),
                        comfyui_pos_prompt  = s.get("comfyui_pos_prompt", "high quality texture, detailed"),
                        comfyui_neg_prompt  = s.get("comfyui_neg_prompt", "blurry, low quality, artifacts"),
                        gemini_model        = s.get("gemini_model", GEMINI_MODELS[0]),
                        gemini_prompt       = s.get("gemini_prompt", GEMINI_DEFAULT_PROMPT),
                        gemini_resolution   = s.get("gemini_resolution", "4K"),
                        gemini_seed         = s.get("gemini_seed", -1),
                    )
                    self._log(f"  업스케일 완료: {img_up.size[0]}x{img_up.size[1]}")

                    # 목표 크기 초과 시 크롭/리사이즈
                    if max(img_up.size) > target:
                        from PIL import Image
                        img_up = img_up.resize((target, target), Image.LANCZOS)

                # 저장
                write_image(img_up, tga_path, fmt=out_fmt)
                self._log(f"  저장: {os.path.basename(tga_path)}", "OK")

                # RGBA 채널 분리
                if s.get("split_channels") and s.get("channels"):
                    saved = split_rgba_channels(img_up, s["channels"], tga_path, fmt=out_fmt)
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
        self.geometry("860x760")
        self.minsize(740, 580)
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
        self.var_esrgan_model   = tk.StringVar(value=list(ESRGAN_MODELS.keys())[0])
        self.var_scale          = tk.StringVar(value="4x")
        self.var_target         = tk.StringVar(value=str(DEFAULT_TARGET))
        self.var_gpu_id         = tk.IntVar(value=0)
        self.var_tile           = tk.IntVar(value=0)
        self.var_output_dir     = tk.StringVar()
        self.var_output_format  = tk.StringVar(value="png")
        self.var_keep_structure = tk.BooleanVar(value=True)
        self.var_overwrite      = tk.BooleanVar(value=True)
        self.var_recursive      = tk.BooleanVar(value=True)
        self.var_split          = tk.BooleanVar(value=False)
        self.var_ch_r           = tk.BooleanVar(value=True)
        self.var_ch_g           = tk.BooleanVar(value=True)
        self.var_ch_b           = tk.BooleanVar(value=True)
        self.var_ch_a           = tk.BooleanVar(value=True)
        self.var_comfyui_use_cloud   = tk.BooleanVar(value=False)
        self.var_comfyui_api_key     = tk.StringVar(value="")
        self.var_comfyui_host        = tk.StringVar(value="127.0.0.1")
        self.var_comfyui_port        = tk.StringVar(value="8188")
        self.var_comfyui_model       = tk.StringVar(value=COMFYUI_UPSCALE_MODELS[0])
        self.var_comfyui_workflow    = tk.StringVar(value="upscale")
        self.var_comfyui_sd_model    = tk.StringVar(value=COMFYUI_SD_CHECKPOINTS[0])
        self.var_comfyui_cn_model    = tk.StringVar(value=COMFYUI_CONTROLNET_MODELS[0])
        self.var_comfyui_cn_strength = tk.DoubleVar(value=1.0)
        self.var_comfyui_denoise     = tk.DoubleVar(value=0.35)
        self.var_comfyui_cfg         = tk.DoubleVar(value=7.0)
        self.var_comfyui_steps       = tk.IntVar(value=20)
        self.var_comfyui_sampler     = tk.StringVar(value="euler_ancestral")
        self.var_comfyui_timeout     = tk.IntVar(value=600)
        self.var_comfyui_pos_prompt  = tk.StringVar(value="high quality texture, detailed")
        self.var_comfyui_neg_prompt  = tk.StringVar(value="blurry, low quality, artifacts")
        self.var_gemini_model        = tk.StringVar(value=GEMINI_MODELS[0])
        self.var_gemini_prompt       = tk.StringVar(value=GEMINI_DEFAULT_PROMPT)
        self.var_gemini_resolution   = tk.StringVar(value="4K")
        self.var_gemini_seed         = tk.IntVar(value=-1)

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

        # 하단 패널 먼저 pack (side="bottom") — 항상 화면에 고정
        self._build_bottom()

        # 노트북 탭 (남은 공간 확장)
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
        btn_add = ttk.Button(bf, text="파일 추가", command=self._add_files)
        btn_add.pack(side="left", padx=(0, 4))
        Tooltip(btn_add, "이미지 파일을 개별 선택하여 추가합니다.\n지원 포맷: DDS, PNG, TGA, JPEG, BMP, WebP, TIFF\n여러 파일을 한 번에 선택할 수 있습니다.")
        btn_folder = ttk.Button(bf, text="폴더 추가", command=self._add_folder)
        btn_folder.pack(side="left", padx=(0, 4))
        Tooltip(btn_folder, "폴더 안의 모든 이미지 파일을 추가합니다.\n지원 포맷: DDS, PNG, TGA, JPEG, BMP, WebP, TIFF\n'하위 폴더 포함'이 체크된 경우 하위 폴더까지 탐색합니다.")
        btn_rem = ttk.Button(bf, text="선택 삭제", command=self._remove_sel)
        btn_rem.pack(side="left", padx=(0, 4))
        Tooltip(btn_rem, "목록에서 선택된 항목을 삭제합니다.")
        btn_clr = ttk.Button(bf, text="전체 초기화", command=self._clear_files)
        btn_clr.pack(side="left")
        Tooltip(btn_clr, "파일 목록 전체를 비웁니다.")
        cb_rec = ttk.Checkbutton(bf, text="하위 폴더 포함", variable=self.var_recursive,
                        style="TCheckbutton")
        cb_rec.pack(side="right")
        Tooltip(cb_rec, "폴더 추가 시 하위 폴더의 DDS 파일도 재귀적으로 탐색합니다.")

        self.lbl_count = ttk.Label(c, text="파일 0개 선택됨", style="Sub.TLabel")
        self.lbl_count.pack(anchor="w", pady=(2, 0))

        # 출력 카드
        c2 = self._card(parent, "💾  출력 설정")

        r1 = self._row(c2, "출력 폴더:")
        ent_out = ttk.Entry(r1, textvariable=self.var_output_dir)
        ent_out.pack(side="left", fill="x", expand=True, padx=4)
        Tooltip(ent_out, "변환된 파일이 저장될 폴더 경로입니다.\n비워두면 원본 파일과 같은 폴더에 저장됩니다.")
        ttk.Button(r1, text="찾아보기", command=self._browse_output).pack(side="right")

        r_fmt = self._row(c2, "출력 포맷:")
        rb_png = ttk.Radiobutton(r_fmt, text="PNG", variable=self.var_output_format,
                                 value="png", style="TRadiobutton")
        rb_png.pack(side="left", padx=(4, 16))
        rb_tga = ttk.Radiobutton(r_fmt, text="TGA", variable=self.var_output_format,
                                 value="tga", style="TRadiobutton")
        rb_tga.pack(side="left")
        Tooltip(rb_png, "PNG 형식으로 저장합니다.\n무손실 압축, 알파 채널 지원.\n범용성이 높고 용량이 작습니다.")
        Tooltip(rb_tga, "TGA 형식으로 저장합니다.\n무압축 또는 RLE 압축, 알파 채널 지원.\n일부 게임 엔진에서 요구합니다.")

        r2 = self._row(c2)
        cb_struct = ttk.Checkbutton(r2, text="폴더 구조 유지", variable=self.var_keep_structure,
                        style="TCheckbutton")
        cb_struct.pack(side="left", padx=(0, 20))
        Tooltip(cb_struct, "원본의 폴더 계층 구조를 출력 폴더에도 그대로 유지합니다.\n예: input/sub/a.dds → output/sub/a.png")
        cb_ow = ttk.Checkbutton(r2, text="기존 파일 덮어쓰기", variable=self.var_overwrite,
                        style="TCheckbutton")
        cb_ow.pack(side="left")
        Tooltip(cb_ow, "출력 폴더에 같은 이름의 파일이 이미 있을 때\n덮어쓸지(체크) 건너뛸지(미체크) 결정합니다.")

        r3 = self._row(c2, "목표 해상도:")
        ent_target = ttk.Entry(r3, textvariable=self.var_target, width=8)
        ent_target.pack(side="left", padx=4)
        Tooltip(ent_target, "업스케일 후 목표 최대 해상도(픽셀)입니다.\n게임 텍스처 표준: 4096 (4K)\n이미 목표 크기 이상인 파일은 리사이즈만 수행합니다.")
        ttk.Label(r3, text="× px  (텍스처 4K = 4096)", style="Card.TLabel").pack(side="left")

    # ── 탭 2: 업스케일 엔진 ──────────────
    def _tab_engine(self, parent):
        # ── 스크롤 가능한 컨테이너 ──────────────
        canvas = tk.Canvas(parent, bg=self.BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        scroll_frame = ttk.Frame(canvas, style="TFrame")
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def _on_frame_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scroll_frame.bind("<MouseWheel>", _on_mousewheel)

        parent = scroll_frame  # 이후 위젯은 scroll_frame 안에 배치

        # 엔진 선택
        c = self._card(parent, "🚀  엔진 선택")

        engines = [
            ("Real-ESRGAN  Python 패키지  (pip install realesrgan basicsr torch)",
             "realesrgan_python"),
            ("ComfyUI  REST API  (로컬 ComfyUI 서버, http://127.0.0.1:8188)",
             "comfyui"),
        ]
        engine_tooltips = {
            "realesrgan_python": (
                "Real-ESRGAN Python 패키지\n\n"
                "pip으로 설치하는 GPU 업스케일러입니다.\n"
                "설치: pip install realesrgan basicsr torch\n\n"
                "  • 고품질 범용 업스케일\n"
                "  • NVIDIA GPU 권장 (CPU 사용 가능하나 매우 느림)\n"
                "  • 별도 exe 불필요"
            ),
            "comfyui": (
                "ComfyUI REST API\n\n"
                "로컬 또는 클라우드 ComfyUI 서버에 연결하여\n"
                "업스케일 워크플로우를 실행합니다.\n\n"
                "  • 로컬: ComfyUI 서버 실행 필요 (127.0.0.1:8188)\n"
                "  • 클라우드: cloud.comfy.org API 키 필요\n"
                "  • ControlNet Tile 사용 시 SD 모델 필요"
            ),
        }
        for label, val in engines:
            rb = ttk.Radiobutton(c, text=label, variable=self.var_engine, value=val,
                                 style="TRadiobutton",
                                 command=self._refresh_engine_ui)
            rb.pack(anchor="w", pady=2)
            Tooltip(rb, engine_tooltips.get(val, ""))

        # ComfyUI 서버 설정
        self.f_comfyui = ttk.Frame(c, style="Card.TFrame")

        # 클라우드 / 로컬 토글
        r_mode = self._row(self.f_comfyui, "서버 모드:", 12)
        rb_local = ttk.Radiobutton(r_mode, text="로컬  (http://host:port)",
                        variable=self.var_comfyui_use_cloud, value=False,
                        style="TRadiobutton",
                        command=self._refresh_comfyui_mode_ui)
        rb_local.pack(side="left", padx=(4, 12))
        rb_cloud2 = ttk.Radiobutton(r_mode, text="클라우드  (cloud.comfy.org)",
                        variable=self.var_comfyui_use_cloud, value=True,
                        style="TRadiobutton",
                        command=self._refresh_comfyui_mode_ui)
        rb_cloud2.pack(side="left")
        Tooltip(rb_local, "로컬에서 실행 중인 ComfyUI 서버에 연결합니다.\nComfyUI를 먼저 실행해야 하며, 기본 주소는 127.0.0.1:8188 입니다.")
        Tooltip(rb_cloud2, "cloud.comfy.org 클라우드 서버를 사용합니다.\nAPI 키가 필요합니다. platform.comfy.org에서 발급받으세요.")

        # 로컬 전용 행
        self.f_comfyui_local = ttk.Frame(self.f_comfyui, style="Card.TFrame")
        r_host = self._row(self.f_comfyui_local, "서버 주소:", 12)
        ttk.Entry(r_host, textvariable=self.var_comfyui_host,
                  width=20).pack(side="left", padx=4)
        ttk.Label(r_host, text="포트:", style="Card.TLabel").pack(side="left", padx=(8, 0))
        ttk.Entry(r_host, textvariable=self.var_comfyui_port,
                  width=7).pack(side="left", padx=4)
        self.f_comfyui_local.pack(fill="x")

        # 클라우드 전용 행
        self.f_comfyui_cloud = ttk.Frame(self.f_comfyui, style="Card.TFrame")
        r_key = self._row(self.f_comfyui_cloud, "API 키:", 12)
        self.ent_api_key = ttk.Entry(r_key, textvariable=self.var_comfyui_api_key,
                                     width=48, show="*")
        self.ent_api_key.pack(side="left", padx=4)
        ttk.Button(r_key, text="보기",
                   command=self._toggle_api_key_visibility).pack(side="left")
        Tooltip(self.ent_api_key,
            "ComfyUI Cloud API 키입니다.\n\n"
            "platform.comfy.org → API Keys 메뉴에서 발급받을 수 있습니다.\n"
            "키는 한 번만 표시되므로 안전한 곳에 보관하세요.")

        r_timeout = self._row(self.f_comfyui_cloud, "타임아웃:", 12)
        sp_timeout = ttk.Spinbox(r_timeout, from_=60, to=3600, increment=60,
                                 textvariable=self.var_comfyui_timeout, width=7)
        sp_timeout.pack(side="left", padx=4)
        ttk.Label(r_timeout, text="초  (클라우드 큐 대기 최대 시간)",
                  style="Sub.TLabel").pack(side="left")
        Tooltip(sp_timeout,
            "클라우드 작업 완료를 기다리는 최대 시간(초)입니다.\n\n"
            "  300  →  5분\n"
            "  600  → 10분 (기본값, 권장)\n"
            "  1800 → 30분 (대형 이미지 / 서버 혼잡 시)\n\n"
            "이 시간 안에 완료되지 않으면 해당 파일은 건너뜁니다.\n"
            "클라우드 서버는 계속 작업 중이므로 취소되지는 않습니다.")

        r_model = self._row(self.f_comfyui, "업스케일 모델:", 12)
        cb_up_model = ttk.Combobox(r_model, textvariable=self.var_comfyui_model,
                     values=COMFYUI_UPSCALE_MODELS, width=36)
        cb_up_model.pack(side="left", padx=4)
        ttk.Label(r_model, text="(models/upscale_models/ 에 배치)",
                  style="Sub.TLabel").pack(side="left")
        Tooltip(cb_up_model,
            "ComfyUI에서 사용할 ESRGAN 업스케일 모델입니다.\n\n"
            "  • RealESRGAN_x4plus      — 범용 고품질 (권장)\n"
            "  • RealESRGAN_x4plus_anime — 애니/만화 스타일\n"
            "  • 4x-UltraSharp          — 선명도 강조\n"
            "  • 8x_NMKD-Superscale     — 8배 초고해상도\n\n"
            "ComfyUI의 models/upscale_models/ 폴더에 파일을 배치해야 합니다.")

        # ── 워크플로우 선택 ──────────────────────
        r_wf1 = self._row(self.f_comfyui, "워크플로우:", 12)
        rb_up = ttk.Radiobutton(r_wf1, text="업스케일 모델  (빠름)",
                        variable=self.var_comfyui_workflow, value="upscale",
                        style="TRadiobutton",
                        command=self._refresh_comfyui_ui)
        rb_up.pack(side="left", padx=(4, 12))
        rb_cn = ttk.Radiobutton(r_wf1, text="ControlNet Tile  (SD 필요)",
                        variable=self.var_comfyui_workflow, value="controlnet_tile",
                        style="TRadiobutton",
                        command=self._refresh_comfyui_ui)
        rb_cn.pack(side="left")
        r_wf2 = self._row(self.f_comfyui, "", 12)
        rb_gm = ttk.Radiobutton(r_wf2, text="Gemini  (Nano Banana Pro)",
                        variable=self.var_comfyui_workflow, value="gemini_image",
                        style="TRadiobutton",
                        command=self._refresh_comfyui_ui)
        rb_gm.pack(side="left", padx=(4, 0))
        Tooltip(rb_up,
            "업스케일 모델 워크플로우\n\n"
            "ESRGAN 등 전용 업스케일 모델로 단순 확대합니다.\n"
            "빠르고 안정적이며 원본에 충실합니다.\n\n"
            "SD/ControlNet 불필요 — 모델 파일만 있으면 됩니다.")
        Tooltip(rb_cn,
            "ControlNet Tile 워크플로우\n\n"
            "① ESRGAN으로 초기 업스케일\n"
            "② SD img2img로 디테일 재생성 (구조 보존)\n\n"
            "SD 체크포인트와 ControlNet 모델이 필요합니다.")
        Tooltip(rb_gm,
            "Gemini Image 워크플로우 (Nano Banana Pro)\n\n"
            "ComfyUI의 GeminiImage2Node 커스텀 노드를 사용합니다.\n"
            "Google Gemini API로 고품질 이미지를 생성/업스케일합니다.\n\n"
            "필요 조건:\n"
            "  • ComfyUI에 Nano Banana Pro 커스텀 노드 설치\n"
            "  • ComfyUI 노드 설정에 Gemini API 키 등록\n\n"
            "API 키는 이 앱이 아닌 ComfyUI 서버에서 관리합니다.")

        # ── ControlNet Tile 세부 설정 (토글) ────
        self.f_comfyui_cn = ttk.Frame(self.f_comfyui, style="Card.TFrame")

        r_sd = self._row(self.f_comfyui_cn, "SD 체크포인트:", 14)
        cb_sd = ttk.Combobox(r_sd, textvariable=self.var_comfyui_sd_model,
                     values=COMFYUI_SD_CHECKPOINTS, width=36)
        cb_sd.pack(side="left", padx=4)
        ttk.Label(r_sd, text="(models/checkpoints/)", style="Sub.TLabel").pack(side="left")
        Tooltip(cb_sd,
            "Stable Diffusion 기본 모델 (체크포인트)\n\n"
            "img2img 재생성에 사용할 SD 모델입니다.\n"
            "텍스처 업스케일에는 사실적인 모델 권장:\n"
            "  • v1-5-pruned-emaonly — SD 1.5 표준 범용\n"
            "  • realisticVision — 사실적 질감\n"
            "  • dreamshaper — 세부 묘사 강화\n\n"
            "ComfyUI의 models/checkpoints/ 폴더에 있어야 합니다.")

        r_cn = self._row(self.f_comfyui_cn, "ControlNet 모델:", 14)
        cb_cn = ttk.Combobox(r_cn, textvariable=self.var_comfyui_cn_model,
                     values=COMFYUI_CONTROLNET_MODELS, width=36)
        cb_cn.pack(side="left", padx=4)
        ttk.Label(r_cn, text="(models/controlnet/)", style="Sub.TLabel").pack(side="left")
        Tooltip(cb_cn,
            "ControlNet Tile 모델\n\n"
            "원본 이미지의 구조(형태, 색상 배치)를 유지하면서\n"
            "SD가 세부 디테일만 재생성하도록 제어합니다.\n\n"
            "  • control_v11f1e_sd15_tile — SD 1.5용 표준 Tile\n"
            "  • controlnet-tile-sdxl — SDXL용 (더 높은 품질)\n\n"
            "models/controlnet/ 폴더에 배치해야 합니다.")

        r_dn = self._row(self.f_comfyui_cn, "Denoise:", 14)
        self.lbl_comfyui_dn = ttk.Label(r_dn, text="0.35", style="Card.TLabel", width=4)
        sc_dn = ttk.Scale(r_dn, from_=0.0, to=1.0, orient="horizontal",
                  variable=self.var_comfyui_denoise, length=120,
                  command=lambda v: self.lbl_comfyui_dn.configure(text=f"{float(v):.2f}"))
        sc_dn.pack(side="left", padx=4)
        self.lbl_comfyui_dn.pack(side="left")
        ttk.Label(r_dn, text="(낮을수록 원본 유지 — 0.3~0.5 권장)",
                  style="Sub.TLabel").pack(side="left", padx=8)
        Tooltip(sc_dn,
            "노이즈 제거 강도 (Denoising Strength)\n\n"
            "SD가 원본 이미지를 얼마나 바꿀지 결정합니다.\n\n"
            "  0.0 → 원본과 동일 (변화 없음)\n"
            "  0.3~0.5 → 구조 유지 + 디테일 향상 ★ 권장\n"
            "  0.7 이상 → 원본과 많이 달라질 수 있음\n"
            "  1.0 → 완전히 새로 생성\n\n"
            "텍스처 업스케일에는 0.3~0.45를 권장합니다.")

        r_cs = self._row(self.f_comfyui_cn, "CFG / Steps:", 14)
        sp_cfg = ttk.Spinbox(r_cs, from_=1.0, to=20.0, increment=0.5,
                    textvariable=self.var_comfyui_cfg,
                    width=6, format="%.1f")
        sp_cfg.pack(side="left", padx=4)
        ttk.Label(r_cs, text="CFG    Steps:", style="Card.TLabel").pack(side="left", padx=(8, 0))
        sp_steps = ttk.Spinbox(r_cs, from_=1, to=100,
                    textvariable=self.var_comfyui_steps, width=5)
        sp_steps.pack(side="left", padx=4)
        Tooltip(sp_cfg,
            "CFG Scale (Classifier Free Guidance)\n\n"
            "프롬프트를 얼마나 강하게 따를지 결정합니다.\n\n"
            "  1~4  → 프롬프트 무시, 자유로운 생성\n"
            "  5~8  → 균형 ★ 권장 (기본값 7)\n"
            "  10+  → 프롬프트에 과도하게 집착, 부자연스러울 수 있음\n\n"
            "텍스처 업스케일에는 6~8이 적당합니다.")
        Tooltip(sp_steps,
            "샘플링 스텝 수\n\n"
            "SD가 이미지를 생성하는 반복 횟수입니다.\n\n"
            "  10~15 → 빠르지만 품질 낮음\n"
            "  20~30 → 균형 ★ 권장 (기본값 20)\n"
            "  50+   → 품질 향상 미미, 처리 시간만 증가\n\n"
            "대부분의 경우 20으로 충분합니다.")

        r_smp = self._row(self.f_comfyui_cn, "샘플러:", 14)
        cb_smp = ttk.Combobox(r_smp, textvariable=self.var_comfyui_sampler,
                     values=COMFYUI_SAMPLERS, state="readonly",
                     width=22)
        cb_smp.pack(side="left", padx=4)
        Tooltip(cb_smp,
            "샘플링 알고리즘\n\n"
            "SD가 노이즈를 제거하는 방식입니다.\n\n"
            "  • euler_ancestral — 다양성 높음, 텍스처에 적합 ★\n"
            "  • euler           — 안정적, 깔끔한 결과\n"
            "  • dpmpp_2m_sde    — 고품질, 약간 느림\n"
            "  • dpmpp_2m        — 빠르고 품질 좋음\n\n"
            "특별한 이유 없으면 euler_ancestral을 권장합니다.")

        r_pos = self._row(self.f_comfyui_cn, "긍정 프롬프트:", 14)
        ent_pos = ttk.Entry(r_pos, textvariable=self.var_comfyui_pos_prompt, width=46)
        ent_pos.pack(side="left", padx=4, fill="x", expand=True)
        Tooltip(ent_pos,
            "긍정 프롬프트 (Positive Prompt)\n\n"
            "SD에게 '이렇게 만들어 달라'고 지시하는 텍스트입니다.\n\n"
            "텍스처 업스케일 권장:\n"
            "  high quality texture, detailed, sharp\n"
            "  4K texture, game asset, PBR material")

        r_neg = self._row(self.f_comfyui_cn, "부정 프롬프트:", 14)
        ent_neg = ttk.Entry(r_neg, textvariable=self.var_comfyui_neg_prompt, width=46)
        ent_neg.pack(side="left", padx=4, fill="x", expand=True)
        Tooltip(ent_neg,
            "부정 프롬프트 (Negative Prompt)\n\n"
            "SD에게 '이렇게 만들지 말라'고 지시하는 텍스트입니다.\n\n"
            "텍스처 업스케일 권장:\n"
            "  blurry, low quality, artifacts, noise,\n"
            "  watermark, text, deformed")

        # ── Gemini 세부 설정 (토글) ─────────────────
        self.f_comfyui_gemini = ttk.Frame(self.f_comfyui, style="Card.TFrame")

        r_gm_model = self._row(self.f_comfyui_gemini, "Gemini 모델:", 14)
        cb_gm_model = ttk.Combobox(r_gm_model, textvariable=self.var_gemini_model,
                         values=GEMINI_MODELS, width=38)
        cb_gm_model.pack(side="left", padx=4)
        Tooltip(cb_gm_model,
            "사용할 Google Gemini 모델입니다.\n\n"
            "  • gemini-3-pro-image-preview         — 최신 고품질\n"
            "  • gemini-2.0-flash-exp-image-generation — 빠른 생성\n\n"
            "ComfyUI의 Nano Banana Pro 노드가 지원하는 모델이어야 합니다.")

        r_gm_res = self._row(self.f_comfyui_gemini, "해상도:", 14)
        cb_gm_res = ttk.Combobox(r_gm_res, textvariable=self.var_gemini_resolution,
                        values=GEMINI_RESOLUTIONS, state="readonly", width=10)
        cb_gm_res.pack(side="left", padx=4)
        Tooltip(cb_gm_res,
            "출력 이미지 해상도입니다.\n\n"
            "  auto → 입력 이미지 크기에 맞게 자동 결정\n"
            "  4K   → 4096px 수준 (텍스처 표준)\n\n"
            "Gemini 모델이 지원하는 범위 내에서 동작합니다.")

        r_gm_seed = self._row(self.f_comfyui_gemini, "시드:", 14)
        sp_gm_seed = ttk.Spinbox(r_gm_seed, from_=-1, to=2**31-1,
                        textvariable=self.var_gemini_seed, width=16)
        sp_gm_seed.pack(side="left", padx=4)
        ttk.Label(r_gm_seed, text="(-1 = 랜덤)", style="Sub.TLabel").pack(side="left", padx=4)
        Tooltip(sp_gm_seed,
            "생성 시드값입니다.\n\n"
            "  -1  → 실행마다 다른 결과 (랜덤)\n"
            "  고정값 → 동일한 조건에서 재현 가능\n\n"
            "같은 이미지를 여러 번 생성할 때 활용합니다.")

        r_gm_prompt = self._row(self.f_comfyui_gemini, "프롬프트:", 14)
        ent_gm_prompt = ttk.Entry(r_gm_prompt, textvariable=self.var_gemini_prompt, width=46)
        ent_gm_prompt.pack(side="left", padx=4, fill="x", expand=True)
        Tooltip(ent_gm_prompt,
            "Gemini에게 전달할 지시 프롬프트입니다.\n\n"
            "권장:\n"
            "  upscale this. refine details.\n"
            "  preserve text. retain composition.\n\n"
            "텍스처 업스케일에는 구성 보존을 강조하는 문구가 효과적입니다.")

        # 모델 선택
        c2 = self._card(parent, "🎯  모델")

        # 통합 모델 행 (엔진에 따라 values/textvariable만 교체)
        self.f_model_row = ttk.Frame(c2, style="Card.TFrame")
        self.f_model_row.pack(fill="x", pady=2)
        ttk.Label(self.f_model_row, text="모델:", style="Card.TLabel",
                  width=8).pack(side="left")
        self.cb_model = ttk.Combobox(self.f_model_row,
                     textvariable=self.var_esrgan_model,
                     values=list(ESRGAN_MODELS.keys()),
                     state="readonly", width=46)
        self.cb_model.pack(side="left", padx=4)
        self.cb_model.bind("<<ComboboxSelected>>", self._on_model_select)
        Tooltip(self.cb_model,
            "업스케일에 사용할 AI 모델입니다.\n\n"
            "  • RealESRGAN_x4plus (범용)    — 일반 사진/텍스처, 4배 확대\n"
            "  • RealESRGAN_x4plus_anime     — 애니/만화 스타일, 4배 확대\n"
            "  • RealESRNet_x4plus (빠름)    — 속도 우선, 4배 확대\n"
            "  • RealESRGAN_x2plus (x2)      — 2배 확대\n\n"
            "선택한 모델에 따라 배율이 자동으로 설정됩니다.")

        # 세부 설정
        c3 = self._card(parent, "🔧  세부 설정")

        self.f_scale_row = ttk.Frame(c3, style="Card.TFrame")
        self.f_scale_row.pack(fill="x", pady=3)
        ttk.Label(self.f_scale_row, text="업스케일 배율:", style="Card.TLabel",
                  width=14).pack(side="left")
        self.cb_scale = ttk.Combobox(self.f_scale_row, textvariable=self.var_scale,
                     values=["2x", "4x"], state="readonly", width=7)
        self.cb_scale.pack(side="left", padx=4)
        Tooltip(self.cb_scale,
            "업스케일 배율입니다.\n\n"
            "  2x → 원본의 2배 크기로 확대\n"
            "  4x → 원본의 4배 크기로 확대 (권장)\n\n"
            "목표 해상도(기본 4096px)에 도달하면 리사이즈로 마무리됩니다.\n"
            "Real-ESRGAN 모델 선택 시 자동으로 설정됩니다.")
        self.lbl_scale_hint = ttk.Label(self.f_scale_row,
                     text="(모델에서 자동 설정)", style="Sub.TLabel")
        self.lbl_scale_hint.pack(side="left", padx=4)

        r2 = self._row(c3, "GPU ID:", 14)
        sp_gpu = ttk.Spinbox(r2, from_=-1, to=7, textvariable=self.var_gpu_id, width=5)
        sp_gpu.pack(side="left", padx=4)
        Tooltip(sp_gpu,
            "사용할 GPU 번호입니다.\n\n"
            "  0    → 첫 번째 GPU (기본값)\n"
            "  1, 2 → 멀티 GPU 환경에서 특정 GPU 지정\n"
            "  -1   → CPU 강제 사용 (매우 느림)\n\n"
            "GPU가 하나라면 0으로 두면 됩니다.")
        sp_gpu.bind("<ButtonRelease-1>", lambda e: self._refresh_model_display())
        sp_gpu.bind("<KeyRelease>",      lambda e: self._refresh_model_display())
        ttk.Label(r2, text="(-1 = CPU 강제)", style="Card.TLabel").pack(side="left")

        r3 = self._row(c3, "타일 크기:", 14)
        sp_tile = ttk.Spinbox(r3, from_=0, to=2048, increment=64, textvariable=self.var_tile, width=7)
        sp_tile.pack(side="left", padx=4)
        Tooltip(sp_tile,
            "이미지를 분할 처리할 타일 크기(픽셀)입니다.\n\n"
            "  0      → 자동 (전체 이미지 한 번에 처리)\n"
            "  256    → VRAM 4GB 이하 환경 권장\n"
            "  512    → VRAM 8GB 환경 권장\n\n"
            "VRAM 부족 오류 발생 시 256~512로 설정하세요.\n"
            "타일이 작을수록 메모리 사용량이 줄지만 속도가 느려집니다.")
        sp_tile.bind("<ButtonRelease-1>", lambda e: self._refresh_model_display())
        sp_tile.bind("<KeyRelease>",      lambda e: self._refresh_model_display())
        ttk.Label(r3, text="(0=자동  |  VRAM 부족 시 256~512)", style="Card.TLabel").pack(side="left")

        # 세부 설정 변수 변경 시 모델 콤보박스 강제 갱신 (Windows readonly 버그 대응)
        for var in (self.var_gpu_id, self.var_tile, self.var_scale):
            var.trace_add("write", lambda *_: self.after(10, self._refresh_model_display))

        self._refresh_engine_ui()

    # ── 탭 3: RGBA 채널 ───────────────────
    def _tab_rgba(self, parent):
        c = self._card(parent, "🎨  RGBA 채널 분리")

        cb_split = ttk.Checkbutton(c, text="채널 분리 활성화 (원본 TGA와 별도로 각 채널을 그레이스케일 TGA로 저장)",
                        variable=self.var_split, style="TCheckbutton",
                        command=self._refresh_rgba_ui)
        cb_split.pack(anchor="w", pady=(0, 10))
        Tooltip(cb_split,
            "DDS의 RGBA 채널을 각각 별도의 그레이스케일 TGA 파일로 저장합니다.\n\n"
            "PBR 텍스처 활용 예:\n"
            "  R 채널 → Roughness (거칠기)\n"
            "  G 채널 → Metallic (금속성)\n"
            "  B 채널 → AO (주변광 차폐)\n"
            "  A 채널 → Opacity (투명도)\n\n"
            "원본 TGA는 그대로 저장되고 채널별 파일이 추가로 생성됩니다.")

        self.f_rgba_opts = ttk.Frame(c, style="Card.TFrame")
        self.f_rgba_opts.pack(fill="x")

        ttk.Label(self.f_rgba_opts, text="추출할 채널:", style="Card.TLabel").pack(anchor="w")
        ch_row = ttk.Frame(self.f_rgba_opts, style="Card.TFrame")
        ch_row.pack(anchor="w", pady=5)

        self.ch_checks = []
        ch_tooltips = {
            "R  (Red)":   "R 채널 (빨강)\nPBR에서 주로 Roughness(거칠기) 또는 Red 마스크에 사용됩니다.",
            "G  (Green)": "G 채널 (초록)\nPBR에서 주로 Metallic(금속성) 또는 Green 마스크에 사용됩니다.",
            "B  (Blue)":  "B 채널 (파랑)\nPBR에서 주로 Ambient Occlusion(AO) 또는 Blue 마스크에 사용됩니다.",
            "A  (Alpha)": "A 채널 (알파/투명도)\n투명도 마스크 또는 Opacity 맵에 사용됩니다.\nDDS에 알파 채널이 없으면 흰색으로 출력됩니다.",
        }
        for text, var in [("R  (Red)",  self.var_ch_r), ("G  (Green)", self.var_ch_g),
                          ("B  (Blue)", self.var_ch_b), ("A  (Alpha)", self.var_ch_a)]:
            cb = ttk.Checkbutton(ch_row, text=text, variable=var, style="TCheckbutton")
            cb.pack(side="left", padx=12)
            self.ch_checks.append(cb)
            Tooltip(cb, ch_tooltips[text])

        # 설명
        c2 = self._card(parent, "ℹ  출력 예시")
        info = (
            "파일명이 'diffuse.dds' 인 경우 (PNG 선택 시):\n\n"
            "  diffuse.png          ← 원본 RGBA 전체\n"
            "  diffuse_R.png        ← R 채널 (그레이스케일)\n"
            "  diffuse_G.png        ← G 채널 (그레이스케일)\n"
            "  diffuse_B.png        ← B 채널 (그레이스케일)\n"
            "  diffuse_A.png        ← A 채널 (투명도)\n\n"
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
        # 하나의 컨테이너를 side="bottom"으로 고정 — 창 크기 변경과 무관하게 항상 표시
        outer = tk.Frame(self, bg=self.BG)
        outer.pack(side="bottom", fill="x", padx=8, pady=(0, 6))

        # ① 버튼 행 (맨 아래 고정 — 먼저 side="bottom" 배치)
        btn_row = tk.Frame(outer, bg=self.BG)
        btn_row.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="로그 지우기", command=self._clear_log).pack(side="left")
        self.btn_stop = ttk.Button(btn_row, text="⬛  중단", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="right", padx=(4, 0))
        self.btn_start = ttk.Button(btn_row, text="▶  변환 시작",
                                    style="Start.TButton", command=self._start)
        self.btn_start.pack(side="right")

        # ② 진행 바 (버튼 바로 위)
        ctrl = tk.Frame(outer, bg=self.BG)
        ctrl.pack(side="bottom", fill="x", pady=(3, 4))
        self.lbl_status = ttk.Label(ctrl, text="대기 중…", style="TLabel")
        self.lbl_status.pack(anchor="w")
        self.prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(ctrl, variable=self.prog_var, maximum=100).pack(fill="x", pady=2)

        # ③ 로그 패널 (나머지 공간)
        bot = ttk.Frame(outer, style="Card.TFrame")
        bot.pack(fill="x", pady=(4, 0))
        ttk.Label(bot, text="로그", style="Head.TLabel").pack(anchor="w", padx=10, pady=(6, 3))

        lf = ttk.Frame(bot, style="Card.TFrame")
        lf.pack(fill="x", padx=10)
        self.log_text = tk.Text(
            lf, height=6, bg="#11111b", fg=self.FG,
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

    # ── UI 상태 갱신 ──────────────────────
    def _refresh_engine_ui(self):
        engine = self.var_engine.get()
        is_python  = engine == "realesrgan_python"
        is_comfyui = engine == "comfyui"

        # ComfyUI 서버 설정 행
        if is_comfyui:
            self.f_comfyui.pack(fill="x", pady=(8, 0))
            self._refresh_comfyui_ui()
        else:
            self.f_comfyui.pack_forget()

        # 모델 행
        if is_python:
            self.f_model_row.pack(fill="x", pady=2)
            self.cb_model.configure(values=list(ESRGAN_MODELS.keys()),
                                    textvariable=self.var_esrgan_model)
            self.cb_model.set(self.var_esrgan_model.get())
        else:
            self.f_model_row.pack_forget()

        # 업스케일 배율
        if is_comfyui:
            self.f_scale_row.pack_forget()
        else:
            self.f_scale_row.pack(fill="x", pady=3)
            if is_python:
                self.cb_scale.configure(state="disabled")
                self.lbl_scale_hint.pack(side="left", padx=4)
            else:
                self.cb_scale.configure(state="readonly")
                self.lbl_scale_hint.pack_forget()

    def _on_model_select(self, event=None):
        """모델 선택 시 업스케일 배율 자동 맞춤 (Real-ESRGAN Python 전용)"""
        if self.var_engine.get() == "realesrgan_python":
            _, scale = ESRGAN_MODELS.get(self.var_esrgan_model.get(), ("", 4))
            self.var_scale.set(f"{scale}x")

    def _refresh_model_display(self):
        if self.var_engine.get() == "realesrgan_python":
            self.cb_model.set(self.var_esrgan_model.get())

    def _refresh_rgba_ui(self):
        enabled = self.var_split.get()
        for cb in self.ch_checks:
            cb.configure(state="normal" if enabled else "disabled")

    def _refresh_comfyui_mode_ui(self):
        if self.var_comfyui_use_cloud.get():
            self.f_comfyui_local.pack_forget()
            self.f_comfyui_cloud.pack(fill="x")
        else:
            self.f_comfyui_cloud.pack_forget()
            self.f_comfyui_local.pack(fill="x")

    def _toggle_api_key_visibility(self):
        current = self.ent_api_key.cget("show")
        self.ent_api_key.configure(show="" if current == "*" else "*")

    def _refresh_comfyui_ui(self):
        self._refresh_comfyui_mode_ui()
        wf = self.var_comfyui_workflow.get()
        if wf == "controlnet_tile":
            self.f_comfyui_cn.pack(fill="x", pady=(6, 0))
        else:
            self.f_comfyui_cn.pack_forget()
        if wf == "gemini_image":
            self.f_comfyui_gemini.pack(fill="x", pady=(6, 0))
        else:
            self.f_comfyui_gemini.pack_forget()

    # ── 파일 목록 핸들러 ─────────────────
    # 지원 확장자 목록
    SUPPORTED_EXTS = [".dds", ".png", ".tga", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]

    def _add_files(self):
        ext_filter = " ".join(
            f"*{e} *{e.upper()}" for e in self.SUPPORTED_EXTS
        )
        files = filedialog.askopenfilenames(
            title="이미지 파일 선택",
            filetypes=[
                ("지원 이미지", ext_filter),
                ("DDS", "*.dds *.DDS"),
                ("PNG", "*.png *.PNG"),
                ("TGA", "*.tga *.TGA"),
                ("JPEG", "*.jpg *.jpeg *.JPG *.JPEG"),
                ("모든 파일", "*.*"),
            ]
        )
        self._push_files(files)

    def _add_folder(self):
        folder = filedialog.askdirectory(title="폴더 선택")
        if not folder:
            return
        rec = self.var_recursive.get()
        patterns = [
            f"{'**/' if rec else ''}{e}"
            for ext in self.SUPPORTED_EXTS
            for e in (f"*{ext}", f"*{ext.upper()}")
        ]
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
            self.var_output_dir.set(os.path.join(folder, "output"))

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

    # ── 처리 시작 / 중단 ─────────────────
    def _start(self):
        if not self.input_files:
            messagebox.showwarning("경고", "이미지 파일을 추가하세요.")
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
        model_name, esrgan_scale = ESRGAN_MODELS.get(
            self.var_esrgan_model.get(), ("RealESRGAN_x4plus", 4))

        if engine == "realesrgan_python":
            scale_val = esrgan_scale
        else:
            scale_val = int(self.var_scale.get().replace("x", ""))

        try:
            target = int(self.var_target.get())
        except ValueError:
            target = DEFAULT_TARGET

        settings = {
            "engine":         engine,
            "model_name":     model_name,
            "scale":          scale_val,
            "target_size":    target,
            "gpu_id":         self.var_gpu_id.get(),
            "tile_size":      self.var_tile.get(),
            "output_format":  self.var_output_format.get(),
            "output_dir":     out_dir,
            "keep_structure": self.var_keep_structure.get(),
            "input_base":     str(Path(self.input_files[0]).parent),
            "overwrite":      self.var_overwrite.get(),
            "split_channels": bool(channels),
            "channels":       channels,
            "comfyui_use_cloud":   self.var_comfyui_use_cloud.get(),
            "comfyui_api_key":     self.var_comfyui_api_key.get().strip(),
            "comfyui_host":        self.var_comfyui_host.get().strip(),
            "comfyui_port":        int(self.var_comfyui_port.get() or 8188),
            "comfyui_model":       self.var_comfyui_model.get(),
            "comfyui_workflow":    self.var_comfyui_workflow.get(),
            "comfyui_sd_model":    self.var_comfyui_sd_model.get(),
            "comfyui_cn_model":    self.var_comfyui_cn_model.get(),
            "comfyui_cn_strength": self.var_comfyui_cn_strength.get(),
            "comfyui_denoise":     self.var_comfyui_denoise.get(),
            "comfyui_cfg":         self.var_comfyui_cfg.get(),
            "comfyui_steps":       self.var_comfyui_steps.get(),
            "comfyui_sampler":     self.var_comfyui_sampler.get(),
            "comfyui_timeout":     self.var_comfyui_timeout.get(),
            "comfyui_pos_prompt":  self.var_comfyui_pos_prompt.get(),
            "comfyui_neg_prompt":  self.var_comfyui_neg_prompt.get(),
            "gemini_model":        self.var_gemini_model.get(),
            "gemini_prompt":       self.var_gemini_prompt.get(),
            "gemini_resolution":   self.var_gemini_resolution.get(),
            "gemini_seed":         self.var_gemini_seed.get(),
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
    import multiprocessing
    multiprocessing.freeze_support()

    app = App()
    app.mainloop()
