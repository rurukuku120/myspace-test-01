# DDS_TGA_Upscaler 프로젝트

## 파일 명명 규칙
- **항상 PascalCase + 언더스코어** 사용
- 예: `DDS_TGA_Upscaler.py`, `Image_Converter.py`, `Batch_Processor.py`
- 소문자 스네이크케이스(`dds_tga_upscaler.py`) 사용 금지

## 작업 디렉토리
- **기본 경로**: `C:/Users/cukir/Documents/Python/`

## 프로젝트 파일
| 파일 | 설명 |
|------|------|
| `DDS_TGA_Upscaler.py` | 메인 GUI 앱 (현재 버전) |
| `DDS_TGA_Upscaler_v1.1.0.py` | v1.1.0 백업 |
| `DDS_TGA_Upscaler_v1.2.0.py` | v1.2.0 백업 |
| `DDS_TGA_Upscaler_v1.3.0.py` | v1.3.0 백업 |
| `DDS_TGA_Upscaler_v1.4.0.py` | v1.4.0 백업 |
| `Validate_Build.py` | 빌드 검증 스크립트 |
| `DDS_TGA_Upscaler.spec` | PyInstaller 스펙 |

## 버전 관리
- 기능 추가 시 **git 커밋 + 버전 백업 파일** 둘 다 생성
- 백업 파일명: `DDS_TGA_Upscaler_v{버전}.py`
- 현재 버전: `1.4.0`

## DDS_TGA_Upscaler 핵심 사항
- GUI: tkinter + ttk, **Catppuccin Mocha** 다크 테마
- 업스케일 엔진: Real-ESRGAN Python / ComfyUI
- 입력 포맷: DDS, PNG, TGA, JPEG, BMP, WebP, TIFF
- 출력 포맷: PNG 또는 TGA (GUI 선택)
- 4K 타겟: 4096×4096 (텍스처 표준)
- RGBA 채널 분리 기능 포함
- DDS 탐색: `*.dds` + `*.DDS` + 기타 포맷 (중복 제거)
- 체크박스: 커스텀 ✓ 이미지 인디케이터 사용
- 필수 패키지: `pip install Pillow numpy`
- 선택 패키지: `pip install texture2ddecoder imageio` (DDS 고급 포맷 지원)

## DDS 디코딩 폴백 체인
1. Pillow 내장 (DXT1/3/5 기본 지원)
2. texture2ddecoder (BC1~BC7 완전 지원) ← 게임 DDS 권장
3. imageio 폴백

## ComfyUI 엔진 사항
- 로컬 모드: `http://host:port` (기본 `127.0.0.1:8188`)
- 클라우드 모드: `https://cloud.comfy.org` + `X-API-Key` 헤더
- 업로드: `/api/upload/image` (클라우드) / `/upload/image` (로컬)
- 폴링: `/api/history_v2/{id}` (클라우드) / `/history/{id}` (로컬)
- 워크플로우: `upscale` / `controlnet_tile` / `gemini_image` (Nano Banana Pro)
- PNG 중간 변환: 입력 파일 → `_source/{stem}.png` 저장 후 ComfyUI 업로드

## Gemini 워크플로우 (Nano Banana Pro)
- ComfyUI 커스텀 노드: `GeminiImage2Node`
- Gemini API 키는 ComfyUI 서버에서 관리 (앱 불필요)
- 모델: `gemini-3-pro-image-preview` 등
- 워크플로우 JSON: node 2(LoadImage) → node 4(GeminiImage2Node) → node 5(SaveImage)

## 레이아웃 규칙
- 하단 버튼 영역(`▶ 변환 시작`)은 `side="bottom"` 고정 — 창 크기와 무관하게 항상 표시
- `_build_bottom()` 호출은 notebook보다 **먼저** (side="bottom" 우선 배치)
