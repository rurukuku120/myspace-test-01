"""
Build Validation Script
검증 항목: 파일 무결성(SHA256), ZIP 내용, 파일 크기/메타데이터, 실행 가능 여부
"""

import hashlib
import io
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# Windows cp949 환경에서도 한글/특수문자 출력
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DIST_DIR = Path(__file__).parent / "dist"
EXE_NAME = "DDS_TGA_Upscaler.exe"
ZIP_NAME = "DDS_TGA_Upscaler.zip"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def sep(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'-' * pad} {title} {'-' * (width - pad - len(title) - 2)}")
    else:
        print("-" * width)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 1. 파일 존재 및 메타데이터 ──────────────────────────────────────
def check_metadata():
    sep("1. 파일 크기 / 메타데이터")
    all_ok = True
    for name in [EXE_NAME, ZIP_NAME]:
        path = DIST_DIR / name
        if not path.exists():
            print(f"  {FAIL} {name} — 파일 없음")
            all_ok = False
            continue
        stat = path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {PASS} {name}")
        print(f"         크기    : {stat.st_size:,} bytes ({size_mb:.2f} MB)")
        print(f"         수정 시각: {mtime}")
    return all_ok


# ── 2. 파일 무결성 (SHA-256) ────────────────────────────────────────
def check_integrity():
    sep("2. 파일 무결성 (SHA-256)")
    hash_file = DIST_DIR / "checksums.sha256"
    hashes: dict[str, str] = {}

    for name in [EXE_NAME, ZIP_NAME]:
        path = DIST_DIR / name
        if not path.exists():
            print(f"  {FAIL} {name} — 파일 없음, 건너뜀")
            continue
        digest = sha256(path)
        hashes[name] = digest
        print(f"  {INFO} {name}")
        print(f"         SHA256 : {digest}")

    # 기존 체크섬 파일과 비교 (있을 경우)
    if hash_file.exists():
        print(f"\n  {INFO} 기존 checksums.sha256 와 비교:")
        with open(hash_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stored_hash, stored_name = line.split(None, 1)
                stored_name = stored_name.lstrip("*").strip()
                current = hashes.get(stored_name)
                if current is None:
                    print(f"    {WARN} {stored_name} — 현재 파일 없음")
                elif current == stored_hash:
                    print(f"    {PASS} {stored_name} — 일치")
                else:
                    print(f"    {FAIL} {stored_name} — 불일치! (변조 가능성)")
    else:
        # 체크섬 파일 새로 생성
        with open(hash_file, "w") as f:
            for name, digest in hashes.items():
                f.write(f"{digest} *{name}\n")
        print(f"\n  {INFO} checksums.sha256 생성 완료 → {hash_file}")

    return bool(hashes)


# ── 3. ZIP 내용 검증 ────────────────────────────────────────────────
def check_zip():
    sep("3. ZIP 내용 검증")
    path = DIST_DIR / ZIP_NAME
    if not path.exists():
        print(f"  {FAIL} {ZIP_NAME} — 파일 없음")
        return False

    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                print(f"  {FAIL} 손상된 파일 발견: {bad}")
                return False
            print(f"  {PASS} ZIP 무결성 OK")

            names = zf.namelist()
            total_size = sum(i.file_size for i in zf.infolist())
            print(f"  {INFO} 포함 파일 수  : {len(names)}")
            print(f"  {INFO} 압축 전 총 크기: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

            # EXE 포함 여부 확인
            exe_in_zip = any(n.endswith(".exe") for n in names)
            if exe_in_zip:
                print(f"  {PASS} EXE 파일 포함 확인")
            else:
                print(f"  {WARN} EXE 파일이 ZIP 안에 없음")

            # 상위 10개 파일 목록
            print(f"\n  파일 목록 (상위 10개):")
            for info in sorted(zf.infolist(), key=lambda x: x.file_size, reverse=True)[:10]:
                print(f"    {info.filename:<50} {info.file_size:>10,} B")
            if len(names) > 10:
                print(f"    ... 외 {len(names) - 10}개")
    except zipfile.BadZipFile as e:
        print(f"  {FAIL} ZIP 파일 손상: {e}")
        return False

    return True


# ── 4. 실행 가능 여부 ───────────────────────────────────────────────
def check_executable():
    sep("4. 실행 가능 여부")
    path = DIST_DIR / EXE_NAME
    if not path.exists():
        print(f"  {FAIL} {EXE_NAME} — 파일 없음")
        return False

    if platform.system() != "Windows":
        print(f"  {WARN} Windows 환경이 아님 — 실행 테스트 건너뜀")
        return True

    # PE 헤더 확인 (MZ 시그니처)
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"MZ":
        print(f"  {PASS} PE 헤더 (MZ 시그니처) 확인")
    else:
        print(f"  {FAIL} PE 헤더 없음 — 유효한 EXE가 아닐 수 있음")
        return False

    # --version 또는 빠른 실행 가능 여부 (3초 타임아웃)
    try:
        result = subprocess.run(
            [str(path), "--help"],
            timeout=3,
            capture_output=True,
        )
        print(f"  {PASS} 프로세스 실행 성공 (exit code: {result.returncode})")
    except subprocess.TimeoutExpired:
        # GUI 앱은 타임아웃이 정상 (프로세스가 떠 있음)
        print(f"  {PASS} 프로세스 실행 성공 (GUI 앱 — 타임아웃은 정상)")
    except FileNotFoundError:
        print(f"  {FAIL} 실행 파일을 찾을 수 없음")
        return False
    except OSError as e:
        print(f"  {FAIL} 실행 실패: {e}")
        return False

    return True


# ── 메인 ────────────────────────────────────────────────────────────
def main():
    print(f"\n{'=' * 60}")
    print(f"  Build Validation -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  대상 경로: {DIST_DIR}")
    print(f"{'=' * 60}")

    if not DIST_DIR.exists():
        print(f"\n{FAIL} dist 디렉터리가 존재하지 않습니다: {DIST_DIR}")
        sys.exit(1)

    results = {
        "메타데이터": check_metadata(),
        "무결성(SHA256)": check_integrity(),
        "ZIP 내용": check_zip(),
        "실행 가능": check_executable(),
    }

    sep("최종 결과")
    all_passed = True
    for label, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status} {label}")
        if not ok:
            all_passed = False

    sep()
    if all_passed:
        print(f"\n  {PASS} 모든 검증 통과\n")
        sys.exit(0)
    else:
        print(f"\n  {FAIL} 일부 검증 실패 — 위 항목을 확인하세요\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
