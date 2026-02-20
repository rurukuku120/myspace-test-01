import os
import csv
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
from threading import Thread

class XMLAttributeAndFileLinkerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XML Attribute + 연관 파일 추출기")
        self.geometry("800x800")
        self.resizable(True, True)
        self.columnconfigure(1, weight=1)
        self.selected_folders = []
        self.create_widgets()

    def create_widgets(self):
        padding = {"padx": 10, "pady": 5}

        # 폴더 선택
        ttk.Label(self, text="XML 폴더 선택 (복수 추가 가능)").grid(row=0, column=0, sticky="w", **padding)
        ttk.Button(self, text="폴더 추가", command=self.add_folder).grid(row=0, column=1, sticky="ew", **padding)
        self.folder_listbox = tk.Listbox(self, height=6, selectmode=tk.BROWSE)
        self.folder_listbox.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10)

        # 저장 위치
        ttk.Label(self, text="CSV/TXT 저장 위치").grid(row=2, column=0, sticky="w", **padding)
        self.save_path_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.save_path_var).grid(row=2, column=1, sticky="ew", **padding)
        ttk.Button(self, text="저장", command=self.select_save_path).grid(row=2, column=2, **padding)

        # Attribute 입력
        ttk.Label(self, text="추출할 Attribute 이름들 (예: name,effect_name,parent)").grid(row=3, column=0, sticky="w", **padding)
        self.attribute_name_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.attribute_name_var).grid(row=3, column=1, columnspan=2, sticky="ew", **padding)

        # 확장자
        ttk.Label(self, text="연관 확장자들 (예: .pmg,.ani)").grid(row=4, column=0, sticky="w", **padding)
        self.extensions_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.extensions_var).grid(row=4, column=1, columnspan=2, sticky="ew", **padding)

        # 제외 단어
        ttk.Label(self, text="제외할 단어 (예: _framework)").grid(row=5, column=0, sticky="w", **padding)
        self.exclude_words_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.exclude_words_var).grid(row=5, column=1, columnspan=2, sticky="ew", **padding)

        # 옵션들
        self.include_subdirs_var = tk.BooleanVar()
        ttk.Checkbutton(self, text="하위 폴더 포함", variable=self.include_subdirs_var).grid(row=6, column=1, sticky="w", **padding)

        self.full_path_display_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="파일 경로 전체 표시 (해제 시 파일명만)", variable=self.full_path_display_var).grid(row=7, column=1, sticky="w", **padding)

        # 진행 바
        self.progress = ttk.Progressbar(self, orient="horizontal", length=420, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, padx=20, pady=10)

        # 저장 형식
        ttk.Label(self, text="저장 형식 선택").grid(row=9, column=0, sticky="w", padx=10, pady=5)
        self.file_format_var = tk.StringVar(value="csv")
        ttk.Radiobutton(self, text="CSV", variable=self.file_format_var, value="csv").grid(row=9, column=1, sticky="w")
        ttk.Radiobutton(self, text="TXT", variable=self.file_format_var, value="txt").grid(row=9, column=2, sticky="w")

        # 행마다 분리 옵션 (기본 OFF)
        self.split_rows_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Attribute 값을 각 행마다 분리", variable=self.split_rows_var).grid(
            row=10, column=1, sticky="w", padx=10, pady=5
        )

        # 실행 버튼
        ttk.Button(self, text="추출 실행", command=self.start_extraction_thread).grid(row=11, column=1, pady=10)

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder and folder not in self.selected_folders:
            self.selected_folders.append(folder)
            self.folder_listbox.insert(tk.END, folder)

    def select_save_path(self):
        selected_format = self.file_format_var.get()
        default_ext = ".csv" if selected_format == "csv" else ".txt"
        filetypes = [("CSV 파일", "*.csv")] if selected_format == "csv" else [("텍스트 파일", "*.txt")]
        file_path = filedialog.asksaveasfilename(
            defaultextension=default_ext, filetypes=filetypes, title="파일 저장 위치 선택"
        )
        if file_path:
            self.save_path_var.set(file_path)

    def clean_filename(self, filename, exclude_words):
        name, _ = os.path.splitext(filename)
        for word in exclude_words:
            name = name.replace(word, "")
        return name

    # ─────────────────────────────────────────────
    #  인코딩/선언 불일치 및 잘못된 속성 보정 (전처리)
    # ─────────────────────────────────────────────
    def detect_declared_encoding(self, raw: bytes):
        """XML 선언에서 encoding 값을 찾아 소문자로 반환 (없으면 None)."""
        try:
            head = raw[:4096].decode("ascii", errors="ignore")
            m = re.search(r'encoding=["\']([A-Za-z0-9_\-]+)["\']', head, flags=re.I)
            return m.group(1).lower() if m else None
        except Exception:
            return None

    def guess_effective_encoding(self, raw: bytes, declared: str | None):
        """BOM/널바이트 패턴을 바탕으로 실제 인코딩 추정.
        - BOM이 있으면 utf-16-le/be
        - 선언이 utf-16인데 널바이트가 드물면 실제 UTF-8로 간주
        - 그 외는 선언 또는 UTF-8
        """
        if raw.startswith(b"\xff\xfe"):
            return "utf-16-le"
        if raw.startswith(b"\xfe\xff"):
            return "utf-16-be"

        decl = (declared or "").lower()
        # 널바이트 비율 체크 (앞 2KB)
        sample = raw[:2048]
        nul_ratio = sample.count(b"\x00") / max(1, len(sample))

        if decl in ("utf-16", "utf16", "utf-16le", "utf-16be"):
            # 진짜 UTF-16이면 nul이 많이 보임(대략 10% 이상)
            if nul_ratio > 0.05:
                # LE/BE 패턴 추정
                if len(sample) > 1 and sample[1] == 0x00:
                    return "utf-16-le"
                elif len(sample) > 0 and sample[0] == 0x00:
                    return "utf-16-be"
                else:
                    return "utf-16-le"
            # 선언만 UTF-16이고 실제는 UTF-8/ASCII
            return "utf-8"

        # 선언이 없거나 UTF-8 류
        return "utf-8"

    def normalize_xml_declaration(self, text: str) -> str:
        """XML 선언의 encoding 속성을 제거/정규화 (content와 선언 불일치로 인한 오류 방지)."""
        def repl(m):
            attrs = m.group(1)
            # encoding="..." 제거
            attrs = re.sub(r'\s+encoding=["\'][^"\']*["\']', '', attrs, flags=re.I)
            # 공백 정리
            attrs = re.sub(r'\s+', ' ', attrs).strip()
            return f'<?xml {attrs}?>'
        return re.sub(r'<\?xml\s+([^?]+?)\s*\?>', repl, text, flags=re.I)

    def cleanup_xml_text(self, text: str) -> str:
        """잘못된 속성 붙임/쓰레기 문자 보정 + 선언 정규화."""
        # 선언 라인 정규화 (encoding 제거)
        text = self.normalize_xml_declaration(text)

        # 따옴표 뒤에 붙은 쓰레기 알파벳 제거:  align="framework"a  -> align="framework"
        text = re.sub(r'(")[A-Za-z](?=\s|/|>)', r'"', text)

        # 속성 따옴표 뒤에 바로 속성명이 붙은 경우 공백 추가:
        # rot_angle="0"sizerate="2" -> rot_angle="0" sizerate="2"
        text = re.sub(r'(?<=")(?=[A-Za-z_])', ' ', text)

        # 선언/태그 앞뒤 공백·널 제거 + 첫 '<' 전부 제거
        text = text.lstrip("\ufeff\000 \t\r\n")
        i = text.find("<")
        if i > 0:
            text = text[i:]
        return text

    def safe_decode_xml(self, raw: bytes):
        """실제 인코딩을 추정하여 디코드하고, 선언/속성 전처리까지 수행."""
        declared = self.detect_declared_encoding(raw)
        effective = self.guess_effective_encoding(raw, declared)
        txt = raw.decode(effective, errors="replace")
        return self.cleanup_xml_text(txt)

    # ─────────────────────────────────────────────
    #  기타 유틸
    # ─────────────────────────────────────────────
    def find_matching_files(self, root_dir, keyword, extension, full_path=True):
        matched = []
        for dirpath, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(extension) and keyword in file:
                    matched.append(os.path.join(dirpath, file) if full_path else file)
        return matched

    def start_extraction_thread(self):
        Thread(target=self.extract_info).start()

    # ─────────────────────────────────────────────
    #  메인 로직
    # ─────────────────────────────────────────────
    def extract_info(self):
        folders = self.selected_folders
        save_path = self.save_path_var.get()
        attr_names = [a.strip() for a in self.attribute_name_var.get().split(",") if a.strip()]
        extensions = [e.strip() for e in self.extensions_var.get().split(",") if e.strip()]
        exclude_words = [w.strip() for w in self.exclude_words_var.get().split(",") if w.strip()]
        include_subdirs = self.include_subdirs_var.get()
        full_path_display = self.full_path_display_var.get()
        selected_format = self.file_format_var.get()
        split_rows = self.split_rows_var.get()

        if not folders or not save_path or not attr_names:
            messagebox.showerror("입력 오류", "필수 항목을 모두 입력해주세요.")
            return

        result = []
        all_xml_files = []

        # XML만 수집
        for folder in folders:
            for dirpath, _, filenames in os.walk(folder):
                if not include_subdirs and dirpath != folder:
                    continue
                for file in filenames:
                    if file.lower().endswith(".xml"):
                        all_xml_files.append((dirpath, file))

        total_files = len(all_xml_files)
        self.progress["maximum"] = total_files
        self.progress["value"] = 0

        for count, (dirpath, file) in enumerate(all_xml_files, 1):
            full_path = os.path.join(dirpath, file)
            cleaned_name = self.clean_filename(file, exclude_words)
            keyword = os.path.splitext(cleaned_name)[0]

            base_row = {ext: "" for ext in extensions}
            base_row["XML 파일명"] = file

            try:
                with open(full_path, "rb") as f:
                    raw = f.read()

                decoded = self.safe_decode_xml(raw)  # 인코딩/선언/속성 보정 완료 텍스트
                root = ET.fromstring(decoded)

                found = {attr: [] for attr in attr_names}
                for elem in root.iter():
                    for attr in attr_names:
                        if attr in elem.attrib:
                            found[attr].append(elem.attrib[attr])

                if split_rows:
                    max_len = max((len(found[attr]) for attr in attr_names), default=1)
                    for i in range(max_len):
                        row = base_row.copy()
                        for attr in attr_names:
                            row[attr] = found[attr][i] if i < len(found[attr]) else ""
                        for ext in extensions:
                            matches = self.find_matching_files(dirpath, keyword, ext, full_path=full_path_display)
                            row[ext] = "\n".join(matches) if matches else ""
                        result.append(row)
                else:
                    row = base_row.copy()
                    for attr in attr_names:
                        row[attr] = "\n".join(found[attr]) if found[attr] else ""
                    for ext in extensions:
                        matches = self.find_matching_files(dirpath, keyword, ext, full_path=full_path_display)
                        row[ext] = "\n".join(matches) if matches else ""
                    result.append(row)

            except Exception as e:
                row = base_row.copy()
                for attr in attr_names:
                    row[attr] = f"오류: {str(e)}"
                for ext in extensions:
                    row[ext] = ""
                result.append(row)
                print(f"[오류] {file}: {e}")

            self.progress["value"] = count
            self.update_idletasks()

        if result:
            fieldnames = extensions + ["XML 파일명"] + attr_names
            try:
                with open(save_path, mode="w", newline="", encoding="utf-8-sig") as f:
                    if selected_format == "csv":
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(result)
                    else:
                        f.write("\t".join(fieldnames) + "\n")
                        for row in result:
                            f.write("\t".join([row.get(field, "") for field in fieldnames]) + "\n")
                messagebox.showinfo("완료", f"{selected_format.upper()} 저장 완료:\n{save_path}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 중 오류 발생:\n{e}")
        else:
            messagebox.showinfo("결과 없음", "조건에 맞는 XML 파일이 없습니다.")

if __name__ == "__main__":
    app = XMLAttributeAndFileLinkerApp()
    app.mainloop()
