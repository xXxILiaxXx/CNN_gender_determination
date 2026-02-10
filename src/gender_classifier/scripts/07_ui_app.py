from __future__ import annotations

"""Минимальный UI для проверки, что Tkinter умеет открывать и показывать изображение.

Запуск:
  python -m src.gender_classifier.scripts.07_ui_app

Зависимости:
  pip install pillow

Примечание (macOS): если диалог выбора файлов не видит Desktop —
это права доступа (Privacy & Security). Пока можно выбрать файл из папки проекта.
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Gender Classifier (UI test)")
        self.geometry("1100x700")
        self.minsize(900, 600)

        # Dark theme
        self.bg = "#2b2b2b"
        self.panel = "#3a3a3a"
        self.fg = "#e6e6e6"
        self.btn = "#5a5a5a"

        self.configure(bg=self.bg)

        # Keep references (важно, иначе картинка пропадёт)
        self._pil_original: Image.Image | None = None
        self._tk_img: ImageTk.PhotoImage | None = None

        self._build_ui()

        # Перерисовка картинки при ресайзе окна
        self._resize_after_id = None
        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        # Верхняя зона: Canvas под изображение
        self.main = tk.Frame(self, bg=self.panel)
        self.main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.canvas = tk.Canvas(self.main, bg=self.panel, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Подпись-заглушка
        self.placeholder = self.canvas.create_text(
            10,
            10,
            anchor="nw",
            text="Нажми 'Открыть фото' слева снизу",
            fill=self.fg,
            font=("Arial", 14),
        )

        # Нижняя панель
        self.bottom = tk.Frame(self, bg=self.bg)
        self.bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 12))

        self.left = tk.Frame(self.bottom, bg=self.bg)
        self.left.pack(side=tk.LEFT, anchor="sw")

        self.btn_open = tk.Button(
            self.left,
            text="Открыть фото",
            command=self.on_open,
            bg=self.btn,
            fg=self.fg,
            relief=tk.FLAT,
            padx=14,
            pady=8,
        )
        self.btn_open.pack(anchor="w")

        self.path_var = tk.StringVar(value="Файл: —")
        self.path_lbl = tk.Label(self.bottom, textvariable=self.path_var, bg=self.bg, fg=self.fg)
        self.path_lbl.pack(side=tk.RIGHT, anchor="se")

    def on_open(self):
        # Важно: initialdir = HOME (так надёжнее, чем Desktop)
        initialdir = str(Path.home())

        path = filedialog.askopenfilename(
            title="Выбрать изображение",
            initialdir=initialdir,
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        p = Path(path)
        if not p.exists():
            messagebox.showerror("Ошибка", f"Файл не найден:\n{p}")
            return

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{e}")
            return

        self._pil_original = img
        self.path_var.set(f"Файл: {p}")
        self._render_image()

    def _on_resize(self, _evt=None):
        # debounce (чтобы не лагало при ресайзе)
        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(80, self._render_image)

    def _render_image(self):
        self._resize_after_id = None
        if self._pil_original is None:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        img = self._pil_original.copy()
        img.thumbnail((cw - 20, ch - 20), Image.Resampling.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(img)

        self.canvas.delete("IMG")
        self.canvas.itemconfigure(self.placeholder, text="")

        # Центруем
        x = cw // 2
        y = ch // 2
        self.canvas.create_image(x, y, image=self._tk_img, anchor="center", tags="IMG")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
