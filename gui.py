#!/usr/bin/env python3
"""A small desktop interface for decoding Morse-code audio."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from decode_audio import decode_audio


BACKGROUND = "#FFF4D7"
SURFACE = "#252727"
TERRACOTTA = "#9A3B3B"
CLAY = "#C08261"
GOLD = "#E2C799"
PALE_OLIVE = "#E2ECBE"
MUTED_TEXT = "#808080"
LOGO_PATH = Path(__file__).resolve().parent / "logo.jpeg"


class DecoderApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=24, style="App.TFrame")
        self.master = master
        self.audio_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.status = tk.StringVar(value="Choose an audio file and an output text file.")
        self.logo_image = ImageTk.PhotoImage(Image.open(LOGO_PATH))
        self._build()

    def _build(self):
        self.grid(sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        ttk.Label(self, image=self.logo_image, style="Logo.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="s", pady=(0, 20)
        )

        ttk.Label(self, text="AUDIO FILE", style="Field.TLabel").grid(
            row=1, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Entry(self, textvariable=self.audio_path, width=52, style="Path.TEntry").grid(
            row=1, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(self, text="Browse", command=self.choose_audio, style="Browse.TButton").grid(
            row=1, column=2, pady=(0, 8)
        )

        ttk.Label(self, text="TEXT FILE", style="Field.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 16)
        )
        ttk.Entry(self, textvariable=self.output_path, width=52, style="Path.TEntry").grid(
            row=2, column=1, sticky="ew", padx=(12, 8), pady=(0, 16)
        )
        ttk.Button(self, text="Browse", command=self.choose_output, style="Browse.TButton").grid(
            row=2, column=2, pady=(0, 16)
        )

        self.decode_button = ttk.Button(
            self, text="Decode", command=self.decode, style="Decode.TButton"
        )
        self.decode_button.grid(row=3, column=0, columnspan=3, pady=(0, 16))
        ttk.Label(self, textvariable=self.status, wraplength=460, style="Status.TLabel").grid(
            row=5, column=0, columnspan=3, sticky="s", pady=(16, 0)
        )

    def choose_audio(self):
        filename = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if filename:
            self.audio_path.set(filename)
            if not self.output_path.get():
                self.output_path.set(str(Path(filename).with_suffix(".txt")))

    def choose_output(self):
        filename = filedialog.asksaveasfilename(
            title="Choose output text file",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filename:
            self.output_path.set(filename)

    def decode(self):
        audio_file = Path(self.audio_path.get()).expanduser()
        output_file = Path(self.output_path.get()).expanduser()
        if not self.audio_path.get() or not audio_file.is_file():
            messagebox.showerror("Audio file required", "Choose an existing WAV audio file.")
            return
        if not self.output_path.get():
            messagebox.showerror("Text file required", "Choose where to save the decoded text.")
            return

        self.decode_button.state(["disabled"])
        self.status.set("Decoding…")
        self.master.update_idletasks()
        try:
            decoded_text = decode_audio(audio_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(decoded_text + "\n", encoding="utf-8")
        except Exception as error:
            self.status.set("Decode failed.")
            messagebox.showerror("Decode failed", str(error))
        else:
            self.status.set("Decoded text saved to {}.".format(output_file))
            messagebox.showinfo("Decode complete", "Decoded text saved to:\n{}".format(output_file))
        finally:
            self.decode_button.state(["!disabled"])


def main():
    root = tk.Tk()
    root.title("Morse Audio Decoder")
    root.configure(background=BACKGROUND)
    root.minsize(620, 320)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("App.TFrame", background=BACKGROUND)
    style.configure("Logo.TLabel", background=BACKGROUND)
    style.configure(
        "Field.TLabel", background=BACKGROUND, foreground=TERRACOTTA, font=("TkDefaultFont", 9, "bold")
    )
    style.configure(
        "Status.TLabel", background=BACKGROUND, foreground=MUTED_TEXT, font=("TkDefaultFont", 9)
    )
    style.configure(
        "Path.TEntry",
        fieldbackground=BACKGROUND,
        foreground="#000000",
        insertcolor="#000000",
        bordercolor=CLAY,
        lightcolor=CLAY,
        darkcolor=CLAY,
        padding=7,
    )
    style.map("Path.TEntry", bordercolor=[("focus", PALE_OLIVE)])
    style.configure(
        "Browse.TButton",
        background=CLAY,
        foreground=BACKGROUND,
        borderwidth=0,
        padding=(12, 7),
        font=("TkDefaultFont", 9, "bold"),
    )
    style.map(
        "Browse.TButton",
        background=[("active", GOLD), ("disabled", SURFACE)],
        foreground=[("disabled", MUTED_TEXT)],
    )
    style.configure(
        "Decode.TButton",
        background=TERRACOTTA,
        foreground=PALE_OLIVE,
        borderwidth=0,
        padding=(26, 9),
        font=("TkDefaultFont", 10, "bold"),
    )
    style.map(
        "Decode.TButton",
        background=[("active", CLAY), ("disabled", SURFACE)],
        foreground=[("disabled", MUTED_TEXT)],
    )
    DecoderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
