"""
Floating GUI for Voice Desktop Automation.
A sleek, always-on-top, draggable pill-shaped widget with:
  - Animated mic button (press & hold to speak)
  - Minimal text input (type command + Enter)
  - Pulsing status glow

Usage:
    python gui.py
"""

import sys
import math
import threading
import tkinter as tk
from tkinter import font as tkfont

from stt import SpeechToText
from intent_extractor import IntentExtractor
from main import ToolExecutor


# ── Palette ─────────────────────────────────────────────────────────

BG           = "#0d0f18"     # Near-black
SURFACE      = "#161829"     # Card surface
SURFACE_ALT  = "#1c1f33"     # Input bg
BORDER       = "#2a2d44"     # Subtle border
ACCENT       = "#6c5ce7"     # Purple accent
ACCENT_GLOW  = "#a29bfe"     # Lighter glow
CYAN         = "#00cec9"     # Cyan for success
RED          = "#ff6b6b"     # Recording / error
AMBER        = "#fdcb6e"     # Processing
TEXT         = "#e2e8f0"     # Primary text
TEXT_DIM     = "#4a5068"     # Placeholder / muted
WHITE        = "#ffffff"


# ── Helpers ─────────────────────────────────────────────────────────

def _round_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Draw a rounded rectangle on a canvas."""
    points = [
        x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r,
        x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
        x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r,
        x1, y1+r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kw)


class FloatingWidget:
    """Futuristic minimal floating automation widget."""

    WIN_W = 340
    WIN_H = 140

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VA")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.93)
        self.root.configure(bg=BG)

        # Position: bottom-right
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = sw - self.WIN_W - 24
        y = sh - self.WIN_H - 50
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}+{x}+{y}")

        # State
        self._drag_x = 0
        self._drag_y = 0
        self._busy = False
        self._recording = False
        self._pulse_angle = 0.0
        self._glow_color = ACCENT

        # Build everything
        self._build_ui()

        # Init pipeline in background
        self.extractor = None
        self.executor = None
        self.stt = None
        self._set_status("initializing", AMBER)
        threading.Thread(target=self._init_pipeline, daemon=True).start()

        # Start animations
        self._animate_pulse()

    # ── UI Construction ─────────────────────────────────────────────

    def _build_ui(self):
        # Outer canvas (draws the pill-shaped background)
        self.canvas = tk.Canvas(
            self.root, width=self.WIN_W, height=self.WIN_H,
            bg=BG, highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Draw main card shape
        _round_rect(self.canvas, 2, 2, self.WIN_W-2, self.WIN_H-2, 18,
                     fill=SURFACE, outline=BORDER, width=1)

        # Subtle top accent line (glow bar)
        self._glow_line = self.canvas.create_line(
            20, 3, self.WIN_W-20, 3, fill=ACCENT, width=2,
        )

        # ── Top bar: title + close ──────────────────────────────────
        title_f = tkfont.Font(family="Segoe UI", size=8, weight="bold")

        self.title_lbl = tk.Label(
            self.root, text="◆  VOICE AUTO", bg=SURFACE, fg=TEXT_DIM,
            font=title_f, cursor="fleur",
        )
        self.canvas.create_window(16, 16, window=self.title_lbl, anchor="nw")
        self.title_lbl.bind("<Button-1>", self._start_drag)
        self.title_lbl.bind("<B1-Motion>", self._on_drag)

        close_f = tkfont.Font(family="Segoe UI", size=9)
        self.close_btn = tk.Label(
            self.root, text="×", bg=SURFACE, fg=TEXT_DIM,
            font=close_f, cursor="hand2", padx=4,
        )
        self.canvas.create_window(self.WIN_W - 16, 16, window=self.close_btn, anchor="ne")
        self.close_btn.bind("<Button-1>", lambda e: self.root.destroy())
        self.close_btn.bind("<Enter>", lambda e: self.close_btn.config(fg=RED))
        self.close_btn.bind("<Leave>", lambda e: self.close_btn.config(fg=TEXT_DIM))

        # Make canvas itself draggable on empty space
        self.canvas.bind("<Button-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # ── Mic button (circle on the left) ─────────────────────────
        mic_cx, mic_cy, mic_r = 40, 75, 22

        # Outer glow ring (animated)
        self._mic_glow = self.canvas.create_oval(
            mic_cx - mic_r - 4, mic_cy - mic_r - 4,
            mic_cx + mic_r + 4, mic_cy + mic_r + 4,
            outline=ACCENT, width=2,
        )
        # Main circle
        self._mic_circle = self.canvas.create_oval(
            mic_cx - mic_r, mic_cy - mic_r,
            mic_cx + mic_r, mic_cy + mic_r,
            fill=ACCENT, outline="",
        )
        # Mic icon text
        mic_icon_f = tkfont.Font(family="Segoe UI Emoji", size=13)
        self._mic_text = self.canvas.create_text(
            mic_cx, mic_cy, text="🎤", font=mic_icon_f,
        )

        # Bind mic area clicks
        for item in (self._mic_circle, self._mic_text, self._mic_glow):
            self.canvas.tag_bind(item, "<ButtonPress-1>", self._on_mic_press)
            self.canvas.tag_bind(item, "<ButtonRelease-1>", self._on_mic_release)

        # ── Input field ─────────────────────────────────────────────
        input_f = tkfont.Font(family="Segoe UI", size=10)
        self.text_entry = tk.Entry(
            self.root, bg=SURFACE_ALT, fg=TEXT, insertbackground=ACCENT_GLOW,
            font=input_f, relief="flat", highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT,
            borderwidth=0,
        )
        self.canvas.create_window(
            74, 56, window=self.text_entry, anchor="nw",
            width=self.WIN_W - 94, height=34,
        )
        self.text_entry.insert(0, "Ask me anything...")
        self.text_entry.configure(fg=TEXT_DIM)
        self.text_entry.bind("<FocusIn>", self._on_entry_focus)
        self.text_entry.bind("<FocusOut>", self._on_entry_unfocus)
        self.text_entry.bind("<Return>", self._on_text_submit)

        # ── Status line ─────────────────────────────────────────────
        status_f = tkfont.Font(family="Consolas", size=8)
        self._status_dot = self.canvas.create_oval(
            18, 112, 24, 118, fill=CYAN, outline="",
        )
        self._status_text = self.canvas.create_text(
            30, 115, text="ready", anchor="w",
            fill=TEXT_DIM, font=status_f,
        )

        # ── Step counter (right side of status) ─────────────────────
        steps_f = tkfont.Font(family="Consolas", size=7)
        self._steps_text = self.canvas.create_text(
            self.WIN_W - 18, 115, text="", anchor="e",
            fill=TEXT_DIM, font=steps_f,
        )

    # ── Animations ──────────────────────────────────────────────────

    def _animate_pulse(self):
        """Subtle pulsing glow on the mic ring."""
        self._pulse_angle += 0.08
        # Oscillate brightness via sine wave
        t = (math.sin(self._pulse_angle) + 1) / 2   # 0..1

        if self._recording:
            self._pulse_angle += 0.12  # faster
            r, g, b = 255, 107, 107    # RED
            dim = 0.3
        elif self._busy:
            r, g, b = 253, 203, 110    # AMBER
            dim = 0.4
        else:
            r, g, b = 108, 92, 231     # ACCENT purple
            dim = 0.2

        brightness = dim + (1 - dim) * t
        color = f"#{int(r*brightness):02x}{int(g*brightness):02x}{int(b*brightness):02x}"
        self.canvas.itemconfig(self._mic_glow, outline=color)

        # Also pulse the top glow line
        self.canvas.itemconfig(self._glow_line, fill=color)

        self.root.after(40, self._animate_pulse)

    # ── Drag ────────────────────────────────────────────────────────

    def _start_drag(self, event):
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _on_drag(self, event):
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    # ── Entry placeholder ───────────────────────────────────────────

    def _on_entry_focus(self, event):
        if self.text_entry.get() == "Ask me anything...":
            self.text_entry.delete(0, "end")
            self.text_entry.configure(fg=TEXT)

    def _on_entry_unfocus(self, event):
        if not self.text_entry.get().strip():
            self.text_entry.insert(0, "Ask me anything...")
            self.text_entry.configure(fg=TEXT_DIM)

    # ── Pipeline init ───────────────────────────────────────────────

    def _init_pipeline(self):
        try:
            self.extractor = IntentExtractor()
            self.executor = ToolExecutor()
            self.stt = SpeechToText()
            self._set_status("ready", CYAN)
        except Exception as e:
            self._set_status(f"error: {e}", RED)

    # ── Status (thread-safe) ────────────────────────────────────────

    def _set_status(self, text: str, color: str = TEXT_DIM):
        def _update():
            self.canvas.itemconfig(self._status_text, text=text, fill=color)
            self.canvas.itemconfig(self._status_dot, fill=color)
        self.root.after(0, _update)

    def _set_steps(self, text: str):
        self.root.after(0, lambda: self.canvas.itemconfig(self._steps_text, text=text))

    # ── Text submit ─────────────────────────────────────────────────

    def _on_text_submit(self, event):
        text = self.text_entry.get().strip()
        if not text or text == "Ask me anything..." or self._busy:
            return
        self.text_entry.delete(0, "end")
        threading.Thread(target=self._run_command, args=(text,), daemon=True).start()

    # ── Mic ─────────────────────────────────────────────────────────

    def _on_mic_press(self, event):
        if self._busy or self.stt is None:
            return
        self._recording = True
        self.canvas.itemconfig(self._mic_circle, fill=RED)
        self._set_status("listening...", RED)
        self._mic_thread = threading.Thread(target=self._record_voice, daemon=True)
        self._mic_thread.start()

    def _on_mic_release(self, event):
        if not self._recording:
            return
        self._recording = False
        self.canvas.itemconfig(self._mic_circle, fill=ACCENT)
        if self.stt:
            self.stt.stop()

    def _record_voice(self):
        try:
            text = self.stt.listen_once()
            if text and text.strip():
                self._set_status(f"\"{text}\"", ACCENT_GLOW)
                self.root.after(0, lambda: self._show_transcription(text))
                self._run_command(text)
            else:
                self._set_status("no speech detected", TEXT_DIM)
        except Exception as e:
            self._set_status("mic error", RED)
            print(f"[GUI] Mic error: {e}")

    def _show_transcription(self, text: str):
        self.text_entry.delete(0, "end")
        self.text_entry.configure(fg=ACCENT_GLOW)
        self.text_entry.insert(0, text)

    # ── Command execution ───────────────────────────────────────────

    def _run_command(self, text: str):
        if self._busy:
            return
        self._busy = True

        try:
            self._set_status("scanning screen...", AMBER)
            screen_context = self.executor.get_screen_context()

            self._set_status("parsing intent...", AMBER)
            commands = self.extractor.extract(
                text,
                screen_context=screen_context,
                action_history=self.executor.action_history,
            )

            if not commands:
                self._set_status("could not parse command", RED)
                self._set_steps("")
                return

            n = len(commands)
            self._set_status("executing...", AMBER)
            self._set_steps(f"{n} steps")

            # Console log
            print(f"\n{'─'*60}")
            print(f"🎤  \"{text}\"")
            print(f"[Context] {screen_context}")
            for i, cmd in enumerate(commands):
                print(f"  {i+1}. {cmd['tool']}({cmd.get('args', {})})")

            results = self.executor.execute(commands)

            self._set_status("done", CYAN)
            self._set_steps(f"✓ {len(results)} steps")
            print(f"✅  Done — {len(results)} step(s)\n")

            # Fade steps text after 5s
            self.root.after(5000, lambda: self._set_steps(""))

        except Exception as e:
            self._set_status("error", RED)
            self._set_steps("")
            print(f"[GUI] Error: {e}")
        finally:
            self._busy = False

    # ── Run ─────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = FloatingWidget()
    app.run()
