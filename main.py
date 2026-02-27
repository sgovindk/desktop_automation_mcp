"""
Main Orchestrator — Wires together STT → Intent Extraction → MCP Tool Execution.
This is the entry point that runs the full voice automation pipeline.

Usage:
    python main.py                  # Voice mode (continuous listening)
    python main.py --once           # Single command mode
    python main.py --text "open chrome"  # Text mode (skip voice)
"""

import sys
import time
import json
import argparse
from pathlib import Path

import pyautogui
import pygetwindow as gw

from stt import SpeechToText
from intent_extractor import IntentExtractor
from config import SCREENSHOT_DIR


# ── Tool executor: maps tool names to actual functions ──────────────

class ToolExecutor:
    """
    Executes parsed tool commands from the IntentExtractor.
    Each tool function matches the MCP server's tool vocabulary.
    """

    def __init__(self):
        # Lazy-load the RF detector (only if vision tools are called)
        self._rf_detector = None

        # Action history — keeps track of recent executed actions for context
        self.action_history: list[str] = []

        # Build the tool dispatch table
        self.tools = {
            "open_app":         self._open_app,
            "click":            self._click,
            "double_click":     self._double_click,
            "right_click":      self._right_click,
            "move_mouse":       self._move_mouse,
            "drag":             self._drag,
            "type_text":        self._type_text,
            "press_key":        self._press_key,
            "hotkey":           self._hotkey,
            "scroll":           self._scroll,
            "screenshot":       self._screenshot,
            "wait":             self._wait,
            "focus_window":     self._focus_window,
            "close_window":     self._close_window,
            "get_window_list":  self._get_window_list,
            "find_element":     self._find_element,
            "click_element":    self._click_element,
            "click_search_bar": self._click_search_bar,
            "detect_ui_elements": self._detect_ui_elements,
        }

    @property
    def rf_detector(self):
        if self._rf_detector is None:
            try:
                from ml_model.detector import UIDetectorRF
                from config import RF_MODEL_TYPE, RF_CONFIDENCE
                self._rf_detector = UIDetectorRF(
                    model_type=RF_MODEL_TYPE,
                    confidence_threshold=RF_CONFIDENCE,
                )
            except FileNotFoundError:
                print("[Exec] ⚠️  RF model not trained yet. Vision tools disabled.")
                print("       Run: python ml_model/collect_data.py → train_rf.py")
                self._rf_detector = False
        return self._rf_detector if self._rf_detector else None

    def execute(self, commands: list[dict]) -> list[str]:
        """Execute a sequence of tool commands."""
        results = []
        for i, cmd in enumerate(commands):
            tool = cmd.get("tool")
            args = cmd.get("args", {})

            if tool not in self.tools:
                msg = f"[Exec] ❌ Unknown tool: '{tool}'"
                print(msg)
                results.append(msg)
                continue

            print(f"[Exec] ▶ Step {i+1}: {tool}({args})")
            try:
                result = self.tools[tool](**args)
                print(f"[Exec] ✅ {result}")
                results.append(result)

                # Track action in history
                self.action_history.append(f"{tool}({json.dumps(args)}) → {result}")
                # Keep only last 15 actions
                if len(self.action_history) > 15:
                    self.action_history = self.action_history[-15:]

            except Exception as e:
                msg = f"[Exec] ❌ {tool} failed: {e}"
                print(msg)
                results.append(msg)

        return results

    # ── App Launching ───────────────────────────────────────────────

    def _open_app(self, app_name: str) -> str:
        import subprocess, os
        app_map = {
            "chrome": "chrome", "google chrome": "chrome", "firefox": "firefox",
            "notepad": "notepad", "calculator": "calc", "explorer": "explorer",
            "file explorer": "explorer", "cmd": "cmd", "command prompt": "cmd",
            "terminal": "cmd", "powershell": "powershell", "task manager": "taskmgr",
            "paint": "mspaint", "word": "winword", "excel": "excel",
            "code": "code", "vscode": "code", "edge": "msedge",
            "settings": "ms-settings:", "control panel": "control",
        }
        key = app_name.lower().strip()
        executable = app_map.get(key, key)

        try:
            if executable.startswith("ms-"):
                os.startfile(executable)
            else:
                subprocess.Popen(
                    f'start "" "{executable}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            time.sleep(1.5)
            return f"Launched '{app_name}'"
        except Exception as e:
            return f"Failed to open '{app_name}': {e}"

    # ── Mouse ───────────────────────────────────────────────────────

    def _click(self, x: int = None, y: int = None, element: str = None, button: str = "left") -> str:
        if element:
            return self._click_on_element(element)
        if x is not None and y is not None:
            pyautogui.click(x, y, button=button)
            return f"Clicked ({x}, {y})"
        return "Click requires x,y or element"

    def _double_click(self, x: int = None, y: int = None, element: str = None) -> str:
        if element:
            loc = self._locate_element(element)
            if loc:
                pyautogui.doubleClick(loc[0], loc[1])
                return f"Double-clicked '{element}' at ({loc[0]}, {loc[1]})"
            return f"Could not find element: '{element}'"
        if x is not None and y is not None:
            pyautogui.doubleClick(x, y)
            return f"Double-clicked ({x}, {y})"
        return "Double-click requires x,y or element"

    def _right_click(self, x: int = None, y: int = None, element: str = None) -> str:
        if x is not None and y is not None:
            pyautogui.rightClick(x, y)
            return f"Right-clicked ({x}, {y})"
        return "Right-click requires x,y"

    def _move_mouse(self, x: int, y: int) -> str:
        pyautogui.moveTo(x, y)
        return f"Moved mouse to ({x}, {y})"

    def _drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> str:
        pyautogui.moveTo(start_x, start_y)
        pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
        return f"Dragged from ({start_x},{start_y}) to ({end_x},{end_y})"

    # ── Keyboard ────────────────────────────────────────────────────

    def _type_text(self, text: str, interval: float = 0.02) -> str:
        pyautogui.write(text, interval=interval)
        return f"Typed: '{text}'"

    def _press_key(self, key: str) -> str:
        pyautogui.press(key)
        return f"Pressed: {key}"

    def _hotkey(self, keys: list[str]) -> str:
        pyautogui.hotkey(*keys)
        return f"Hotkey: {'+'.join(keys)}"

    # ── Scroll ──────────────────────────────────────────────────────

    def _scroll(self, direction: str = "down", clicks: int = 3) -> str:
        amount = clicks if direction == "up" else -clicks
        pyautogui.scroll(amount)
        return f"Scrolled {direction} by {clicks}"

    # ── Screenshot ──────────────────────────────────────────────────

    def _screenshot(self) -> str:
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            from PIL import Image
            pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

        filename = f"screenshot_{int(time.time())}.png"
        filepath = str(Path(SCREENSHOT_DIR) / filename)
        pil_img.save(filepath)
        return f"Screenshot saved: {filepath}"

    # ── Wait ────────────────────────────────────────────────────────

    def _wait(self, seconds: float = 1.0) -> str:
        time.sleep(seconds)
        return f"Waited {seconds}s"

    # ── Window Management ───────────────────────────────────────────

    def _focus_window(self, title: str) -> str:
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"No window matching '{title}'"
        win = windows[0]
        try:
            if win.isMinimized:
                win.restore()
            win.activate()
            return f"Focused: '{win.title}'"
        except Exception as e:
            return f"Focus failed: {e}"

    def _close_window(self, title: str = None) -> str:
        if title:
            windows = gw.getWindowsWithTitle(title)
            if windows:
                try:
                    windows[0].close()
                    return f"Closed: '{windows[0].title}'"
                except Exception as e:
                    return f"Close failed: {e}"
            return f"No window matching '{title}'"
        # Close active window
        pyautogui.hotkey("alt", "F4")
        return "Closed active window"

    def _get_window_list(self) -> str:
        result = []
        for win in gw.getAllWindows():
            if win.title.strip():
                result.append(win.title)
        return f"Open windows: {result}"

    def get_screen_context(self) -> str:
        """
        Gather a snapshot of the current screen state:
        - Active window title (tells us which app is in foreground)
        - Detected UI elements (via RF model)
        This context is sent to the LLM so it can skip redundant steps.
        """
        lines = []

        # 1. Active window title
        try:
            active = gw.getActiveWindow()
            if active and active.title.strip():
                title = active.title.strip()
                lines.append(f"Active window: \"{title}\"")

                # Infer app and site from title
                title_lower = title.lower()
                if "chrome" in title_lower or "- google chrome" in title_lower:
                    lines.append("Browser: Google Chrome is in the foreground.")
                    # Extract website from Chrome title (format: "Page Title - Google Chrome")
                    if " - google chrome" in title_lower:
                        page_title = title.rsplit(" - Google Chrome", 1)[0].strip()
                        if page_title:
                            lines.append(f"Page title: \"{page_title}\"")
                        # Detect known sites from page title
                        if "youtube" in title_lower:
                            lines.append("Website: YouTube is currently open.")
                        elif "google" in title_lower:
                            lines.append("Website: Google is currently open.")
                        elif "github" in title_lower:
                            lines.append("Website: GitHub is currently open.")
                        elif "reddit" in title_lower:
                            lines.append("Website: Reddit is currently open.")
                        elif "wikipedia" in title_lower:
                            lines.append("Website: Wikipedia is currently open.")
                elif "firefox" in title_lower:
                    lines.append("Browser: Firefox is in the foreground.")
                elif "edge" in title_lower:
                    lines.append("Browser: Microsoft Edge is in the foreground.")
                elif "notepad" in title_lower:
                    lines.append("App: Notepad is in the foreground.")
                elif "explorer" in title_lower:
                    lines.append("App: File Explorer is in the foreground.")
                elif "code" in title_lower or "visual studio" in title_lower:
                    lines.append("App: VS Code is in the foreground.")
            else:
                lines.append("Active window: Desktop / no active window")
        except Exception:
            lines.append("Active window: unknown")

        # 2. Detected UI elements (quick RF scan)
        try:
            if self.rf_detector:
                import mss, cv2
                import numpy as np
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                detections = self.rf_detector.detect(frame, max_proposals=300)
                if detections:
                    element_summary = []
                    for d in detections[:8]:  # Top 8
                        element_summary.append(
                            f"{d['label']} (conf={d['confidence']:.2f}, "
                            f"center=({d['center']['x']},{d['center']['y']}))"
                        )
                    lines.append(f"Visible UI elements: {', '.join(element_summary)}")
                else:
                    lines.append("Visible UI elements: none detected")
        except Exception:
            pass  # Don't fail the whole pipeline if detection fails

        return "\n".join(lines)

    # ── Vision: RF-based element detection ──────────────────────────

    def _detect_ui_elements(self) -> str:
        if not self.rf_detector:
            return "RF model not loaded (train it first)"
        detections = self.rf_detector.detect_screenshot()
        if not detections:
            return "No UI elements detected"
        labels = [f"{d['label']}({d['confidence']:.2f})" for d in detections[:10]]
        return f"Detected {len(detections)} elements: {', '.join(labels)}"

    def _find_element(self, element: str) -> str:
        """Find a UI element by description using RF detector."""
        loc = self._locate_element(element)
        if loc:
            return f"Found '{element}' at ({loc[0]}, {loc[1]})"
        return f"Could not find '{element}' on screen"

    def _click_element(self, element_index: int = 0) -> str:
        if not self.rf_detector:
            return "RF model not loaded"
        detections = self.rf_detector.detect_screenshot()
        if not detections:
            return "No UI elements detected"
        if element_index >= len(detections):
            return f"Index {element_index} out of range ({len(detections)} detected)"
        target = detections[element_index]
        cx, cy = target["center"]["x"], target["center"]["y"]
        pyautogui.click(cx, cy)
        return f"Clicked '{target['label']}' at ({cx}, {cy})"

    def _click_search_bar(self) -> str:
        """Find and click search bar using RF detector, retrying up to 3 times."""
        if not self.rf_detector:
            return "RF model not loaded"

        import mss, cv2
        import numpy as np

        max_retries = 3
        retry_wait = 2.0  # seconds between retries

        for attempt in range(1, max_retries + 1):
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                img = sct.grab(monitor)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            result = self.rf_detector.find_search_bar(frame)
            if result:
                cx, cy = result["center"]["x"], result["center"]["y"]
                pyautogui.click(cx, cy)
                return f"Clicked search bar at ({cx}, {cy}) [attempt {attempt}]"

            if attempt < max_retries:
                print(f"[Exec] ⏳ Search bar not found (attempt {attempt}/{max_retries}), waiting {retry_wait}s for page to load...")
                time.sleep(retry_wait)

        return "No search bar found after 3 attempts"

    def _click_on_element(self, element_desc: str) -> str:
        """Try to find and click a described element."""
        loc = self._locate_element(element_desc)
        if loc:
            pyautogui.click(loc[0], loc[1])
            return f"Clicked '{element_desc}' at ({loc[0]}, {loc[1]})"
        return f"Could not find '{element_desc}' — try giving coordinates"

    def _locate_element(self, description: str):
        """
        Attempt to locate an element using:
        1. RF detector (if model trained)
        2. Return None if not found
        """
        if not self.rf_detector:
            return None

        import mss, cv2
        import numpy as np
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Map description to class name
        desc_lower = description.lower()
        class_map = {
            "search": "search_bar", "search bar": "search_bar",
            "button": "button", "text field": "text_field",
            "text box": "text_field", "input": "text_field",
            "video": "video_thumbnail", "thumbnail": "video_thumbnail",
            "link": "link", "icon": "icon", "tab": "tab",
            "address bar": "address_bar", "url bar": "address_bar",
            "menu": "menu_item",
        }

        target_class = None
        for key, cls in class_map.items():
            if key in desc_lower:
                target_class = cls
                break

        if target_class:
            result = self.rf_detector.find_element(frame, target_class)
            if result:
                return (result["center"]["x"], result["center"]["y"])

        return None


# ── Main pipeline ───────────────────────────────────────────────────

def process_command(text: str, extractor: IntentExtractor, executor: ToolExecutor):
    """Process a single voice/text command through the full pipeline."""
    print(f"\n{'─'*60}")
    print(f"🎤  Command: \"{text}\"")
    print(f"{'─'*60}")

    # Step 0: Gather screen context
    print("[Context] Scanning current screen state...")
    screen_context = executor.get_screen_context()
    print(f"[Context] {screen_context}")

    # Step 1: Extract intent → list of tool commands (with context)
    print("[Intent] Parsing command...")
    commands = extractor.extract(
        text,
        screen_context=screen_context,
        action_history=executor.action_history,
    )

    if not commands:
        print("[Intent] ❌ Could not parse any commands.")
        return

    print(f"[Intent] Extracted {len(commands)} command(s):")
    for i, cmd in enumerate(commands):
        print(f"  {i+1}. {cmd['tool']}({json.dumps(cmd.get('args', {}))})")

    # Step 2: Execute each command
    print()
    results = executor.execute(commands)

    print(f"\n{'─'*60}")
    print(f"✅  Done — {len(results)} step(s) executed")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser(description="Voice Desktop Automation")
    parser.add_argument("--once", action="store_true",
                        help="Listen for a single command then exit")
    parser.add_argument("--text", type=str, default=None,
                        help="Process a text command instead of using voice")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🎙️  Voice Desktop Automation System")
    print("=" * 60)

    # Initialize components
    extractor = IntentExtractor()
    executor = ToolExecutor()

    # Text mode
    if args.text:
        process_command(args.text, extractor, executor)
        return

    # Voice mode
    stt = SpeechToText()

    if args.once:
        # Single command
        print("\n  Listening for one command...")
        text = stt.listen_once()
        if text:
            process_command(text, extractor, executor)
        else:
            print("  No speech detected.")
    else:
        # Continuous loop
        print("\n  Continuous mode — speak commands naturally.")
        print("  Press Ctrl+C to stop.\n")

        def on_speech(text: str):
            if text.strip():
                process_command(text, extractor, executor)

        stt.listen_loop(callback=on_speech)


if __name__ == "__main__":
    main()
