"""
MCP Server for Desktop Automation.
Exposes atomic tools for clicking, typing, screenshots, opening apps,
scrolling, window management, and UI element detection via YOLO.
Uses stdio transport for zero-overhead local communication.
"""

import os
import sys
import time
import base64
import subprocess
from io import BytesIO
from pathlib import Path

import pyautogui
import pygetwindow as gw
import mss
from PIL import Image
from mcp.server.fastmcp import FastMCP

from config import (
    MCP_SERVER_NAME,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE,
    SCREENSHOT_DIR,
)

# ── Safety: disable pyautogui fail-safe (move mouse to corner to abort) ──
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Small pause between pyautogui actions

# ── Create screenshot directory ──
Path(SCREENSHOT_DIR).mkdir(exist_ok=True)

# ── Initialize MCP Server ──
mcp = FastMCP(MCP_SERVER_NAME)

# ── Lazy-load YOLO model (only when detect_ui_elements is called) ──
_yolo_model = None


def _get_yolo_model():
    """Load YOLO model on first use."""
    global _yolo_model
    if _yolo_model is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model not found at '{YOLO_MODEL_PATH}'. "
                "Train your model first and place best.pt there."
            )
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[MCP] YOLO model loaded from {YOLO_MODEL_PATH}")
    return _yolo_model


# ═══════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════

# ── App Launching ───────────────────────────────────────────────────

@mcp.tool()
def open_app(app_name: str) -> str:
    """Launch an application by name (e.g. 'chrome', 'notepad', 'explorer')."""
    app_map = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "firefox": "firefox",
        "notepad": "notepad",
        "calculator": "calc",
        "explorer": "explorer",
        "file explorer": "explorer",
        "cmd": "cmd",
        "command prompt": "cmd",
        "terminal": "cmd",
        "powershell": "powershell",
        "task manager": "taskmgr",
        "paint": "mspaint",
        "word": "winword",
        "excel": "excel",
        "code": "code",
        "vscode": "code",
    }

    key = app_name.lower().strip()
    executable = app_map.get(key, key)

    try:
        subprocess.Popen(
            executable,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Launched '{app_name}' (executable: {executable})"
    except Exception as e:
        # Fallback: try os.startfile for Windows
        try:
            os.startfile(executable)
            return f"Launched '{app_name}' via startfile"
        except Exception as e2:
            return f"Failed to open '{app_name}': {e2}"


# ── Mouse Actions ──────────────────────────────────────────────────

@mcp.tool()
def click(x: int, y: int, button: str = "left") -> str:
    """Click at screen coordinates (x, y). Button: 'left', 'right', or 'middle'."""
    pyautogui.click(x, y, button=button)
    return f"Clicked ({x}, {y}) with {button} button"


@mcp.tool()
def double_click(x: int, y: int) -> str:
    """Double-click at screen coordinates (x, y)."""
    pyautogui.doubleClick(x, y)
    return f"Double-clicked ({x}, {y})"


@mcp.tool()
def right_click(x: int, y: int) -> str:
    """Right-click at screen coordinates (x, y)."""
    pyautogui.rightClick(x, y)
    return f"Right-clicked ({x}, {y})"


@mcp.tool()
def move_mouse(x: int, y: int) -> str:
    """Move mouse cursor to screen coordinates (x, y)."""
    pyautogui.moveTo(x, y)
    return f"Moved mouse to ({x}, {y})"


@mcp.tool()
def drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> str:
    """Click and drag from (start_x, start_y) to (end_x, end_y)."""
    pyautogui.moveTo(start_x, start_y)
    pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
    return f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"


# ── Keyboard Actions ───────────────────────────────────────────────

@mcp.tool()
def type_text(text: str, interval: float = 0.02) -> str:
    """Type text using the keyboard, character by character."""
    pyautogui.write(text, interval=interval)
    return f"Typed: '{text}'"


@mcp.tool()
def press_key(key: str) -> str:
    """Press a single key (enter, esc, tab, space, backspace, delete, up, down, left, right, f1-f12, etc.)."""
    pyautogui.press(key)
    return f"Pressed key: {key}"


@mcp.tool()
def hotkey(keys: list[str]) -> str:
    """Press a keyboard shortcut. Example: ['ctrl', 'c'] for copy."""
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {'+'.join(keys)}"


# ── Scrolling ──────────────────────────────────────────────────────

@mcp.tool()
def scroll(direction: str = "down", clicks: int = 3) -> str:
    """Scroll up or down. Direction: 'up' or 'down'. Clicks = scroll amount."""
    amount = clicks if direction == "up" else -clicks
    pyautogui.scroll(amount)
    return f"Scrolled {direction} by {clicks}"


# ── Screenshot ─────────────────────────────────────────────────────

@mcp.tool()
def screenshot(save: bool = False) -> str:
    """Take a screenshot. Returns base64-encoded PNG. Optionally saves to disk."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        img = sct.grab(monitor)
        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

    # Convert to base64
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    result = f"Screenshot taken ({pil_img.width}x{pil_img.height})"

    if save:
        filename = f"screenshot_{int(time.time())}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        pil_img.save(filepath)
        result += f" | Saved to {filepath}"

    return b64


@mcp.tool()
def get_screen_size() -> dict:
    """Get the primary screen resolution."""
    w, h = pyautogui.size()
    return {"width": w, "height": h}


# ── Window Management ─────────────────────────────────────────────

@mcp.tool()
def focus_window(title: str) -> str:
    """Bring a window to the foreground by partial title match."""
    windows = gw.getWindowsWithTitle(title)
    if not windows:
        return f"No window found matching '{title}'"
    win = windows[0]
    try:
        if win.isMinimized:
            win.restore()
        win.activate()
        return f"Focused window: '{win.title}'"
    except Exception as e:
        return f"Found window '{win.title}' but failed to focus: {e}"


@mcp.tool()
def get_window_list() -> list[dict]:
    """List all open windows with their titles and positions."""
    result = []
    for win in gw.getAllWindows():
        if win.title.strip():  # Skip empty-title windows
            result.append({
                "title": win.title,
                "x": win.left,
                "y": win.top,
                "width": win.width,
                "height": win.height,
                "minimized": win.isMinimized,
            })
    return result


@mcp.tool()
def close_window(title: str) -> str:
    """Close a window by partial title match."""
    windows = gw.getWindowsWithTitle(title)
    if not windows:
        return f"No window found matching '{title}'"
    try:
        windows[0].close()
        return f"Closed window: '{windows[0].title}'"
    except Exception as e:
        return f"Failed to close window: {e}"


# ── Timing ─────────────────────────────────────────────────────────

@mcp.tool()
def wait(seconds: float = 1.0) -> str:
    """Pause execution for a number of seconds."""
    time.sleep(seconds)
    return f"Waited {seconds}s"


# ── Mouse Info ─────────────────────────────────────────────────────

@mcp.tool()
def get_mouse_position() -> dict:
    """Get the current mouse cursor position."""
    x, y = pyautogui.position()
    return {"x": x, "y": y}


# ── Vision: UI Element Detection (Random Forest) ──────────────────

@mcp.tool()
def detect_ui_elements() -> list[dict]:
    """
    Take a screenshot and run the trained Random Forest model to detect UI elements.
    Returns a list of detected elements with label, bounding box, and confidence.
    """
    detector = _get_rf_detector()
    detections = detector.detect_screenshot()

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


@mcp.tool()
def find_ui_element(element_type: str) -> dict:
    """
    Find a specific UI element type on screen.
    Types: button, search_bar, text_field, video_thumbnail, link, icon, tab, address_bar, menu_item.
    Returns the highest-confidence match with bbox and center coordinates.
    """
    detector = _get_rf_detector()

    # Capture screenshot
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    result = detector.find_element(frame, element_type)
    if result:
        return result
    return {"error": f"No '{element_type}' found on screen"}


@mcp.tool()
def click_element(element_index: int = 0) -> str:
    """
    Detect UI elements on screen, then click the center of the element at the given index.
    Index 0 = highest confidence detection. Use detect_ui_elements first to see what's available.
    """
    detections = detect_ui_elements()
    if not detections:
        return "No UI elements detected on screen"
    if element_index >= len(detections):
        return f"Index {element_index} out of range. Only {len(detections)} elements detected."

    target = detections[element_index]
    cx, cy = target["center"]["x"], target["center"]["y"]
    pyautogui.click(cx, cy)
    return f"Clicked '{target['label']}' at ({cx}, {cy}) [confidence: {target['confidence']}]"


@mcp.tool()
def click_search_bar() -> str:
    """
    Find the search bar on screen using the trained RF model and click it.
    Useful for searching within YouTube, Google, etc.
    """
    detector = _get_rf_detector()

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    result = detector.find_search_bar(frame)
    if result:
        cx, cy = result["center"]["x"], result["center"]["y"]
        pyautogui.click(cx, cy)
        return f"Clicked search bar at ({cx}, {cy}) [confidence: {result['confidence']}]"
    return "No search bar detected on screen"


# ═══════════════════════════════════════════════════════════════════
#  RUN SERVER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[MCP] Starting '{MCP_SERVER_NAME}' server (stdio transport)...")
    mcp.run(transport="stdio")
