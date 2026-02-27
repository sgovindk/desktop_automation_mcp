"""
Intent Extractor using Groq API (Llama 3.1 8B).
Converts natural language voice commands into a JSON array
of sequential MCP tool calls for desktop automation.
"""

import json
from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
)

# ── System prompt that teaches the LLM our tool vocabulary ──────

SYSTEM_PROMPT = """You are a desktop automation command parser. Convert the user's natural language instruction into a JSON object with a single key "commands" containing an array of sequential tool calls.

Available tools:

1. open_app        — Launch an application.
   args: { "app_name": "<name>" }

2. click           — Click at screen coordinates or on a detected UI element.
   args: { "x": <int>, "y": <int> }
   OR:   { "element": "<description of what to click>" }

3. double_click    — Double-click at coordinates or on a UI element.
   args: { "x": <int>, "y": <int> }
   OR:   { "element": "<description>" }

4. right_click     — Right-click.
   args: { "x": <int>, "y": <int> }
   OR:   { "element": "<description>" }

5. type_text       — Type text using the keyboard.
   args: { "text": "<string>" }

6. press_key       — Press a single key (enter, esc, tab, space, backspace, delete, up, down, left, right, f1-f12, etc.)
   args: { "key": "<key_name>" }

7. hotkey          — Press a keyboard shortcut.
   args: { "keys": ["<key1>", "<key2>", ...] }

8. scroll          — Scroll up or down.
   args: { "direction": "up" | "down", "clicks": <int> }
   Use clicks=1000 for a normal scroll, clicks=2000 for a big scroll. Minimum 1000.

9. screenshot      — Take a screenshot of the current screen.
   args: {}

10. wait           — Pause for a duration.
    args: { "seconds": <float> }

11. focus_window   — Bring a window to the foreground by title.
    args: { "title": "<partial window title>" }

12. find_element   — Use vision model to locate a UI element on screen.
    args: { "element": "<description of element to find>" }

13. click_search_bar — Use a trained ML model (Random Forest) to detect the search bar on screen and click it.
    args: {}
    USE THIS whenever the user wants to search within a website like YouTube, Google, etc.

14. click_element  — Detect all UI elements on screen with the ML model and click one by index.
    args: { "element_index": <int> }

Rules:
- Output ONLY valid JSON: { "commands": [ { "tool": "...", "args": { ... } }, ... ] }
- Break complex instructions into small atomic steps in the correct order.
- When the user says to click on something described visually (a button, link, thumbnail, etc.), use "click" with "element" arg, NOT coordinates.
- Add "wait" (1 second) after open_app to let the app launch.
- Add "wait" (3 seconds) after pressing enter to navigate to a website URL (e.g. after typing youtube.com and pressing enter). The page needs time to fully load before interacting with it.
- After typing in a browser address bar, use press_key "enter" to navigate, then "wait" 3 seconds for the page to load.
- When searching within a website (YouTube, Google, etc.):
  1. WAIT for the page to fully load first (add "wait" 3 seconds after navigating to the site).
  2. Then use "click_search_bar" to find and click the search bar using the ML vision model.
  3. Then use "type_text" to type the search query.
  4. Then use "press_key" with "enter" to submit.
- Do NOT just type text randomly — always click the search bar FIRST before typing a search query.
- After typing in a search box, use press_key "enter" to search.
- Do NOT add explanations, markdown, or anything outside the JSON.

CONTEXT-AWARENESS (VERY IMPORTANT):
- You will receive the current screen context: the active window title, the URL in the address bar (if a browser), and a list of visible UI elements detected by the ML vision model.
- You will also receive a short history of recently executed actions.
- USE THIS CONTEXT to avoid redundant steps:
  - If Chrome is already the foreground window, do NOT open_app Chrome again. Use focus_window instead only if needed.
  - If the user is already on youtube.com (check the window title or URL), do NOT navigate to youtube.com again.
  - If the user is already on the correct website and wants to search, just click_search_bar → type_text → press_key enter.
  - If a recent action already opened an app or navigated to a site, do NOT repeat it.
- ALWAYS consider what is ALREADY on screen before generating commands.
- Only generate the REMAINING steps needed from the current state, not the full sequence from scratch.
"""


class IntentExtractor:
    """Converts natural language text to a sequence of MCP tool commands."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is empty. Get a free key at https://console.groq.com/keys "
                "and set it in config.py"
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        print(f"[Intent] Groq client ready (model: {GROQ_MODEL})")

    def extract(
        self,
        text: str,
        screen_context: str = "",
        action_history: list[str] = None,
    ) -> list[dict]:
        """
        Send natural language text to Groq and return a list of tool commands.

        Args:
            text: The transcribed voice command.
            screen_context: Description of what's currently on screen.
            action_history: List of recently executed actions.

        Returns:
            List of dicts like [{"tool": "open_app", "args": {"app_name": "chrome"}}, ...]
        """
        if not text.strip():
            return []

        # Build the user message with context
        user_message = ""

        if screen_context:
            user_message += f"[CURRENT SCREEN STATE]\n{screen_context}\n\n"

        if action_history:
            recent = action_history[-8:]  # Last 8 actions max
            user_message += "[RECENT ACTIONS]\n"
            for action in recent:
                user_message += f"- {action}\n"
            user_message += "\n"

        user_message += f"[USER COMMAND]\n{text}"

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        # Handle both { "commands": [...] } and bare [...]
        if isinstance(parsed, dict) and "commands" in parsed:
            commands = parsed["commands"]
        elif isinstance(parsed, list):
            commands = parsed
        else:
            print(f"[Intent] Unexpected response format: {raw}")
            commands = []

        return commands


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = IntentExtractor()

    test_queries = [
        "Open Chrome and go to YouTube and search for tenet trailer",
        "Click on the first video",
        "Open Notepad and type hello world",
        "Take a screenshot",
        "Close this window",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: \"{query}\"")
        print(f"{'='*60}")
        commands = extractor.extract(query)
        print(json.dumps(commands, indent=2))
