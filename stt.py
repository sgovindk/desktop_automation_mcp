"""
Speech-to-Text module using faster-whisper.
Captures microphone audio, detects silence to segment utterances,
and transcribes using the Whisper base.en model (int8 quantized).
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    SAMPLE_RATE,
    CHANNELS,
    BLOCK_SIZE,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    MAX_RECORDING_DURATION,
    MIN_UTTERANCE_DURATION,
)


class SpeechToText:
    """Captures mic audio, detects speech boundaries, and transcribes."""

    def __init__(self):
        print(f"[STT] Loading Whisper model '{WHISPER_MODEL}' ({WHISPER_COMPUTE_TYPE})...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("[STT] Model loaded.")

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False

    # ── Audio capture callback ──────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        """Called by sounddevice for every audio block. Pushes data to queue."""
        if status:
            print(f"[STT] Audio warning: {status}")
        self._audio_queue.put(indata.copy())

    # ── Silence / voice detection ───────────────────────────────────

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        """Root-mean-square energy of an audio chunk."""
        return float(np.sqrt(np.mean(audio ** 2)))

    # ── Core: wait for one full utterance ───────────────────────────

    def _record_utterance(self) -> np.ndarray | None:
        """
        Blocks until the user speaks and then stops speaking.
        Returns the audio as a float32 numpy array, or None on timeout/noise.
        """
        chunks: list[np.ndarray] = []
        speech_started = False
        silence_start: float | None = None
        recording_start: float | None = None

        while self._running:
            try:
                block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rms = self._rms(block)

            # ── Waiting for speech to begin ──
            if not speech_started:
                if rms >= SILENCE_THRESHOLD:
                    speech_started = True
                    recording_start = time.time()
                    silence_start = None
                    chunks.append(block)
                continue

            # ── Recording speech ──
            chunks.append(block)

            if rms < SILENCE_THRESHOLD:
                # Silence detected
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    # Enough silence → utterance complete
                    break
            else:
                silence_start = None

            # Safety: cap max recording length
            if recording_start and (time.time() - recording_start >= MAX_RECORDING_DURATION):
                print("[STT] Max recording duration reached.")
                break

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        duration = len(audio) / SAMPLE_RATE

        if duration < MIN_UTTERANCE_DURATION:
            return None  # Too short — probably noise

        return audio

    # ── Transcription ───────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run faster-whisper on a float32 audio array and return text."""
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,             # Greedy decoding for speed
            language="en",
            vad_filter=True,         # Built-in voice activity filter
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    # ── Public API ──────────────────────────────────────────────────

    def listen_once(self) -> str:
        """
        Open the microphone, wait for one utterance, transcribe it, and return.
        Blocks until speech is detected and the user stops talking.
        """
        self._running = True

        # Drain any stale data
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        ):
            print("[STT] 🎤  Listening... (speak now)")
            audio = self._record_utterance()

        self._running = False

        if audio is None:
            return ""

        print(f"[STT] Captured {len(audio)/SAMPLE_RATE:.1f}s of audio. Transcribing...")
        text = self._transcribe(audio)
        return text

    def listen_loop(self, callback):
        """
        Continuously listen for utterances and call `callback(text)` for each.
        Runs forever until KeyboardInterrupt.

        Args:
            callback: A function that receives the transcribed text string.
        """
        self._running = True

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        ):
            print("[STT] 🎤  Continuous listening started. Press Ctrl+C to stop.")
            try:
                while self._running:
                    audio = self._record_utterance()
                    if audio is None:
                        continue
                    duration = len(audio) / SAMPLE_RATE
                    print(f"[STT] Captured {duration:.1f}s of audio. Transcribing...")
                    text = self._transcribe(audio)
                    if text:
                        callback(text)
                    print("[STT] 🎤  Listening again...")
            except KeyboardInterrupt:
                print("\n[STT] Stopped.")
                self._running = False

    def stop(self):
        """Signal the listener to stop."""
        self._running = False


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    stt = SpeechToText()

    print("\n=== Single utterance mode ===")
    result = stt.listen_once()
    print(f"\n📝  You said: \"{result}\"\n")

    print("=== Continuous mode (Ctrl+C to stop) ===")
    stt.listen_loop(callback=lambda text: print(f"\n📝  You said: \"{text}\"\n"))
