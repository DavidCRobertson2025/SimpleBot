import os
import time
import random
import threading
import wave
import subprocess
import re
import string
import math
import struct
import queue

import pyaudio
import digitalio
from PIL import ImageFont  # (only used to measure fonts? Not needed now; kept minimal)

os.environ.setdefault("SDL_RENDER_DRIVER", "software")
# os.environ.setdefault("SDL_VIDEODRIVER", "wayland")
# os.environ.setdefault("SDL_FBDEV", "/dev/fb0")

# Pick a video driver that matches the session we're actually running in.
# - If DISPLAY is set, we are on X11 (LightDM/Xorg).
# - If WAYLAND_DISPLAY is set, we are on Wayland.
# - Otherwise, don't force anything (pygame will decide).
if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
    os.environ.pop("SDL_VIDEODRIVER", None)  # let SDL use x11
elif "WAYLAND_DISPLAY" in os.environ and os.environ["WAYLAND_DISPLAY"]:
    os.environ["SDL_VIDEODRIVER"] = "wayland"
else:
    os.environ.pop("SDL_VIDEODRIVER", None)

import pygame

import os
from dotenv import load_dotenv

from openai import OpenAI, APIConnectionError

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

client = OpenAI()

import board

# ================================================================
#  PLATFORM SELECTION
# ================================================================
# Set this True on Raspberry Pi 5. False for Pi 4.
USE_PI5 = True

# ================================================================
#  OPENAI CONFIGURATION
# ================================================================
CHAT_MODEL = "gpt-4.1-mini"
TTS_MODEL = "gpt-4o-mini-tts"
VOICE_NAME = "echo"

# ================================================================
#  UI (FREENOVE 4.3" DSI Touchscreen) via pygame
# ================================================================
class TouchUI:
    """
    Full-screen status display for the DSI touchscreen.

    IMPORTANT:
      - Run ui.loop() on the MAIN THREAD.
      - From background threads, call ui.post(status, body) to update text.
      - Do NOT call any pygame display functions from background threads.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.status = "Starting..."
        self.body = ""
        self.running = True

        # Queue for cross-thread UI updates
        self.msgq = queue.SimpleQueue()

        # Init pygame (video/display should be used on main thread; init here is OK
        # as long as you construct TouchUI on the main thread too)
        pygame.init()
        pygame.display.init()
        if not pygame.display.get_init():
            raise RuntimeError("pygame display failed to initialize")

        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("SimpleChatbot")

        # Hide mouse cursor (safe after display set_mode)
        pygame.mouse.set_visible(False)

        self.w, self.h = self.screen.get_size()

        # Font sizes tuned for ~4.3" 480x800-ish screens
        self.font_status = pygame.font.SysFont(None, 54)
        self.font_body = pygame.font.SysFont(None, 40)
        self.font_small = pygame.font.SysFont(None, 28)

        self.bg = (0, 0, 0)
        self.fg = (255, 255, 255)
        self.dim = (180, 180, 180)

        # Simple “alive” indicator
        self._tick = 0

    # ---- Thread-safe API for other threads ----
    def post(self, status: str, body: str = ""):
        """Queue a UI update from any thread."""
        self.msgq.put(((status or "").strip(), (body or "").strip()))

    def set(self, status: str, body: str = ""):
        # Backwards-compatible alias
        self.post(status, body)

    def stop(self):
        """Request UI loop shutdown."""
        self.running = False

    # ---- Internal helpers ----
    def _wrap_text(self, text, font, max_width):
        if not text:
            return []
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if font.size(test)[0] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _drain_messages(self):
        """Apply all queued updates (called from main/UI thread)."""
        updated = False
        while True:
            try:
                status, body = self.msgq.get_nowait()
            except Exception:
                break
            with self._lock:
                self.status = status
                self.body = body
            updated = True
        return updated
    def post(self, status: str, body: str = ""):
        self.msgq.put(((status or "").strip(), (body or "").strip()))

    def set(self, status: str, body: str = ""):
        self.post(status, body)

    # ---- Main-thread loop ----
    def loop(self):
        """
        Run on the MAIN THREAD.
        """
        clock = pygame.time.Clock()
        padding = 24
        line_gap = 12

        # Force an initial paint
        needs_redraw = True

        while self.running:
            # Handle events (main thread)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            while not self.msgq.empty():
                status, body = self.msgq.get_nowait()
                with self._lock:
                    self.status = status
                    self.body = body
            with self._lock:
                status = self.status
                body = self.body

            self.screen.fill(self.bg)

            # Apply updates from worker threads
            if self._drain_messages():
                needs_redraw = True

            # Redraw at 30fps, or only when needed (either is fine)
            # We'll redraw every frame to keep it simple & responsive.
            needs_redraw = True

            if needs_redraw:
                with self._lock:
                    status = self.status
                    body = self.body

                self.screen.fill(self.bg)
                # Debugging code to test color
                # self._tick = (self._tick + 1) % 60
                # bg = (255, 0, 0) if self._tick < 30 else (0, 0, 255)
                # self.screen.fill(bg)

                max_w = self.w - 2 * padding

                # Wrap body
                body_lines = self._wrap_text(body, self.font_body, max_w)

                # Render status (single line; if too long, fall back to smaller)
                status_font = self.font_status if self.font_status.size(status)[0] <= max_w else self.font_body
                status_surf = status_font.render(status, True, self.fg)

                body_surfs = [self.font_body.render(line, True, self.dim) for line in body_lines[:6]]

                # Compute vertical centering
                total_h = status_surf.get_height()
                if body_surfs:
                    total_h += line_gap + sum(s.get_height() for s in body_surfs) + line_gap * (len(body_surfs) - 1)

                y = (self.h - total_h) // 2

                # Draw status centered
                x = (self.w - status_surf.get_width()) // 2
                self.screen.blit(status_surf, (x, y))
                y += status_surf.get_height() + line_gap

                # Draw body lines centered
                for s in body_surfs:
                    x = (self.w - s.get_width()) // 2
                    self.screen.blit(s, (x, y))
                    y += s.get_height() + line_gap

                # “Alive” dot in bottom-right (optional)
                self._tick = (self._tick + 1) % 60
                dot_on = self._tick < 30
                dot_color = self.dim if dot_on else (60, 60, 60)
                pygame.draw.circle(self.screen, dot_color, (self.w - 18, self.h - 18), 6)

                pygame.display.flip()
                needs_redraw = False

            clock.tick(30)

        pygame.quit()


# ================================================================
#  AUDIO DEVICE CONFIGURATION
# ================================================================
IDLE_PHRASES = [
    "Ready when you are.",
    "Anything I can help with?",
    "I'm here whenever you need me.",
    "Just say the word.",
    "How can I help?",
]

def find_audio_devices():
    p = pyaudio.PyAudio()
    input_index = None
    output_index = None

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "").lower()


        # prefer known USB mic
        if input_index is None and info.get("maxInputChannels") > 0:
            if "USB PnP Audio Device" in name:
                input_index = i

        # fallback option
        if input_index is None and info.get("maxInputChannels") > 0:
            if "usb" in name or "mic" in name or "microphone" in name:
                input_index = i

        # prefer known USB speaker
        if output_index is None and info.get("maxOutputChannels") > 0:
            if "UACDemoV1.0" in name:
                output_index = i

        # Fallback option
        if output_index is None and info.get("maxOutputChannels") > 0:
            if "usb" in name or "audio" in name or "speaker" in name:
                output_index = i

    p.terminate()
    return input_index, output_index

# ================================================================
#  GLOBAL STATE FLAGS
# ================================================================
is_running = True
is_speaking = False
is_thinking = False
is_armed = False
is_offline = False

# Mouth smoothing factor
MOUTH_SMOOTHING = 0.6
previous_audio_level = 0.0

# ================================================================
#  NEOPIXEL MOUTH CONFIGURATION (Pi 4 / Pi 5)
# ================================================================
NEOPIXEL_PIN = board.D13
NUM_PIXELS = 8

if USE_PI5:
    import adafruit_pixelbuf
    from adafruit_raspberry_pi5_neopixel_write import neopixel_write

    class Pi5PixelBuf(adafruit_pixelbuf.PixelBuf):
        def __init__(self, pin, size, **kwargs):
            self._pin = pin
            super().__init__(size=size, **kwargs)
        def _transmit(self, buf):
            neopixel_write(self._pin, buf)

    pixels = Pi5PixelBuf(
        NEOPIXEL_PIN,
        NUM_PIXELS,
        auto_write=True,
        byteorder="BRG",
    )
else:
    import neopixel
    pixels = neopixel.NeoPixel(
        NEOPIXEL_PIN,
        NUM_PIXELS,
        auto_write=True,
        pixel_order=neopixel.GRB,
    )

def show_mouth(amplitude, color=(255, 255, 255)):
    amplitude = max(0.0, min(1.0, amplitude))
    num_lit = int(round(amplitude * NUM_PIXELS))

    pixels.fill((0, 0, 0))
    center_left = NUM_PIXELS // 2 - 1
    center_right = NUM_PIXELS // 2

    for i in range(num_lit // 2):
        left_pos = center_left - i
        right_pos = center_right + i
        if 0 <= left_pos < NUM_PIXELS:
            pixels[left_pos] = color
        if 0 <= right_pos < NUM_PIXELS:
            pixels[right_pos] = color

    pixels.show()

def clear_mouth():
    pixels.fill((0, 0, 0))
    pixels.show()

# ================================================================
#  LISTEN BUTTON + LED CONFIGURATION
# ================================================================
BUTTON_PIN = board.D22       # Physical pin 15
LISTEN_LED_PIN = board.D23   # Physical pin 16

listen_button = digitalio.DigitalInOut(BUTTON_PIN)
listen_button.direction = digitalio.Direction.INPUT
listen_button.pull = digitalio.Pull.UP  # not pressed = True

listen_led = digitalio.DigitalInOut(LISTEN_LED_PIN)
listen_led.direction = digitalio.Direction.OUTPUT
listen_led.value = False

def button_is_off() -> bool:
    return listen_button.value  # True means OFF (pull-up)

def update_listen_led_state():
    global is_armed, is_thinking, is_speaking
    if not is_armed:
        listen_led.value = False
    elif not (is_thinking or is_speaking):
        listen_led.value = True

def led_blink_loop():
    global is_running, is_thinking, is_speaking, is_armed
    while is_running:
        if not is_armed:
            listen_led.value = False
            time.sleep(0.1)
            continue
        if is_thinking or is_speaking:
            listen_led.value = True
            time.sleep(0.3)
            listen_led.value = False
            time.sleep(0.3)
            continue
        listen_led.value = True
        time.sleep(0.1)

def idle_speech_loop(ui: TouchUI):
    global is_running, is_speaking, is_thinking, is_armed
    last_spoke_time = time.time()
    IDLE_INTERVAL = 90
    while is_running:
        time.sleep(1)
        if not is_armed or is_thinking or is_speaking:
            last_spoke_time = time.time()
            continue
        if time.time() - last_spoke_time > IDLE_INTERVAL:
            phrase = random.choice(IDLE_PHRASES)
            speak_text(ui, phrase, color=(0, 255, 0))
            last_spoke_time = time.time()

# ================================================================
#  AUDIO HELPERS (no audioop)
# ================================================================
def rms_from_int16_bytes(data: bytes) -> float:
    """
    Compute RMS of 16-bit little-endian mono audio chunk.
    Returns 0..1 float (roughly).
    """
    if not data:
        return 0.0
    count = len(data) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack("<" + "h" * count, data)
    # RMS
    mean_sq = sum((s * s) for s in samples) / count
    rms = math.sqrt(mean_sq) / 32768.0
    return max(0.0, min(1.0, rms))

def curve_level(rms: float) -> float:
    """
    Map RMS (0..1) to a nicer mouth amplitude (0..1)
    without numpy/log10. This saturates smoothly.
    """
    x = 55.0 * rms
    return x / (1.0 + x)

# ================================================================
#  TRANSCRIPTION
# ================================================================
def transcribe_audio(ui: TouchUI, filename: str) -> str:
    ui.set("Transcribing…", "")
    print("[transcribing] opening file:", filename)

    try:
        with open(filename, "rb") as audio_file:
            print("[transcribing] calling OpenAI Whisper...")
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            print("[transcribe] OpenAI returned.")

        global is_offline
        is_offline = False
        return result.text.strip()
    except APIConnectionError:
        ui.set("No internet", "Check Wi-Fi (OpenAI unreachable)")
        set_offline_visual()
        return ""

def is_meaningful_text(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if t in {"stop", "exit", "quit"}:
        return True
    if len(t) < 5:
        return False
    letters = sum(1 for c in t if c.isalpha())
    non_space = sum(1 for c in t if not c.isspace())
    if non_space > 0 and letters / non_space < 0.5:
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 2:
        return False
    if not any(ch in "aeiou" for ch in t):
        return False
    if t in {"uh", "umm", "mm", "hmm"}:
        return False
    return True

# ================================================================
#  RECORDING
# ================================================================
def record_audio(
    ui: TouchUI,
    filename="input.wav",
    max_record_seconds=12.0,
):
    """
    Push-to-talk recording:
      - Start recording immediately (no RMS threshold gate)
      - Keep recording while button is held
      - Stop and save when button is released
    """
    audio_interface = pyaudio.PyAudio()
    RATE = 48000          # USB mic-friendly
    CHUNK = 1024

    input_index, _ = find_audio_devices()
    if input_index is None:
        ui.set("No microphone found", "Check USB mic")
        return None

    try:
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=input_index,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        ui.set("Mic open failed", str(e))
        audio_interface.terminate()
        return None

    ui.set("Listening", "Speak now…")
    frames = []
    start_time = time.time()

    try:
        while True:
            # Stop when user releases the button
            if button_is_off():
                break

            # Safety stop
            if (time.time() - start_time) > max_record_seconds:
                break

            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

    if not frames:
        return None

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # paInt16 = 2 bytes
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return filename

# ================================================================
#  OFFLINE VISUAL
# ================================================================
def set_offline_visual():
    global is_offline, is_thinking, is_speaking
    is_offline = True
    is_thinking = False
    is_speaking = False
    for _ in range(3):
        show_mouth(1.0, color=(255, 0, 0))
        time.sleep(0.25)
        clear_mouth()
        time.sleep(0.25)

# ================================================================
#  TTS + MOUTH
# ================================================================
def speak_text(ui, text: str, color=(0, 255, 0)):
    """Speak via TTS and animate mouth with amplitude levels."""
    global is_speaking, previous_audio_level, is_offline, is_armed

    is_speaking = True

    mp3_path = "speech_output.mp3"
    wav_path = "speech_output.wav"

    ui.set("Speaking", text)

    # --- Generate TTS (OpenAI) ---
    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=VOICE_NAME,
            input=text
        ) as response:
            response.stream_to_file(mp3_path)
        is_offline = False

    except APIConnectionError:
        ui.set("No internet", "TTS failed (OpenAI unreachable)")
        set_offline_visual()
        is_speaking = False
        return

    # --- Convert MP3 to mono WAV at 48kHz ---
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    wave_file = wave.open(wav_path, "rb")
    audio_interface = pyaudio.PyAudio()

    # --- Choose output device robustly ---
    _, output_index = find_audio_devices()

    if output_index is None:
        # Try PyAudio default output device
        try:
            default_info = audio_interface.get_default_output_device_info()
            output_index = int(default_info["index"])
            print(
                f"⚠️ No specific speaker found; using PyAudio default output "
                f"index={output_index} ({default_info.get('name', '')})"
            )
        except Exception as e:
            print(f"❌ No default output device available. Skipping audio playback. ({e})")
            clear_mouth()
            is_speaking = False
            wave_file.close()
            audio_interface.terminate()
            # Clean up files
            for path in (mp3_path, wav_path):
                if os.path.exists(path):
                    os.remove(path)
            return

    # --- Open output stream ---
    output_stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=48000,
        output=True,
        output_device_index=output_index,
        frames_per_buffer=1024,
    )

    chunk_size = 512
    audio_playback_delay = 0.07  # lip-sync correction

    data = wave_file.readframes(chunk_size)
    playback_start_time = time.time() + audio_playback_delay

    while data:
        update_listen_led_state()

        rms = rms_from_int16_bytes(data)
        level = curve_level(rms)

        level = (
            MOUTH_SMOOTHING * previous_audio_level +
            (1 - MOUTH_SMOOTHING) * level
        )
        previous_audio_level = level

        show_mouth(level, color=color)

        while time.time() < playback_start_time:
            pass

        output_stream.write(data)
        playback_start_time += chunk_size / wave_file.getframerate()
        data = wave_file.readframes(chunk_size)

    clear_mouth()

    output_stream.stop_stream()
    output_stream.close()
    audio_interface.terminate()
    wave_file.close()

    # --- Clean up temporary files ---
    for path in (mp3_path, wav_path):
        if os.path.exists(path):
            os.remove(path)

    is_speaking = False


# ================================================================
#  MAIN
# ================================================================
def main():
    """
    IMPORTANT:
      - TouchUI.loop() MUST run on the main thread (pygame display context).
      - The chatbot logic runs in a worker thread and communicates via ui.post().
    """
    global is_running, is_thinking, is_armed, is_speaking, is_offline

    clear_mouth()

    # Create UI on main thread
    ui = TouchUI()
    ui.post("Chatbot starting…", "")

    def bot_worker():
        global is_running, is_thinking, is_armed, is_speaking, is_offline

        time.sleep(0.5)

        # Start background threads (safe: these don't call pygame)
        threading.Thread(target=led_blink_loop, daemon=True).start()
        threading.Thread(target=idle_speech_loop, args=(ui,), daemon=True).start()

        # Startup announcement (audio + neopixels only)
        speak_text(ui, "I'm ready. Press the button and ask me a question.", color=(0, 255, 0))

        # --- Push-to-talk: armed ONLY while button is held down ---
        ui.set("Idle", "Hold button to talk")

        try:
            while is_running and ui.running:
                # Button logic:
                # listen_button.value == True  -> released (OFF)
                # listen_button.value == False -> pressed (ON)
                pressed = not button_is_off()   # True while held down
                is_armed = pressed

                update_listen_led_state()

                # If not pressed, just idle and wait
                if not pressed:
                    is_thinking = False
                    is_speaking = False
                    time.sleep(0.02)
                    continue

                # Pressed: listen for speech
                ui.set("Listening", "Speak now…")

                audio_path = record_audio(ui)
                if audio_path is None:
                    is_thinking = False
                    ui.set("Idle", "Hold button to talk")
                    continue

                # Transcribe
                is_thinking = True
                ui.set("Transcribing...", "")

                try:
                    t0 = time.time()
                    user_text = transcribe_audio(ui, audio_path)
                    print(f"[transcribe] done in {time.time() - t0:.2f}s: {user_text!r}")
                    print(f"[post-transcribe] button_is_off={button_is_off()} pressed_snapshot={pressed} user_text_len={len(user_text)}")
                except Exception as e:
                    print ("[transcribe] ERROR:", repr(e))
                    ui.set("Transcribe error", str(e))
                    user_text = ""
                finally:
                    is_thinking = False

                try:
                    os.remove(audio_path)
                except FileNotFoundError:
                    pass

                ui.set("You said:", user_text)

                print(f"[meaningful] empty={not bool(user_text.strip())} meaningful={is_meaningful_text(user_text)} text={user_text!r}")

                if not user_text or not is_meaningful_text(user_text):
                    ui.set("Idle", "Hold button to talk")
                    time.sleep(0.2)
                    continue

                # Chat
                ui.set("Thinking...", "")
                print(f"[chat] sending to {CHAT_MODEL}: {user_text!r}")

                try:
                    t0 = time.time()
                    completion = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a calm, expressive AI. "
                                    "Respond concisely in 1 sentence unless necessary. "
                                    "Do NOT start with greetings like 'Hello', 'Hi', or 'How can I help you today?'. "
                                    "Just answer the user's request directly. "
                                    "Also output emotion as one of: happy, sad, neutral, angry, surprised. "
                                    "Format: <text> [emotion: <label>]"
                                ),
                            },
                            {"role": "user", "content": user_text},
                        ],
                    )
                    dt = time.time() - t0
                    response = completion.choices[0].message.content
                    print(f"[chat] got response in {dt:.2f}s: {response!r}")
                    is_offline = False

                except APIConnectionError:
                    ui.set("No internet", "Chat failed (OpenAI unreachable)")
                    set_offline_visual()
                    continue

                except Exception as e:
                    print("[chat] ERROR:", repr(e))
                    ui.set("Chat error", str(e))
                    continue

                full_reply = response.strip()
                match = re.search(r"\[emotion:\s*(\w+)\]", full_reply, re.IGNORECASE)
                emotion = match.group(1).lower() if match else "neutral"
                reply_text = re.sub(r"\[emotion:.*\]", "", full_reply).strip()

                EMOTION_COLORS = {
                    "happy": (0, 255, 255),
                    "sad": (255, 0, 0),
                    "angry": (0, 255, 0),
                    "surprised": (255, 255, 0),
                    "neutral": (0, 255, 0),
                }
                color = EMOTION_COLORS.get(emotion, (0, 255, 0))

                # Speak (if released during speaking, speak_text already handles stop)
                is_thinking = False
                ui.set("Bot:", reply_text)
                speak_text(ui, reply_text, color=color)

                ui.set("Idle", "Hold button to talk")

        except KeyboardInterrupt:
            pass
        finally:
            # Signal UI to exit
            ui.stop()

    # Start chatbot logic in the background
    threading.Thread(target=bot_worker, daemon=True).start()

    try:
        # Run UI loop on MAIN THREAD
        ui.loop()
    finally:
        # Cleanup (runs when UI exits)
        is_running = False
        is_armed = False
        try:
            listen_led.value = False
        except Exception:
            pass
        try:
            clear_mouth()
        except Exception:
            pass


if __name__ == "__main__":
    main()

