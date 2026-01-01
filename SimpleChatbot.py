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

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APIConnectionError

import board

# ================================================================
#  PLATFORM SELECTION
# ================================================================
# Set this True on Raspberry Pi 5. False for Pi 4.
USE_PI5 = True

# ================================================================
#  OPENAI CONFIGURATION
# ================================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    # ---- Main-thread loop ----
    def loop(self):
        """
        Run on the MAIN THREAD.
        """
        clock = pygame.time.Clock()
        padding = 24
        line_gap = 12
        print(">>> UI loop started")

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

#                 self.screen.fill(self.bg)
                # Debugging code to test color
                self._tick = (self._tick + 1) % 60
                bg = (255, 0, 0) if self._tick < 30 else (0, 0, 255)
                self.screen.fill(bg)

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

                print(">>> UI flip")
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

        if input_index is None and info.get("maxInputChannels") > 0:
            if "usb" in name or "mic" in name or "microphone" in name:
                input_index = i

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
    try:
        with open(filename, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
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
    threshold=0.055,            # RMS threshold (0..1)
    silence_duration=0.6,
    max_wait_for_speech=5.0,
    max_record_seconds=12.0
):
    audio_interface = pyaudio.PyAudio()
    RATE = 44100
    CHUNK = 1024

    input_index, _ = find_audio_devices()
    if input_index is None:
        ui.set("No microphone found", "Check USB mic")
        return None

    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=input_index,
        frames_per_buffer=CHUNK
    )

    ui.set("Listening", "Say something…")
    frames = []
    recording_started = False
    silence_start = None
    start_time = time.time()

    try:
        while True:
            if button_is_off():
                ui.set("Button off", "Cancelling…")
                break

            if not recording_started and (time.time() - start_time) > max_wait_for_speech:
                ui.set("No speech detected", "Timeout")
                break

            if recording_started and (time.time() - start_time) > max_record_seconds:
                ui.set("Max recording length", "Stopping…")
                break

            update_listen_led_state()

            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = rms_from_int16_bytes(data)

            if not recording_started:
                if rms >= threshold:
                    recording_started = True
                    frames.append(data)
                continue

            frames.append(data)

            if rms < threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_duration:
                    break
            else:
                silence_start = None

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

    if not frames:
        return None

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio_interface.get_sample_size(pyaudio.paInt16))
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
def speak_text(ui: TouchUI, text: str, color=(0, 255, 0)):
    global is_speaking, previous_audio_level, is_offline, is_armed
    is_speaking = True

    mp3_path = "speech_output.mp3"
    wav_path = "speech_output.wav"

    ui.set("Speaking", text)

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

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    wave_file = wave.open(wav_path, "rb")
    audio_interface = pyaudio.PyAudio()

    _, output_index = find_audio_devices()
    if output_index is None:
        ui.set("No speaker found", "Check USB audio output")
        is_speaking = False
        wave_file.close()
        audio_interface.terminate()
        return

    output_stream = audio_interface.open(
        format=audio_interface.get_format_from_width(wave_file.getsampwidth()),
        channels=wave_file.getnchannels(),
        rate=48000,
        output=True,
        output_device_index=output_index,
    )

    chunk_size = 512
    audio_playback_delay = 0.07

    data = wave_file.readframes(chunk_size)
    playback_start_time = time.time() + audio_playback_delay

    while data:
        if is_armed and button_is_off():
            break

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

        # Armed state
        button_on = not button_is_off()
        is_armed = button_on
        last_armed = button_on

        try:
            while is_running and ui.running:
                button_on = not button_is_off()

                # Handle latch/unlatch
                if button_on != last_armed:
                    if button_on:
                        is_armed = True
                        is_thinking = False
                        is_speaking = False
                        clear_mouth()
                        ui.post("Awake", "Listening when you are…")
                    else:
                        is_armed = False
                        is_thinking = False
                        is_speaking = False
                        clear_mouth()
                        ui.post("Sleeping", "Switch ON to wake")
                    last_armed = button_on

                update_listen_led_state()

                if not is_armed:
                    time.sleep(0.05)
                    continue

                # Record
                audio_path = record_audio(ui)
                if audio_path is None:
                    is_thinking = False
                    continue

                # If turned off mid-record, discard
                if button_is_off():
                    is_thinking = False
                    try:
                        os.remove(audio_path)
                    except FileNotFoundError:
                        pass
                    continue

                # Transcribe
                is_thinking = True
                user_text = transcribe_audio(ui, audio_path)

                try:
                    os.remove(audio_path)
                except FileNotFoundError:
                    pass

                ui.post("You said:", user_text)

                if not user_text or not user_text.strip():
                    is_thinking = False
                    time.sleep(0.4)
                    continue

                norm = user_text.lower().strip()
                norm = norm.translate(str.maketrans("", "", string.punctuation))
                first = norm.split()[0] if norm.split() else ""

                if first in {"quit", "exit", "stop"}:
                    ui.post("Stopping", "Goodbye")
                    is_running = False
                    is_armed = False
                    break

                if not is_meaningful_text(user_text):
                    ui.post("Heard noise", "Ignoring…")
                    is_thinking = False
                    time.sleep(0.4)
                    continue

                # Chat completion
                ui.post("Thinking…", "")
                if button_is_off():
                    is_thinking = False
                    continue

                try:
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
                    response = completion.choices[0].message.content
                    is_offline = False

                except APIConnectionError:
                    ui.post("No internet", "Chat failed (OpenAI unreachable)")
                    set_offline_visual()
                    is_thinking = False
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

                is_thinking = False
                if button_is_off():
                    continue

                ui.post("Bot:", reply_text)
                speak_text(ui, reply_text, color=color)

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

