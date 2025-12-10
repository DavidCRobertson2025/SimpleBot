import os
import time
import random
import threading
import wave
import pyaudio
import numpy as np
import audioop
import subprocess
import re
import digitalio
import sys

# Load the .env file with the ChatGPT API key
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APIConnectionError

import board
import busio
from adafruit_pca9685 import PCA9685

# ================================================================
#  PLATFORM SELECTION
# ================================================================
# Set this to True if running on Raspberry Pi 5.
# Set to False for Raspberry Pi 4 (default).
USE_PI5 = True


# ================================================================
#  OPENAI CONFIGURATION
# ================================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4.1-mini"
TTS_MODEL = "gpt-4o-mini-tts"
VOICE_NAME = "echo"


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

        # Find USB microphone
        if input_index is None and info.get("maxInputChannels") > 0:
            if "usb" in name or "mic" in name or "microphone" in name:
                input_index = i

        # Find USB speaker
        if output_index is None and info.get("maxOutputChannels") > 0:
            if "usb" in name or "audio" in name or "speaker" in name:
                output_index = i

    p.terminate()
    return input_index, output_index


# ================================================================
#  SERVO / EYE CONFIGURATION
# ================================================================

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Servo channel mapping
LEFT_X, LEFT_Y, LEFT_BLINK = 0, 1, 2
RIGHT_X, RIGHT_Y, RIGHT_BLINK = 3, 4, 5

# Movement limits
X_LIMITS = (70, 110)
Y_LIMITS = (70, 110)
BLINK_LIMITS = (0, 40)

# Servo directions
DIR_LEFT_X = 1
DIR_LEFT_Y = 1
DIR_LEFT_BLINK = 1

DIR_RIGHT_X = 1
DIR_RIGHT_Y = -1
DIR_RIGHT_BLINK = -1

# Blink configuration
BLINK_OPEN_LEFT = -12
BLINK_OPEN_RIGHT = 0
BLINK_SIDE_DELAY = 0.07

MOVE_STEP = 1
MOVE_DELAY = 0.01
BLINK_INTERVAL = (7, 12)
BLINK_SPEED = 0.003
BLINK_HOLD = 0.10

last_blink_timestamp = 0.0

# Servo pulse widths
MIN_PULSE_MS = 0.5
MAX_PULSE_MS = 2.5
PERIOD_MS = 20.0

# Mouth smoothing factor
MOUTH_SMOOTHING = 0.6  # 0 = jumpy, 1 = smooth
previous_audio_level = 0.0

# Global state flags
is_running = True
is_speaking = False
is_thinking = False
is_armed = False  # True when button is ON (pressed), False when OFF
is_offline = False  # True when OpenAI can't be reached

# Track servo angles
current_servo_angles = {}


# ================================================================
#  NEOPIXEL MOUTH CONFIGURATION (Pi 4 / Pi 5)
# ================================================================

NEOPIXEL_PIN = board.D13
NUM_PIXELS = 8

if USE_PI5:
    # ------------------------------------------------------------
    # Raspberry Pi 5 implementation
    # ------------------------------------------------------------
    import adafruit_pixelbuf
    from adafruit_raspberry_pi5_neopixel_write import neopixel_write

    class Pi5PixelBuf(adafruit_pixelbuf.PixelBuf):
        """Custom PixelBuf implementation for Raspberry Pi 5."""

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
    # ------------------------------------------------------------
    # Raspberry Pi 4 implementation
    # ------------------------------------------------------------
    import neopixel
    pixels = neopixel.NeoPixel(
        NEOPIXEL_PIN,
        NUM_PIXELS,
        auto_write=True,
        pixel_order=neopixel.GRB,   # change if colors look off
    )


def show_mouth(amplitude, color=(256, 256, 256)):
    """Display mouth levels symmetrically based on amplitude."""
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
    """Turn off all mouth LEDs."""
    pixels.fill((0, 0, 0))
    pixels.show()


# ================================================================
#  LISTEN BUTTON + LED CONFIGURATION
# ================================================================

# We'll use GPIO22 for the button, GPIO23 for the LED
BUTTON_PIN = board.D22   # Physical pin 15
LISTEN_LED_PIN = board.D23  # Physical pin 16

# Button: input with pull-up, pressed when pulled to GND
listen_button = digitalio.DigitalInOut(BUTTON_PIN)
listen_button.direction = digitalio.Direction.INPUT
listen_button.pull = digitalio.Pull.UP  # not pressed = True, pressed = False

# LED: output, off by default
listen_led = digitalio.DigitalInOut(LISTEN_LED_PIN)
listen_led.direction = digitalio.Direction.OUTPUT
listen_led.value = False


# ================================================================
#  SERVO + EYE CONTROL
# ================================================================

def set_servo_angle(channel, direction, angle):
    """Send corrected angle to PCA9685 servo."""
    if direction == -1:
        angle = 180 - angle

    pulse_range = MAX_PULSE_MS - MIN_PULSE_MS
    pulse_width = MIN_PULSE_MS + (pulse_range * angle / 180.0)
    duty_cycle = int((pulse_width / PERIOD_MS) * 65535)

    pca.channels[channel].duty_cycle = duty_cycle


def move_servos_together(angle_targets, current_angles):
    """Smoothly move several servos together."""
    max_steps = 0
    for ch, (_, target) in angle_targets.items():
        max_steps = max(max_steps, abs(target - current_angles.get(ch, target)))

    if max_steps == 0:
        return

    for step in range(0, max_steps + 1, MOVE_STEP):
        for ch, (direction, target) in angle_targets.items():
            start = current_angles.get(ch, target)
            if start == target:
                continue

            t = min(1.0, step / max_steps)
            new_angle = int(start + (target - start) * t)

            set_servo_angle(ch, direction, new_angle)

        time.sleep(MOVE_DELAY)

    for ch, (_, target) in angle_targets.items():
        current_angles[ch] = target


def random_eye_position(scale=1.0):
    """Generate random eye coordinates."""
    x_mid = (X_LIMITS[0] + X_LIMITS[1]) // 2
    y_mid = (Y_LIMITS[0] + Y_LIMITS[1]) // 2

    x_radius = int((X_LIMITS[1] - X_LIMITS[0]) / 2 * scale)
    y_radius = int((Y_LIMITS[1] - Y_LIMITS[0]) / 2 * scale)

    x = random.randint(x_mid - x_radius, x_mid + x_radius)
    y = random.randint(y_mid - y_radius, y_mid + y_radius)

    return x, y


def blink_eyes(probability=1.0):
    """Full natural blink with staggered eyelid motion."""
    global is_armed
    if not is_armed:
        return

    if random.random() > probability:
        return

    left_open = BLINK_OPEN_LEFT
    right_open = BLINK_OPEN_RIGHT
    closed = BLINK_LIMITS[1]

    left_range = closed - left_open
    right_range = closed - right_open

    steps_total = max(left_range, right_range)
    if steps_total <= 0:
        return

    side_offset_steps = int(round(BLINK_SIDE_DELAY / BLINK_SPEED))

    # Closing motion
    for step in range(0, steps_total + 1):
        left_progress = min(step, left_range) / left_range if left_range > 0 else 1.0

        right_step_corrected = max(0, step - side_offset_steps)
        right_progress = min(right_step_corrected, right_range) / right_range if right_range > 0 else 1.0

        left_angle = int(left_open + left_progress * left_range)
        right_angle = int(right_open + right_progress * right_range)

        set_servo_angle(LEFT_BLINK, DIR_LEFT_BLINK, left_angle)
        set_servo_angle(RIGHT_BLINK, DIR_RIGHT_BLINK, right_angle)

        time.sleep(BLINK_SPEED)

    time.sleep(BLINK_HOLD)

    # Opening motion
    for step in range(steps_total, -1, -1):
        left_progress = min(step, left_range) / left_range if left_range > 0 else 1.0

        right_step_corrected = max(0, step - side_offset_steps)
        right_progress = min(right_step_corrected, right_range) / right_range if right_range > 0 else 1.0

        left_angle = int(left_open + left_progress * left_range)
        right_angle = int(right_open + right_progress * right_range)

        set_servo_angle(LEFT_BLINK, DIR_LEFT_BLINK, left_angle)
        set_servo_angle(RIGHT_BLINK, DIR_RIGHT_BLINK, right_angle)

        time.sleep(BLINK_SPEED)


def wink():
    """Random single-eye wink."""

    global last_blink_timestamp, is_armed
    if not is_armed:
        return

    last_blink_timestamp = time.time()
    chosen_side = random.choice(["left", "right"])

    left_open = BLINK_OPEN_LEFT
    right_open = BLINK_OPEN_RIGHT
    closed = BLINK_LIMITS[1]

    steps = abs(closed - left_open)

    # LEFT wink
    if chosen_side == "left":
        for step in range(steps + 1):
            angle = int(left_open + (closed - left_open) * (step / steps))
            set_servo_angle(LEFT_BLINK, DIR_LEFT_BLINK, angle)
            time.sleep(BLINK_SPEED)

        time.sleep(BLINK_HOLD)

        for step in range(steps, -1, -1):
            angle = int(left_open + (closed - left_open) * (step / steps))
            set_servo_angle(LEFT_BLINK, DIR_LEFT_BLINK, angle)
            time.sleep(BLINK_SPEED)

    else:  # RIGHT wink
        for step in range(steps + 1):
            angle = int(right_open + (closed - right_open) * (step / steps))
            set_servo_angle(RIGHT_BLINK, DIR_RIGHT_BLINK, angle)
            time.sleep(BLINK_SPEED)

        time.sleep(BLINK_HOLD)

        for step in range(steps, -1, -1):
            angle = int(right_open + (closed - right_open) * (step / steps))
            set_servo_angle(RIGHT_BLINK, DIR_RIGHT_BLINK, angle)
            time.sleep(BLINK_SPEED)

    last_blink_timestamp = time.time()


def blink_twice():
    """Double blink."""
    global last_blink_timestamp, is_armed
    if not is_armed:
        return

    for _ in range(2):
        blink_eyes(probability=1.0)
        last_blink_timestamp = time.time()
        time.sleep(0.3)


def eyes_idle_loop():
    """Background idle movement and blinking for eyes."""
    global is_running, is_speaking, is_thinking, last_blink_timestamp, is_armed, is_offline

    next_blink = time.time() + random.uniform(*BLINK_INTERVAL)

    while is_running:

        # If no internet, hold the offline pose (set_offline_face already did it)
        if is_offline:
            time.sleep(0.1)
            continue

        if not is_armed:
            time.sleep(0.1)
            continue

        now = time.time()

        # THINKING MODE
        if is_thinking:
            for _ in range(2):
                new_x, new_y = random_eye_position(scale=0.5)
                targets = {
                    LEFT_X: (DIR_LEFT_X, new_x),
                    LEFT_Y: (DIR_LEFT_Y, new_y),
                    RIGHT_X: (DIR_RIGHT_X, new_x),
                    RIGHT_Y: (DIR_RIGHT_Y, new_y)
                }

                move_servos_together(targets, current_servo_angles)

                if random.random() < 0.3 and now - last_blink_timestamp > 0.2:
                    blink_eyes(probability=1.0)
                    last_blink_timestamp = time.time()

                time.sleep(1)

        # SPEAKING MODE
        elif is_speaking:
            new_x, new_y = random_eye_position(scale=0.3)
            targets = {
                LEFT_X: (DIR_LEFT_X, new_x),
                LEFT_Y: (DIR_LEFT_Y, new_y),
                RIGHT_X: (DIR_RIGHT_X, new_x),
                RIGHT_Y: (DIR_RIGHT_Y, new_y)
            }

            move_servos_together(targets, current_servo_angles)

            if random.random() < 0.2 and now - last_blink_timestamp > 0.2:
                blink_eyes(probability=1.0)
                last_blink_timestamp = time.time()

            time.sleep(random.uniform(0.8, 1.8))

        # IDLE MODE
        else:
            new_x, new_y = random_eye_position(scale=1.0)
            targets = {
                LEFT_X: (DIR_LEFT_X, new_x),
                LEFT_Y: (DIR_LEFT_Y, new_y),
                RIGHT_X: (DIR_RIGHT_X, new_x),
                RIGHT_Y: (DIR_RIGHT_Y, new_y)
            }

            move_servos_together(targets, current_servo_angles)

            if now >= next_blink:
                blink_eyes(probability=1.0)
                last_blink_timestamp = time.time()
                next_blink = time.time() + random.uniform(*BLINK_INTERVAL)

            time.sleep(random.uniform(1, 3))


def led_blink_loop():
    """
    LED behavior:
    - OFF if not armed
    - BLINK when busy (thinking or speaking)
    - SOLID ON when ready for input
    """
    global is_running, is_thinking, is_speaking, listen_led, is_armed

    while is_running:
        if not is_armed:
            listen_led.value = False
            time.sleep(0.1)
            continue

        # Busy (cannot accept input): blink
        if is_thinking or is_speaking:
            listen_led.value = True
            time.sleep(0.3)
            listen_led.value = False
            time.sleep(0.3)
            continue

        # Ready: solid ON
        listen_led.value = True
        time.sleep(0.1)


def idle_speech_loop():
    global is_running, is_speaking, is_thinking, is_armed

    last_spoke_time = time.time()
    IDLE_INTERVAL = 90   # seconds of silence before it talks (adjust here)

    while is_running:
        time.sleep(1)

        # Do NOT speak while the bot is:
        # - Sleeping (switch off)
        # - Thinking
        # - Speaking
        if not is_armed:
            last_spoke_time = time.time()
            continue

        if is_thinking or is_speaking:
            last_spoke_time = time.time()
            continue

        # If too much quiet time passes, say an idle phrase
        if time.time() - last_spoke_time > IDLE_INTERVAL:
            phrase = random.choice(IDLE_PHRASES)
            print(f"[Idle message] {phrase}")
            speak_text(phrase, color=(0, 255, 0))
            last_spoke_time = time.time()


def center_eyes():
    """Move eyes and eyelids to neutral center positions."""
    neutral_x = (X_LIMITS[0] + X_LIMITS[1]) // 2
    neutral_y = (Y_LIMITS[0] + Y_LIMITS[1]) // 2

    left_blink_open = BLINK_OPEN_LEFT
    right_blink_open = BLINK_OPEN_RIGHT

    current_servo_angles.update({
        LEFT_X: neutral_x,
        LEFT_Y: neutral_y,
        LEFT_BLINK: left_blink_open,
        RIGHT_X: neutral_x,
        RIGHT_Y: neutral_y,
        RIGHT_BLINK: right_blink_open
    })

    set_servo_angle(LEFT_X, DIR_LEFT_X, neutral_x)
    set_servo_angle(LEFT_Y, DIR_LEFT_Y, neutral_y)
    set_servo_angle(LEFT_BLINK, DIR_LEFT_BLINK, left_blink_open)

    set_servo_angle(RIGHT_X, DIR_RIGHT_X, neutral_x)
    set_servo_angle(RIGHT_Y, DIR_RIGHT_Y, neutral_y)
    set_servo_angle(RIGHT_BLINK, DIR_RIGHT_BLINK, right_blink_open)


def set_eyelids_closed():
    """Close both eyelids fully (sleep state)."""
    closed = BLINK_LIMITS[1]

    # 1. Drive eyelids to the closed angle
    set_servo_angle(LEFT_BLINK, DIR_LEFT_BLINK, closed)
    set_servo_angle(RIGHT_BLINK, DIR_RIGHT_BLINK, closed)
    current_servo_angles[LEFT_BLINK] = closed
    current_servo_angles[RIGHT_BLINK] = closed

    # 2. Give the servos time to actually move there
    time.sleep(0.3)  # tweak 0.2–0.4s if needed

    # 3. Then relax the servos by turning off PWM on those channels
    pca.channels[LEFT_BLINK].duty_cycle = 0
    pca.channels[RIGHT_BLINK].duty_cycle = 0


def set_eyelids_open():
    """Open both eyelids to normal trim (awake state)."""
    left_open = BLINK_OPEN_LEFT
    right_open = BLINK_OPEN_RIGHT
    set_servo_angle(LEFT_BLINK, DIR_LEFT_BLINK, left_open)
    set_servo_angle(RIGHT_BLINK, DIR_RIGHT_BLINK, right_open)
    current_servo_angles[LEFT_BLINK] = left_open
    current_servo_angles[RIGHT_BLINK] = right_open


def set_offline_face():
    """
    Show a clear visual 'no internet' state:
    - Cross-eyed
    - Red mouth blink
    """
    global is_offline, is_thinking, is_speaking
    is_offline = True
    is_thinking = False
    is_speaking = False

    # Cross-eye pose: left eye looks right, right eye looks left
    neutral_y = (Y_LIMITS[0] + Y_LIMITS[1]) // 2
    left_x_cross = X_LIMITS[1]   # rightmost
    right_x_cross = X_LIMITS[0]  # leftmost

    targets = {
        LEFT_X: (DIR_LEFT_X, left_x_cross),
        LEFT_Y: (DIR_LEFT_Y, neutral_y),
        RIGHT_X: (DIR_RIGHT_X, right_x_cross),
        RIGHT_Y: (DIR_RIGHT_Y, neutral_y),
    }
    move_servos_together(targets, current_servo_angles)

    # Make sure eyelids are open so the pose is visible
    set_eyelids_open()

    # Red mouth blink a few times
    for _ in range(3):
        show_mouth(1.0, color=(255, 0, 0))
        time.sleep(0.25)
        clear_mouth()
        time.sleep(0.25)


# ================================================================
#  AUDIO RECORDING + TRANSCRIPTION
# ================================================================

def transcribe_audio(filename):
    """Use Whisper API to convert audio to text."""
    print("🧠 Transcribing...")

    try:
        with open(filename, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        # If we got here, internet is working again
        global is_offline
        is_offline = False
        return result.text.strip()
    except APIConnectionError:
        print("❌ No internet: cannot reach OpenAI for transcription. Check Wi-Fi.")
        set_offline_face()
        return ""


def is_meaningful_text(text: str) -> bool:
    """
    Heuristic filter to ignore background noise / nonsense.
    Returns False for very short, low-information, or non-speechy strings.
    Allows specific short commands like 'stop', 'exit', 'quit'.
    """
    if not text:
        return False

    t = text.strip().lower()

    # Always allow these short commands
    ALWAYS_ALLOW = {"stop", "exit", "quit"}
    if t in ALWAYS_ALLOW:
        return True

    # Too short overall (e.g., "uh", "ok")
    if len(t) < 5:
        return False

    # Very few alphabetic characters (mostly symbols / numbers)
    letters = sum(1 for c in t if c.isalpha())
    non_space = sum(1 for c in t if not c.isspace())
    if non_space > 0 and letters / non_space < 0.5:
        return False

    # Require at least 2 words with some vowels
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 2:
        return False

    if not any(ch in "aeiou" for ch in t):
        return False

    # Add any specific junk patterns Whisper often produces in your room:
    NOISE_PATTERNS = {"uh", "umm", "mm", "hmm"}
    if t in NOISE_PATTERNS:
        return False

    return True



# ================================================================
#  SPEECH SYNTHESIS + MOUTH MOVEMENT
# ================================================================

def record_audio(filename="input.wav", threshold=2400, silence_duration=0.6):
    """
    Voice-activated audio recorder:
      - Starts when RMS > threshold
      - Stops when RMS < threshold for silence_duration seconds
      - Aborts immediately if the listen button is turned OFF
    """
    audio_interface = pyaudio.PyAudio()

    RATE = 44100
    CHUNK = 1024

    input_index, _ = find_audio_devices()
    if input_index is None:
        print("❌ No microphone found.")
        return None

    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=input_index,
        frames_per_buffer=CHUNK
    )

    print("🎤 Listening for speech...")

    frames = []
    recording_started = False
    silence_start = None

    try:
        while True:
            update_listen_led_state()

            # Abort if button turned off
            if listen_button.value:
                print("🔕 Button turned off — cancelling recording.")
                break

            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(data, 2)

            if not recording_started:
                # Wait for speech above noise floor
                if rms >= threshold:
                    print("🛑 Recording started!")
                    recording_started = True
                    frames.append(data)
                continue

            # Once recording has started:
            frames.append(data)

            # Detect silence
            if rms < threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_duration:
                    print("🛑 Silence detected — stopping.")
                    break
            else:
                silence_start = None

    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted.")

    print("🛑 Finished recording.")

    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

    if not frames:
        print("⚠️ No audio captured.")
        return None

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio_interface.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return filename


def speak_text(text, color=(0, 0, 255)):
    """Speak via TTS and animate mouth with amplitude levels."""
    global is_speaking, previous_audio_level

    is_speaking = True

    mp3_path = "speech_output.mp3"
    wav_path = "speech_output.wav"

    # Generate TTS
    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=VOICE_NAME,
            input=text
        ) as response:
            response.stream_to_file(mp3_path)
        # If we got here, internet is working again
        global is_offline
        is_offline = False

    except APIConnectionError:
        print("❌ No internet: cannot reach OpenAI for speech. Check Wi-Fi.")
        set_offline_face()
        is_speaking = False

        # Optional: a visual “error” blink using the mouth LEDs
        for _ in range(3):
            show_mouth(1.0, color=(255, 0, 0))
            time.sleep(0.25)
            clear_mouth()
            time.sleep(0.25)
        is_speaking = False
        return

    # Convert to mono WAV at 48kHz
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    wave_file = wave.open(wav_path, 'rb')
    audio_interface = pyaudio.PyAudio()

    _, output_index = find_audio_devices()
    if output_index is None:
        print("❌ No speaker found.")
        return

    output_stream = audio_interface.open(
        format=audio_interface.get_format_from_width(wave_file.getsampwidth()),
        channels=wave_file.getnchannels(),
        rate=48000,
        output=True,
        output_device_index=output_index,  # USB speaker index
    )

    chunk_size = 512
    audio_playback_delay = 0.07  # lip-sync correction

    data = wave_file.readframes(chunk_size)
    playback_start_time = time.time() + audio_playback_delay

    while data:
        # Keep LED in sync with button even while speaking
        update_listen_led_state()

        rms = audioop.rms(data, 2) / 32768.0
        level = min(1.0, np.log10(1 + 55 * rms))

        level = (
            MOUTH_SMOOTHING * previous_audio_level +
            (1 - MOUTH_SMOOTHING) * level
        )

        previous_audio_level = level

        show_mouth(level, color=color)

        while time.time() < playback_start_time:
            # Tight sync loop; could be relaxed if needed
            pass

        output_stream.write(data)
        playback_start_time += chunk_size / wave_file.getframerate()

        data = wave_file.readframes(chunk_size)

    clear_mouth()
    output_stream.stop_stream()
    output_stream.close()
    audio_interface.terminate()
    wave_file.close()

    os.remove(mp3_path)
    os.remove(wav_path)

    is_speaking = False


# ================================================================
#  MAIN LOOP + EMOTION PROCESSING
# ================================================================

def update_listen_led_state():
    """
    LED logic (non-blinking decisions):
    - OFF if switch off
    - BLINKING handled by led_blink_loop when busy
    - SOLID ON when armed & ready
    """
    global listen_led, is_armed, is_thinking, is_speaking

    if not is_armed:
        listen_led.value = False
    elif not (is_thinking or is_speaking):
        # Ready state
        listen_led.value = True


def main():
    global is_running, is_thinking, last_blink_timestamp, is_armed

    center_eyes()
    clear_mouth()

    print("🤖 Animatronic Chatbot Ready.")

    # If we're in an interactive terminal, ask for ENTER.
    # If running under systemd/cron (no TTY), skip the prompt.
    if sys.stdin.isatty():
        input("Press ENTER to begin...")
    else:
        # Optional: small delay so you can see logs if run manually as a service
        time.sleep(1)

    # Start background threads
    eye_thread = threading.Thread(target=eyes_idle_loop)
    eye_thread.start()

    led_thread = threading.Thread(target=led_blink_loop)
    led_thread.start()

    idle_thread = threading.Thread(target=idle_speech_loop)
    idle_thread.start()

    # ▶️ Startup announcement
    speak_text("I'm ready. Press the button and ask me a question.", color=(0, 255, 0))

    # Initialize armed state based on current button, but DO NOT move lids yet
    button_on = not listen_button.value
    is_armed = button_on
    last_armed = button_on

    try:
        while True:
            # Update is_armed based on button
            button_on = not listen_button.value
            is_armed = button_on

            # Only move eyelids when the state changes AFTER startup
            if button_on != last_armed:
                if button_on:
                    set_eyelids_open()    # waking up
                else:
                    set_eyelids_closed()  # going to sleep
                last_armed = button_on

            # Update LED based on current state
            update_listen_led_state()

            # If switch is OFF: don't listen, just idle/sleep
            if not button_on:
                is_thinking = False
                time.sleep(0.05)
                continue

            # Switch is ON: listen once
            print("🎤 Listening for speech...")
            audio_path = record_audio()

            # If recording was cancelled (button turned off or no audio), skip this turn
            if audio_path is None:
                is_thinking = False
                continue

            is_thinking = True
            user_text = transcribe_audio(audio_path)

            print(f"🧑 You said: {user_text}")

            os.remove(audio_path)

            # 1️⃣ Completely empty / whitespace → ignore
            if not user_text or not user_text.strip():
                print("⚠️ Nothing clear was transcribed; not sending to OpenAI.")
                is_thinking = False
                time.sleep(0.5)   # small pause so it doesn't spin too fast
                continue

            norm = user_text.lower().strip()

            # 2️⃣ Exit commands (must work even if short)
            if norm in ["quit", "exit", "stop"]:
                is_running = False
                pca.deinit()
                clear_mouth()
                print("👋 Goodbye.")
                break

            # 3️⃣ Easter eggs (also allowed even if short-ish)
            if "wink for me" in norm or norm.startswith("wink") or "can you wink" in norm:
                is_thinking = False
                print("✨ Easter Egg: wink")
                wink()
                continue

            if "blink twice" in norm and "understand" in norm:
                is_thinking = False
                print("✨ Easter Egg: blink twice")
                blink_twice()
                continue

            # 4️⃣ Noise filter – ignore junk / room noise
            if not is_meaningful_text(user_text):
                print("⚠️ Transcription looks like noise; ignoring.")
                is_thinking = False
                time.sleep(0.5)
                continue

            # ----------------------------------------------
            # Normal conversation
            # ----------------------------------------------
            print("🤔 Thinking...")

            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "You are a calm, expressive AI. "
                            "Respond concisely in 1 sentence unless necessary. "
                            "Do NOT start with greetings like 'Hello', 'Hi', or 'How can I help you today?'. "
                            "Just answer the user's request directly. "
                            "Also output emotion as one of: happy, sad, neutral, angry, surprised. "
                            "Format: <text> [emotion: <label>]"
                        )},
                        {"role": "user", "content": user_text},
                    ]
                )
                global is_offline
                is_offline = False

            except APIConnectionError:
                print("❌ No internet: cannot reach OpenAI for chat completion. Check Wi-Fi.")
                # Optional visual feedback
                set_offline_face()
                for _ in range(3):
                    show_mouth(1.0, color=(255, 0, 0))
                    time.sleep(0.25)
                    clear_mouth()
                    time.sleep(0.25)
                is_thinking = False
                continue  # skip this turn and go back to waiting for the next question

            # ----------------------------------------------
            # Handle model response + emotion
            # ----------------------------------------------
            full_reply = response.choices[0].message.content.strip()

            # Extract emotion
            match = re.search(r"\[emotion:\s*(\w+)\]", full_reply, re.IGNORECASE)
            emotion = match.group(1).lower() if match else "neutral"

            # Strip label before TTS
            reply_text = re.sub(r"\[emotion:.*\]", "", full_reply).strip()

            EMOTION_COLORS = {
                "happy": (0, 255, 255),      # yellow-ish
                "sad": (255, 0, 0),          # blue
                "angry": (0, 255, 0),        # red
                "surprised": (255, 255, 0),  # purple
                "neutral": (0, 255, 0),      # default green
            }

            color = EMOTION_COLORS.get(emotion, (0, 255, 0))

            is_thinking = False

            print(f"🤖 {reply_text}  [{emotion}]")
            speak_text(reply_text, color=color)

    except KeyboardInterrupt:
        is_running = False
        pca.deinit()
        clear_mouth()
        print("\n👋 Program stopped.")


if __name__ == "__main__":
    main()
