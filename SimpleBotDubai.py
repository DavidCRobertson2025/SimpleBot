#!/usr/bin/env python3
"""
simplebot.py

Raspberry Pi 5 + Waveshare 2.13" V4 e-paper + USB mic + USB speaker

Behavior:
- Active/Sleep switch on GPIO22 (pin 15) with LED on GPIO23 (pin 16)
- When ACTIVE:
    1. Listen for English speech (voice-activated).
    2. Transcribe with Whisper (language='en').
    3. Filter out non-meaningful / noisy transcriptions.
    4. Send transcription as a question to ChatGPT with a workshop-coach role.
    5. Speak the answer (<= ~3 spoken sentences per prompt).
    6. Show a short summary version (<= 40 words) on the e-ink screen.

Assumes:
- Mic = PyAudio device index 2
- Speaker = PyAudio device index 1
- TTS = gpt-4o-mini-tts, voice "echo"
- Chat model = gpt-4.1-mini
"""

import os
import time
import wave
import subprocess

import numpy as np
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError

from PIL import Image, ImageDraw, ImageFont

# Waveshare e-Paper driver
from waveshare_epd import epd2in13_V4

# GPIO via Blinka
import board
import digitalio

# ---------------------------------------------------------
# OpenAI setup
# ---------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

STT_MODEL = "whisper-1"
CHAT_MODEL = "gpt-4.1-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "echo"

# Thread ID for Assistant API (persists until program restarts)
assistant_thread_id = None

# Flag to interrupt current operation when button is pressed
interrupt_flag = False

# ---------------------------------------------------------
# Audio device configuration (based on your current listing)
# ---------------------------------------------------------
MIC_DEVICE_INDEX = 2      # 'USB PnP Audio Device: Audio (hw:3,0)' – has input
SPEAKER_DEVICE_INDEX = 1  # 'UACDemoV1.0: USB Audio (hw:2,0)' – speaker out
AUDIO_RATE = 48000        # Hz, supported by your USB devices

# ---------------------------------------------------------
# Button + LED configuration
# ---------------------------------------------------------
# Active/Sleep button on GPIO22, physical pin 15
BUTTON_PIN = board.D22
# Active/Sleep LED on GPIO23, physical pin 16
LISTEN_LED_PIN = board.D23

listen_button = digitalio.DigitalInOut(BUTTON_PIN)
listen_button.direction = digitalio.Direction.INPUT
listen_button.pull = digitalio.Pull.UP  # pressed/latched to GND = False

listen_led = digitalio.DigitalInOut(LISTEN_LED_PIN)
listen_led.direction = digitalio.Direction.OUTPUT
listen_led.value = False  # off initially


def is_button_on() -> bool:
    """
    Returns True when the Active/Sleep button is in the 'active' position
    (latched/connected to GND). With pull-up, that means value == False.
    """
    return not listen_button.value


def update_led(armed: bool):
    """Turn LED on when armed (listening), off when sleeping."""
    listen_led.value = bool(armed)


# ---------------------------------------------------------
# Meaningful text filter (for English only)
# ---------------------------------------------------------
def is_meaningful_text(text: str) -> bool:
    """
    Heuristic filter to ignore background noise / nonsense.
    For English-only input, we require:
      - Not empty
      - At least 2 words
      - Mostly alphabetic characters
      - Contains vowels
      - No non-Latin scripts (Arabic, CJK, etc.)
    """
    if not text:
        return False

    t = text.strip()
    if not t:
        return False

    # Reject Arabic and common non-Latin scripts explicitly
    for ch in t:
        code = ord(ch)
        # Arabic block
        if 0x0600 <= code <= 0x06FF:
            return False
        # CJK & related (Japanese, Chinese, Korean, etc.)
        if 0x3040 <= code <= 0x9FFF:
            return False

    lower = t.lower()

    # Extract "words" as alphabetic sequences
    import re
    words = re.findall(r"[a-zA-Z]+", lower)

    # Require at least 2 words (filters out "yes", "ok thanks", etc.)
    if len(words) < 2:
        return False

    # Require vowels in the combined words (English-ish)
    combined = "".join(words)
    if not any(ch in "aeiou" for ch in combined):
        return False

    # Check that most non-space characters are alphabetic
    non_space_chars = [c for c in lower if not c.isspace()]
    if not non_space_chars:
        return False

    alpha = sum(1 for c in non_space_chars if c.isalpha())
    ratio_alpha = alpha / len(non_space_chars)

    # Require at least 70% alphabetic characters (avoid numbers/symbol junk)
    if ratio_alpha < 0.7:
        return False

    # Common filler noises (just in case)
    NOISE_PATTERNS = {"uh", "umm", "mm", "hmm"}
    if lower in NOISE_PATTERNS:
        return False

    return True



# ---------------------------------------------------------
# Audio: recording (voice-activated, English-only)
# ---------------------------------------------------------
def record_audio(
    filename="input.wav",
    threshold=8000.0,
    silence_duration=0.3,
    max_duration=8.0,
):
    """
    Voice-activated recording:
      - Waits until RMS >= threshold to start
      - Stops when RMS < threshold for silence_duration seconds
      - Never exceeds max_duration
      - Aborts immediately if Active/Sleep button is turned off

    Returns filename or None if nothing captured.
    """
    p = pyaudio.PyAudio()
    CHUNK = 1024

    input_index = MIC_DEVICE_INDEX

    try:
        dev_info = p.get_device_info_by_index(input_index)
        print(f"🎤 Using input device {input_index}: {dev_info.get('name')!r}")
    except Exception as e:
        print(f"❌ Could not get info for input device {input_index}: {e}")
        p.terminate()
        return None

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=input_index,
        )
        print(f"✅ Opened microphone at {AUDIO_RATE} Hz")
    except Exception as e:
        print(f"❌ Could not open mic device {input_index} at {AUDIO_RATE} Hz: {e}")
        p.terminate()
        return None

    print("👂 Waiting for speech...")
    frames = []
    recording_started = False
    silence_start = None
    start_time = time.time()

    try:
        while True:
            # Abort if button flips OFF while recording
            if not is_button_on():
                print("🛑 Button turned OFF — cancelling recording.")
                frames = []
                break

            if time.time() - start_time > max_duration:
                print("⏱️ Max recording duration reached.")
                break

            data = stream.read(CHUNK, exception_on_overflow=False)

            samples = np.frombuffer(data, dtype=np.int16)
            if samples.size == 0:
                continue

            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

            if not recording_started:
                if rms >= threshold:
                    print("🎙️ Recording started!")
                    recording_started = True
                    frames.append(data)
                # else just wait for speech
                continue

            # Already recording
            frames.append(data)

            if rms < threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_duration:
                    print("🛑 Silence detected, stopping.")
                    break
            else:
                silence_start = None

    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted.")
    except Exception as e:
        print(f"⚠️ Recording error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if not frames:
        print("⚠️ No audio captured.")
        return None

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b"".join(frames))

    return filename


def record_while_button_on(
    filename="input.wav",
    max_duration=30.0,
):
    """
    Records audio continuously while the button is ON.
    Stops recording when button is turned OFF.
    
    Args:
        filename: Output WAV file path
        max_duration: Maximum recording duration in seconds (safety limit)
    
    Returns:
        filename or None if nothing captured.
    """
    p = pyaudio.PyAudio()
    CHUNK = 1024

    input_index = MIC_DEVICE_INDEX

    try:
        dev_info = p.get_device_info_by_index(input_index)
        print(f"🎤 Using input device {input_index}: {dev_info.get('name')!r}")
    except Exception as e:
        print(f"❌ Could not get info for input device {input_index}: {e}")
        p.terminate()
        return None

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=input_index,
        )
        print(f"✅ Opened microphone at {AUDIO_RATE} Hz")
    except Exception as e:
        print(f"❌ Could not open mic device {input_index} at {AUDIO_RATE} Hz: {e}")
        p.terminate()
        return None

    print("👂 Recording while button is ON...")
    frames = []
    start_time = time.time()

    try:
        while is_button_on():
            if time.time() - start_time > max_duration:
                print("⏱️ Max recording duration reached.")
                break
            
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        # Button went OFF, stop recording
        if frames:
            print("🛑 Button turned OFF — stopping recording.")
        else:
            print("⚠️ No audio captured.")
    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted.")
    except Exception as e:
        print(f"⚠️ Recording error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if not frames:
        print("⚠️ No audio captured.")
        return None

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b"".join(frames))

    return filename


# ---------------------------------------------------------
# Transcription (Whisper, English-only)
# ---------------------------------------------------------
def transcribe_audio(filename: str) -> str:
    """Transcribe audio with Whisper, forcing English language."""
    if not filename or not os.path.exists(filename):
        return ""

    print("🧠 Transcribing with Whisper (English only)...")

    try:
        with open(filename, "rb") as f:
            result = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
                language="en",  # force English
            )
        text = result.text.strip()
        print(f"📝 Heard: {text}")
        return text
    except APIConnectionError:
        print("❌ Cannot reach OpenAI (network issue).")
        return ""
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return ""
    finally:
        try:
            os.remove(filename)
        except OSError:
            pass


# ---------------------------------------------------------
# TTS + Playback
# ---------------------------------------------------------

def ensure_volume(target_percent=75):
    """
    Ensures the USB speaker volume is set to the given percentage.
    Tries PCM first, then Master if PCM is not available.
    """
    try:
        # Try setting PCM
        subprocess.run(
            ["amixer", "-c", "2", "sset", "PCM", f"{target_percent}%"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    try:
        # Try Master if PCM doesn't exist
        subprocess.run(
            ["amixer", "-c", "2", "sset", "Master", f"{target_percent}%"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    print(f"🔊 Ensured speaker volume = {target_percent}%")



def speak_text(text: str):
    """Use OpenAI TTS to speak the text through the speaker."""
    if not text:
        return

    mp3_path = "tts_output.mp3"
    wav_path = "tts_output.wav"

    # Make sure volume is loud enough
    ensure_volume(25)

    print("🔊 Generating speech...")
    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
        ) as response:
            response.stream_to_file(mp3_path)
    except APIConnectionError:
        print("❌ Cannot reach OpenAI for TTS.")
        return
    except Exception as e:
        print(f"⚠️ TTS generation error: {e}")
        return

    # Convert MP3 to stereo 48 kHz WAV with gain
    try:
        volume_db = 10.0  # ~3x louder
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", mp3_path,
                "-ac", "2",                # stereo
                "-ar", str(AUDIO_RATE),    # 48000 Hz
                "-filter:a", f"volume={volume_db}dB",
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        print(f"⚠️ ffmpeg conversion error: {e}")
        return
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

    # Play WAV via PyAudio
    try:
        wf = wave.open(wav_path, "rb")
        p = pyaudio.PyAudio()

        output_index = SPEAKER_DEVICE_INDEX
        try:
            dev_info = p.get_device_info_by_index(output_index)
            print(f"🔊 Using output device {output_index}: {dev_info.get('name')!r}")
        except Exception as e:
            print(f"❌ Could not get info for output device {output_index}: {e}")
            wf.close()
            p.terminate()
            return

        try:
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=output_index,
            )
        except Exception as e:
            print(f"⚠️ Could not open output stream on device {output_index}: {e}")
            wf.close()
            p.terminate()
            return

        chunk = 1024
        data = wf.readframes(chunk)
        while data:
            # Interrupt playback if button goes ON (user wants to speak)
            if is_button_on():
                print("🔕 Button ON — interrupting playback to listen.")
                break
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
    except Exception as e:
        print(f"⚠️ Playback error: {e}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ---------------------------------------------------------
# WiFi connectivity utilities
# ---------------------------------------------------------
def check_wifi_connected() -> bool:
    """
    Check if WiFi interface (wlan0) has an IP address.
    Returns True if WiFi interface is connected, False otherwise.
    This does NOT check internet connectivity.
    """
    try:
        result = subprocess.run(
            ["ip", "addr", "show", "wlan0"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return "inet " in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"⚠️ WiFi check error: {e}")
    
    return False


def check_internet_connected() -> bool:
    """
    Check if internet connectivity is available by pinging a reliable server.
    Returns True if internet is reachable, False otherwise.
    """
    try:
        ping_result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", "8.8.8.8"],
            capture_output=True,
            timeout=3
        )
        return ping_result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"⚠️ Internet check error: {e}")
    
    return False


def try_connect_wifi():
    """
    Attempt to connect to WiFi using available methods.
    Tries nmcli first, then wpa_cli if available.
    """
    print("📡 Attempting to connect to WiFi...")
    
    # Method 1: Try nmcli (NetworkManager)
    try:
        result = subprocess.run(
            ["nmcli", "radio", "wifi", "on"],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            # Try to reconnect
            subprocess.run(
                ["nmcli", "device", "wifi", "rescan"],
                capture_output=True,
                timeout=5
            )
            time.sleep(2)
            return check_wifi_connected()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Method 2: Try wpa_cli (wpa_supplicant)
    try:
        result = subprocess.run(
            ["wpa_cli", "-i", "wlan0", "reconnect"],
            capture_output=True,
            timeout=5
        )
        time.sleep(3)
        return check_wifi_connected()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return False


# ---------------------------------------------------------
# E-Paper display utilities
# ---------------------------------------------------------
def init_epd():
    """Initialize the 2.13\" V4 e-Paper and clear it."""
    epd = epd2in13_V4.EPD()
    print("🖼 Initializing e-Paper...")
    epd.init()
    epd.Clear(0xFF)  # white
    return epd


def draw_text_on_epd(epd, text: str):
    """
    Draw wrapped text in *landscape* orientation, regardless of panel's native
    orientation. We create a landscape canvas and rotate if needed.
    """
    if not text:
        text = "(no text)"

    panel_w = epd.width
    panel_h = epd.height

    # Logical landscape canvas
    logical_w = max(panel_w, panel_h)
    logical_h = min(panel_w, panel_h)

    image = Image.new("1", (logical_w, logical_h), 255)  # white
    draw = ImageDraw.Draw(image)

    # Font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()

    def wrap_text(t, max_width):
        words = t.split()
        lines = []
        line = ""
        for word in words:
            test_line = (line + " " + word).strip()
            try:
                width = draw.textlength(test_line, font=font)
            except AttributeError:
                width, _ = draw.textsize(test_line, font=font)
            if width <= max_width or not line:
                line = test_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    max_width = logical_w - 4
    # Support manual line breaks first
    raw_lines = text.split("\n")
    lines = []
    for raw in raw_lines:
        wrapped = wrap_text(raw, max_width)
        lines.extend(wrapped)


    try:
        bbox = font.getbbox("Ay")
        line_height = bbox[3] - bbox[1] + 2
    except Exception:
        _, line_height = draw.textsize("Ay", font=font)
        line_height += 2

    y = 0
    for line in lines:
        if y + line_height > logical_h:
            break
        draw.text((2, y), line, font=font, fill=0)
        y += line_height

    print("📜 Updating e-Paper display (landscape aware)...")
    if panel_w < panel_h:
        rotated = image.rotate(-90, expand=True)
    else:
        rotated = image

    epd.display(epd.getbuffer(rotated))


def draw_centered_message(epd, text: str):
    """
    Clear the screen and draw a short message centered in the middle
    of the landscape-oriented display.
    """
    if not text:
        text = "Thinking..."

    panel_w = epd.width
    panel_h = epd.height

    # Logical landscape canvas
    logical_w = max(panel_w, panel_h)
    logical_h = min(panel_w, panel_h)

    # White background
    image = Image.new("1", (logical_w, logical_h), 255)
    draw = ImageDraw.Draw(image)

    # Font (slightly bigger for a status message)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()

    # Measure text
    try:
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)

    # Center coordinates
    x = (logical_w - text_w) // 2
    y = (logical_h - text_h) // 2

    # Draw text in black
    draw.text((x, y), text, font=font, fill=0)

    # Rotate if panel is physically portrait
    if panel_w < panel_h:
        rotated = image.rotate(-90, expand=True)
    else:
        rotated = image

    print(f"📜 Updating e-Paper display with centered message: {text!r}")
    epd.display(epd.getbuffer(rotated))


# ---------------------------------------------------------
# ChatGPT coach logic
# ---------------------------------------------------------
def call_chatgpt(user_text: str) -> tuple[str, str]:
    """
    Send the user's question to ChatGPT with the workshop-coach role
    and return (spoken_bilingual, screen_arabic).

    We instruct the model to format its reply EXACTLY as:

        <short English answer, max ~3 sentences>
        [ar_spoken]
        <short Arabic answer, max ~3 sentences>
        [screen]
        <short Arabic screen version, max 40 words>

    We then:
      - Speak BOTH English + Arabic together.
      - Show ONLY the Arabic screen version on the e-ink display.
    """

    system_prompt = (
        "You are a workshop innovation coach embedded in a small physical robot on the table.\n"
        "Participants speak to you in English. You respond in BOTH English and Arabic.\n"
        "\n"
        "Rules for your answer:\n"
        "1) First, give a short English answer (max 3 spoken sentences).\n"
        "2) Then, give a short Arabic answer (max 3 spoken sentences).\n"
        "3) Then, give a very short Arabic summary for the e-ink screen (max 40 words).\n"
        "4) If participants ask you for definitions or background, explain in very simple language.\n"
        "5) If they say they are done, challenge them with risks, edge cases, or alternative users.\n"
        "\n"
        "Format your reply EXACTLY as:\n"
        "<english spoken answer>\n"
        "[ar_spoken]\n"
        "<arabic spoken answer>\n"
        "[screen]\n"
        "<short arabic screen version>\n"
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
    except APIConnectionError:
        print("❌ Cannot reach OpenAI for chat completion.")
        return "", ""
    except Exception as e:
        print(f"⚠️ Chat completion error: {e}")
        return "", ""

    full = resp.choices[0].message.content.strip()

    # Split off the [screen] section first
    spoken_block = full
    screen_block = ""
    screen_marker = "[screen]"
    if screen_marker in full:
        before, after = full.split(screen_marker, 1)
        spoken_block = before.strip()
        screen_block = after.strip()

    # Within the spoken block, split English vs Arabic spoken parts
    en_spoken = spoken_block
    ar_spoken = ""
    ar_marker = "[ar_spoken]"
    if ar_marker in spoken_block:
        before, after = spoken_block.split(ar_marker, 1)
        en_spoken = before.strip()
        ar_spoken = after.strip()

    # Build bilingual spoken answer: English followed by Arabic
    spoken_bilingual = (en_spoken + "\n" + ar_spoken).strip()

    # Screen text: Arabic-only block after [screen]
    screen_ar = screen_block.strip()

    # Safety fallbacks if the model didn't quite follow format
    if not screen_ar:
        # Prefer Arabic spoken as screen text if available
        screen_ar = ar_spoken or en_spoken

    # Ensure screen text is not too long
    words = screen_ar.split()
    if len(words) > 40:
        screen_ar = " ".join(words[:40])

    return spoken_bilingual, screen_ar


# ---------------------------------------------------------
# Assistant API coach logic
# ---------------------------------------------------------
def call_assistant(user_text: str) -> tuple[str, str]:
    """
    Send the user's question to OpenAI Assistant API and return (spoken_bilingual, screen_arabic).
    
    Uses a persistent thread that stays alive until the program restarts.
    The assistant should be configured with the same system instructions as call_chatgpt.
    
    Returns the same format as call_chatgpt:
      - spoken_bilingual: English + Arabic text to speak
      - screen_ar: Short Arabic text for e-ink display
    """
    global assistant_thread_id
    
    if not ASSISTANT_ID:
        print("❌ ASSISTANT_ID not found in environment variables.")
        return "", ""
    
    try:
        # Create thread if it doesn't exist
        if assistant_thread_id is None:
            print("🧵 Creating new Assistant thread...")
            thread = client.beta.threads.create()
            assistant_thread_id = thread.id
            print(f"✅ Thread created: {assistant_thread_id}")
        else:
            print(f"🧵 Using existing thread: {assistant_thread_id}")
        
        # Add user message to thread
        print("📝 Adding message to thread...")
        client.beta.threads.messages.create(
            thread_id=assistant_thread_id,
            role="user",
            content=user_text
        )
        
        # Create and run the assistant
        print("🤖 Running assistant...")
        run = client.beta.threads.runs.create(
            thread_id=assistant_thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        # Wait for the run to complete
        print("⏳ Waiting for assistant response...")
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=assistant_thread_id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                print(f"❌ Assistant run {run_status.status}: {getattr(run_status, 'last_error', 'Unknown error')}")
                return "", ""
            elif run_status.status == "requires_action":
                print("⚠️ Assistant requires action (function calling) - not supported in this implementation.")
                return "", ""
            
            time.sleep(0.5)  # Poll every 500ms
        
        # Retrieve the assistant's response
        print("📥 Retrieving assistant response...")
        messages = client.beta.threads.messages.list(
            thread_id=assistant_thread_id,
            limit=1
        )
        
        if not messages.data or len(messages.data) == 0:
            print("⚠️ No response from assistant.")
            return "", ""
        
        # Get the latest message content
        latest_message = messages.data[0]
        if latest_message.role != "assistant":
            print("⚠️ Latest message is not from assistant.")
            return "", ""
        
        # Extract text content from message
        full = ""
        for content in latest_message.content:
            if content.type == "text":
                full = content.text.value
                break
        
        if not full:
            print("⚠️ No text content in assistant response.")
            return "", ""
        
        full = full.strip()
        
        # Parse the response using the same logic as call_chatgpt
        # Split off the [screen] section first
        spoken_block = full
        screen_block = ""
        screen_marker = "[screen]"
        if screen_marker in full:
            before, after = full.split(screen_marker, 1)
            spoken_block = before.strip()
            screen_block = after.strip()
        
        # Within the spoken block, split English vs Arabic spoken parts
        en_spoken = spoken_block
        ar_spoken = ""
        ar_marker = "[ar_spoken]"
        if ar_marker in spoken_block:
            before, after = spoken_block.split(ar_marker, 1)
            en_spoken = before.strip()
            ar_spoken = after.strip()
        
        # Build bilingual spoken answer: English followed by Arabic
        spoken_bilingual = (en_spoken + "\n" + ar_spoken).strip()
        
        # Screen text: Arabic-only block after [screen]
        screen_ar = screen_block.strip()
        
        # Safety fallbacks if the model didn't quite follow format
        if not screen_ar:
            # Prefer Arabic spoken as screen text if available
            screen_ar = ar_spoken or en_spoken
        
        # Ensure screen text is not too long
        words = screen_ar.split()
        if len(words) > 40:
            screen_ar = " ".join(words[:40])
        
        return spoken_bilingual, screen_ar
        
    except APIConnectionError:
        print("❌ Cannot reach OpenAI for Assistant API.")
        return "", ""
    except Exception as e:
        print(f"⚠️ Assistant API error: {e}")
        return "", ""


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
def main():
    epd = None
    
    # Step 1: Initialize e-Paper and show "starting"
    try:
        epd = init_epd()
        if epd is not None:
            draw_centered_message(epd, "Starting...")
            print("📺 Display: Starting...")
            time.sleep(1)
    except Exception as e:
        print(f"⚠️ Failed to init e-Paper: {e}")
        epd = None
    
    # Step 2: Check WiFi connectivity
    if epd is not None:
        try:
            draw_centered_message(epd, "Checking WiFi...")
            print("📺 Display: Checking WiFi...")
        except Exception as e:
            print(f"⚠️ Error updating display: {e}")
    
    print("📡 Checking WiFi connectivity...")
    wifi_connected = check_wifi_connected()
    
    if wifi_connected:
        if epd is not None:
            try:
                draw_centered_message(epd, "WiFi Connected")
                print("📺 Display: WiFi Connected")
            except Exception as e:
                print(f"⚠️ Error updating display: {e}")
        print("✅ WiFi is connected")
        time.sleep(1.5)
    else:
        if epd is not None:
            try:
                draw_text_on_epd(epd, "WiFi not connected. Trying to connect...")
                print("📺 Display: WiFi not connected. Trying to connect...")
            except Exception as e:
                print(f"⚠️ Error updating display: {e}")
        print("⚠️ WiFi is not connected. Attempting to connect...")
        
        # Try to connect
        wifi_connected = try_connect_wifi()
        
        if not wifi_connected:
            if epd is not None:
                try:
                    draw_text_on_epd(epd, "WiFi not connected. Please restart.")
                    print("📺 Display: WiFi not connected. Please restart.")
                except Exception as e:
                    print(f"⚠️ Error updating display: {e}")
            print("❌ WiFi is not connected. Please restart the device.")
            return
    
    # Step 3: Check Internet connectivity
    if epd is not None:
        try:
            draw_centered_message(epd, "Checking Internet...")
            print("📺 Display: Checking Internet...")
        except Exception as e:
            print(f"⚠️ Error updating display: {e}")
    
    print("🌐 Checking Internet connectivity...")
    internet_connected = check_internet_connected()
    
    if not internet_connected:
        if epd is not None:
            try:
                draw_text_on_epd(epd, "Internet not connected. Please restart.")
                print("📺 Display: Internet not connected. Please restart.")
            except Exception as e:
                print(f"⚠️ Error updating display: {e}")
        print("❌ Internet is not connected. Please restart the device.")
        return
    
    if epd is not None:
        try:
            draw_centered_message(epd, "Internet Connected")
            print("📺 Display: Internet Connected")
        except Exception as e:
            print(f"⚠️ Error updating display: {e}")
    print("✅ Internet is connected")
    time.sleep(1.5)
    
    # Step 4: Initialize agent
    if epd is not None:
        try:
            draw_centered_message(epd, "Initializing agent...")
            print("📺 Display: Initializing agent...")
        except Exception as e:
            print(f"⚠️ Error updating display: {e}")
    print("🤖 Initializing agent...")
    time.sleep(1)
    
    # Step 5: Ready state
    armed = is_button_on()
    update_led(armed)

    if epd is not None:
        try:
            if armed:
                draw_centered_message(epd, "Listening...")
            else:
                draw_text_on_epd(epd, "Push the button to ask me a question.")
        except Exception as e:
            print(f"⚠️ Error updating display: {e}")

    print("✅ SimpleBot coach is running. Ctrl+C to exit.\n")
    print("📌 Press button to start listening, release to process and respond.\n")

    try:
        last_armed = armed
        audio_path = None
        processing = False
        
        while True:
            # Poll button
            armed = is_button_on()
            
            # Button state changed
            if armed != last_armed:
                update_led(armed)
                state_str = "LISTENING" if armed else "PROCESSING"
                print(f"🔁 Button state changed → {state_str}")
                last_armed = armed
                
                # Button just went ON: interrupt anything and start listening
                if armed:
                    print("🎤 Button ON — starting to listen (interrupting any current operation)...")
                    processing = False  # Stop any processing
                    audio_path = None  # Reset any previous recording
                    if epd is not None:
                        try:
                            draw_centered_message(epd, "Listening...")
                        except Exception as e:
                            print(f"⚠️ Error updating display: {e}")
                    # Start recording immediately
                    print("\a", end="", flush=True)
                    audio_path = record_while_button_on()
                
                # Button just went OFF: stop recording and process
                elif not armed and audio_path and not processing:
                    print("🛑 Button OFF — processing recorded audio...")
                    processing = True
                    if epd is not None:
                        try:
                            draw_centered_message(epd, "Processing...")
                        except Exception as e:
                            print(f"⚠️ Error updating display: {e}")
                    
                    # Process what was recorded
                    text = transcribe_audio(audio_path)
                    audio_path = None  # Clear after processing
                    
                    # Check if button went ON again (user interrupted)
                    if is_button_on():
                        print("🛑 Button pressed again — interrupting processing.")
                        processing = False
                        continue
                    
                    if not text:
                        print("⚠️ No transcription.\n")
                        processing = False
                        if epd is not None:
                            try:
                                draw_text_on_epd(epd, "Push the button to ask me a question.")
                            except Exception as e:
                                print(f"⚠️ Error updating display: {e}")
                        continue
                    
                    # Filter out non-meaningful junk
                    if not is_meaningful_text(text):
                        print(f"⚠️ Transcription looks like noise/junk: {text!r} — ignoring.\n")
                        processing = False
                        if epd is not None:
                            try:
                                draw_text_on_epd(epd, "Push the button to ask me a question.")
                            except Exception as e:
                                print(f"⚠️ Error updating display: {e}")
                        continue
                    
                    # Exit commands 
                    lower = text.strip().lower()
                    if lower in {"stop", "exit", "quit"}:
                        print("🛑 Stop command received — shutting down.")
                        speak_text("Okay, stopping now.")
                        break
                    
                    # Check if button went ON again (user interrupted)
                    if is_button_on():
                        print("🛑 Button pressed again — interrupting processing.")
                        processing = False
                        continue
                    
                    # Show "Thinking..." on e-ink while we call Assistant
                    if epd is not None:
                        try:
                            draw_centered_message(epd, "Thinking...")
                        except Exception as e:
                            print(f"⚠️ Error drawing thinking message: {e}")
                    
                    # Send to Assistant API coach
                    print("🤔 Asking Assistant coach...")
                    spoken_answer, screen_summary = call_assistant(text)
                    
                    # Check again if button went ON (interrupt thinking)
                    if is_button_on():
                        print("🛑 Button pressed — interrupting response.")
                        processing = False
                        continue
                    
                    if not spoken_answer and not screen_summary:
                        print("⚠️ No response from Assistant, skipping.\n")
                        processing = False
                        if epd is not None:
                            try:
                                draw_text_on_epd(epd, "Push the button to ask me a question.")
                            except Exception as e:
                                print(f"⚠️ Error updating display: {e}")
                        continue
                    
                    print(f"🤖 Spoken: {spoken_answer}")
                    print(f"📺 Screen: {screen_summary}")
                    
                    # Update e-ink screen
                    if epd is not None:
                        try:
                            draw_text_on_epd(epd, screen_summary)
                        except Exception as e:
                            print(f"⚠️ Display error: {e}")
                    
                    # Check again if button went ON (interrupt before speaking)
                    if is_button_on():
                        print("🛑 Button pressed — interrupting before speaking.")
                        processing = False
                        continue
                    
                    # Speak answer (will be interrupted if button goes ON)
                    speak_text(spoken_answer)
                    
                    # Final check after speaking
                    if is_button_on():
                        print("🛑 Button pressed — ready to listen.")
                        processing = False
                        continue
                    
                    print("\n--- Ready for next question ---\n")
                    processing = False
                    if epd is not None:
                        try:
                            draw_text_on_epd(epd, "Push the button to ask me a question.")
                        except Exception as e:
                            print(f"⚠️ Error updating display: {e}")
            
            # If button is ON and we're not recording, start recording
            elif armed and audio_path is None and not processing:
                print("🎤 Button ON — starting to listen...")
                if epd is not None:
                    try:
                        draw_centered_message(epd, "Listening...")
                    except Exception as e:
                        print(f"⚠️ Error updating display: {e}")
                print("\a", end="", flush=True)
                audio_path = record_while_button_on()
            
            # If button is OFF and no audio to process, just wait
            elif not armed and audio_path is None and not processing:
                time.sleep(0.1)
                continue
            else:
                # Button state hasn't changed, just wait a bit
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n👋 Exiting SimpleBot...")

    finally:
        if epd is not None:
            try:
                print("💤 Putting e-Paper to sleep.")
                epd.sleep()
            except Exception:
                pass
        try:
            listen_button.deinit()
            listen_led.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
