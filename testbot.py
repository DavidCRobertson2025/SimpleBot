#!/usr/bin/env python3
"""
testbot.py

Simple pipeline:
  Mic -> Whisper STT -> E-Ink text -> TTS audio

PLUS:
  - Active/Sleep button on GPIO22 (pin 15)
  - Active/Sleep LED on GPIO23 (pin 16)
    * When button is latched (pulled to GND), bot is active and LED is ON.
    * When button is open, bot is sleeping and LED is OFF.

Requirements (Python packages in your venv):
  pip install openai pyaudio python-dotenv pillow adafruit-blinka lgpio
  # Plus the Waveshare e-Paper Python library:
  #   git clone https://github.com/waveshare/e-Paper.git
  #   cd e-Paper/RaspberryPi_JetsonNano/python
  #   sudo python3 setup.py install

Hardware:
  - Raspberry Pi 5
  - USB microphone
  - USB speaker / USB sound card for audio out
  - Waveshare 2.13" e-Paper HAT V4 (250x122), SPI enabled in raspi-config
  - Active/Sleep button: GPIO22 (pin 15) -> switch -> GND
  - Active/Sleep LED: GPIO23 (pin 16) via resistor -> LED -> GND
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

# Waveshare e-Paper driver (installed from e-Paper repo)
from waveshare_epd import epd2in13_V4

# GPIO via Blinka
import board
import digitalio

# ---------------------------------------------------------
# OpenAI setup
# ---------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STT_MODEL = "whisper-1"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "echo"

# ---------------------------------------------------------
# Button + LED configuration
# ---------------------------------------------------------
# Active/Sleep button on GPIO22, physical pin 15
BUTTON_PIN = board.D22
# Active/Sleep LED on GPIO23, physical pin 16
LISTEN_LED_PIN = board.D23

# Button: input with pull-up. When latched to GND → value = False.
listen_button = digitalio.DigitalInOut(BUTTON_PIN)
listen_button.direction = digitalio.Direction.INPUT
listen_button.pull = digitalio.Pull.UP

# LED: output, off by default
listen_led = digitalio.DigitalInOut(LISTEN_LED_PIN)
listen_led.direction = digitalio.Direction.OUTPUT
listen_led.value = False  # off to start

def is_button_on() -> bool:
    """
    Returns True when the Active/Sleep button is in the 'active' position
    (latched/connected to GND). With a pull-up, that means value == False.
    """
    return not listen_button.value

def update_led(armed: bool):
    """Turn LED on when armed (listening), off when sleeping."""
    listen_led.value = bool(armed)

# ---------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------
def find_audio_devices():
    """
    Hard-code the correct USB devices:

    - input_index = 3  (KT USB Audio mic)
    - output_index = 1 (UACDemoV1.0 USB speaker)
    """

    input_index = 2
    output_index = 1

    print("=== Using fixed audio devices ===")
    print(f"🎤 Mic input device index = {input_index}")
    print(f"🔊 Speaker output device index = {output_index}")

    return input_index, output_index

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "").lower()
        in_ch = info.get("maxInputChannels", 0)
        out_ch = info.get("maxOutputChannels", 0)

        if in_ch > 0 and input_index is None:
            # Prefer USB mic if present
            if "usb" in name or "mic" in name or "microphone" in name:
                input_index = i

        if out_ch > 0 and output_index is None:
            # Prefer USB speaker / audio device
            if "usb" in name or "audio" in name or "speaker" in name:
                output_index = i

    # Fallback: first any input/output if still None
    if input_index is None:
        for i in range(p.get_device_count()):
            if p.get_device_info_by_index(i).get("maxInputChannels", 0) > 0:
                input_index = i
                break

    if output_index is None:
        for i in range(p.get_device_count()):
            if p.get_device_info_by_index(i).get("maxOutputChannels", 0) > 0:
                output_index = i
                break

    p.terminate()
    return input_index, output_index


def record_audio(
    filename="input.wav",
    threshold=5500,
    silence_duration=0.3,
    max_duration=7.0,
):
    """
    Voice-activated recording:
      - Waits for volume above threshold
      - Records until silence for `silence_duration` seconds
      - Stops after `max_duration` seconds just in case
      - Aborts immediately if the Active/Sleep button is switched OFF

    Returns filename or None if nothing captured.
    """
    p = pyaudio.PyAudio()
    CHUNK = 1024

    input_index, _ = find_audio_devices()
    if input_index is None:
        print("❌ No microphone found.")
        p.terminate()
        return None

    print(f"🎤 Using input device index {input_index}")

    # Try a few common sample rates until one works
    candidate_rates = []

    # First, try the device's default sample rate if available
    try:
        dev_info = p.get_device_info_by_index(input_index)
        default_rate = int(dev_info.get("defaultSampleRate", 44100))
        candidate_rates.append(default_rate)
    except Exception:
        pass

    # Add some typical USB mic rates
    for r in (44100, 48000, 16000):
        if r not in candidate_rates:
            candidate_rates.append(r)

    stream = None
    RATE = None
    tried = []

    for rate in candidate_rates:
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=input_index,
            )
            RATE = rate
            print(f"✅ Opened microphone at {RATE} Hz")
            break
        except OSError as e:
            tried.append((rate, str(e)))
            continue

    if stream is None or RATE is None:
        print("❌ Could not open microphone at any sample rate. Tried:")
        for r, err in tried:
            print(f"   - {r} Hz → {err}")
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

            # Convert bytes to numpy array of int16 and compute RMS
            samples = np.frombuffer(data, dtype=np.int16)
            if samples.size == 0:
                continue

            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

            if not recording_started:
                # Wait for voice above threshold
                if rms >= threshold:
                    print("🎙️ Recording started!")
                    recording_started = True
                    frames.append(data)
                continue

            # We are now recording
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
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return filename



def transcribe_audio(filename: str) -> str:
    """Send audio file to Whisper and return transcribed text."""
    if not filename or not os.path.exists(filename):
        return ""

    print("🧠 Transcribing with Whisper...")
    try:
        with open(filename, "rb") as f:
            result = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
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


def speak_text(text: str):
    """Use OpenAI TTS to speak the text through the speaker."""
    if not text:
        return

    mp3_path = "tts_output.mp3"
    wav_path = "tts_output.wav"

    print("🔊 Generating speech...")
    try:
        # Stream TTS to MP3 file
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

    # Convert MP3 → WAV (mono) for easy playback
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", wav_path],
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

        _, output_index = find_audio_devices()
        if output_index is None:
            print("❌ No speaker device found.")
            wf.close()
            p.terminate()
            return

        print(f"🔊 Using output device index {output_index}")
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            output_device_index=output_index,
        )

        data = wf.readframes(1024)
        while data:
            # If button flips OFF during playback, you can choose to stop early
            if not is_button_on():
                print("🔕 Button OFF — stopping playback early.")
                break
            stream.write(data)
            data = wf.readframes(1024)

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
# E-Paper display utilities
# ---------------------------------------------------------
def init_epd():
    """Initialize the 2.13\" V4 e-Paper and clear it."""
    epd = epd2in13_V4.EPD()
    print("🖼 Initializing e-Paper...")
    epd.init()
    epd.Clear(0xFF)  # 0xFF = white
    return epd


def draw_text_on_epd(epd, text: str):
    """
    Draw wrapped text in *landscape* orientation, regardless of the panel's
    native (width, height). We create a landscape canvas and rotate only if
    the panel is physically in portrait mode.
    """
    if not text:
        text = "(no text)"

    # Physical panel dimensions reported by the driver
    panel_w = epd.width
    panel_h = epd.height

    # We want to draw in landscape: width = longer side, height = shorter side
    logical_w = max(panel_w, panel_h)
    logical_h = min(panel_w, panel_h)

    # Create a 1-bit (black/white) landscape image
    image = Image.new("1", (logical_w, logical_h), 255)  # 255 = white
    draw = ImageDraw.Draw(image)

    # Choose a font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except Exception:
        font = ImageFont.load_default()

    # Simple word-wrapping based on pixel width
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

    max_width = logical_w - 4   # small margin
    lines = wrap_text(text, max_width)

    # Compute line height
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
        draw.text((2, y), line, font=font, fill=0)  # 0 = black
        y += line_height

    # Now map our landscape canvas onto the physical panel.
    # If the panel is "portrait" (height > width), rotate the image so it fits.
    print("📜 Updating e-Paper display (landscape aware)...")

    if panel_w < panel_h:
        # Panel is taller than wide → rotate our landscape image
        rotated = image.rotate(-90, expand=True)
        # If this is still the wrong way around, change 90 to -90
    else:
        # Panel is already landscape
        rotated = image

    epd.display(epd.getbuffer(rotated))


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
def main():
    epd = None
    try:
        epd = init_epd()
    except Exception as e:
        print(f"⚠️ Failed to init e-Paper: {e}")
        epd = None

    # Set initial armed/sleep state based on button
    armed = is_button_on()
    update_led(armed)

    if epd is not None:
        if armed:
            draw_text_on_epd(epd, "Ready.\nAsk me a question.")
        else:
            draw_text_on_epd(epd, "Sleeping.\nFlip the switch to wake me.")

    print("✅ testbot is running. Ctrl+C to exit.\n")

    try:
        last_armed = armed
        while True:
            # Poll button
            armed = is_button_on()
            if armed != last_armed:
                # State changed
                update_led(armed)
                if epd is not None:
                    if armed:
                        draw_text_on_epd(epd, "Ready.\nAsk me a question.")
                    else:
                        draw_text_on_epd(epd, "Sleeping.\nFlip the switch to wake me.")
                state_str = "ACTIVE (listening)" if armed else "SLEEPING"
                print(f"🔁 Button state changed → {state_str}")
                last_armed = armed

            if not armed:
                # Sleep mode: just wait and poll button
                time.sleep(0.1)
                continue

            # ACTIVE mode:
            print("\a", end="", flush=True)  # Beep via terminal bell
            print("🎤 Say something...")

            audio_path = record_audio()
            if audio_path is None:
                print("⚠️ Nothing recorded, trying again.\n")
                time.sleep(0.5)
                continue

            text = transcribe_audio(audio_path)
            if not text:
                print("⚠️ No transcription, trying again.\n")
                time.sleep(0.5)
                continue

            # Show text on e-paper
            if epd is not None:
                try:
                    draw_text_on_epd(epd, text)
                except Exception as e:
                    print(f"⚠️ Display error: {e}")

            # Speak it back
            speak_text(text)

            print("\n--- Ready for the next phrase ---\n")
            # Small delay to avoid hammering the e-paper
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n👋 Exiting testbot...")

    finally:
        if epd is not None:
            try:
                print("💤 Putting e-Paper to sleep.")
                epd.sleep()
            except Exception:
                pass
        # Clean up GPIO
        try:
            listen_button.deinit()
            listen_led.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
