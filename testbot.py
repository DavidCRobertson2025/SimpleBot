{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env python3\
"""\
testbot.py\
\
Simple pipeline:\
  Mic -> Whisper STT -> E-Ink text -> TTS audio\
\
Requirements (Python packages in your venv):\
  pip install openai pyaudio python-dotenv pillow\
  # Plus the Waveshare e-Paper Python library in your home directory:\
  #   git clone https://github.com/waveshare/e-Paper.git\
  #   cd e-Paper/RaspberryPi_JetsonNano/python\
  #   sudo python3 setup.py install    (or pip install .)\
\
Hardware:\
  - Raspberry Pi 5\
  - USB microphone\
  - USB speaker / USB sound card for audio out\
  - Waveshare 2.13" e-Paper HAT V4 (250x122), SPI enabled in raspi-config\
"""\
\
import os\
import time\
import wave\
import audioop\
import subprocess\
\
import pyaudio\
from dotenv import load_dotenv\
from openai import OpenAI, APIConnectionError\
\
from PIL import Image, ImageDraw, ImageFont\
\
# Waveshare e-Paper driver (installed from e-Paper repo)\
from waveshare_epd import epd2in13_V4\
\
# ---------------------------------------------------------\
# OpenAI setup\
# ---------------------------------------------------------\
load_dotenv()\
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))\
\
STT_MODEL = "whisper-1"\
TTS_MODEL = "gpt-4o-mini-tts"\
TTS_VOICE = "echo"\
\
# ---------------------------------------------------------\
# Audio utilities\
# ---------------------------------------------------------\
def find_audio_devices():\
    """\
    Find reasonable input/output devices.\
    Returns (input_index, output_index).\
    """\
    p = pyaudio.PyAudio()\
    input_index = None\
    output_index = None\
\
    for i in range(p.get_device_count()):\
        info = p.get_device_info_by_index(i)\
        name = info.get("name", "").lower()\
        in_ch = info.get("maxInputChannels", 0)\
        out_ch = info.get("maxOutputChannels", 0)\
\
        if in_ch > 0 and input_index is None:\
            # Prefer USB mic if present\
            if "usb" in name or "mic" in name or "microphone" in name:\
                input_index = i\
\
        if out_ch > 0 and output_index is None:\
            # Prefer USB speaker / audio device\
            if "usb" in name or "audio" in name or "speaker" in name:\
                output_index = i\
\
    # Fallback: first any input/output if still None\
    if input_index is None:\
        for i in range(p.get_device_count()):\
            if p.get_device_info_by_index(i).get("maxInputChannels", 0) > 0:\
                input_index = i\
                break\
\
    if output_index is None:\
        for i in range(p.get_device_count()):\
            if p.get_device_info_by_index(i).get("maxOutputChannels", 0) > 0:\
                output_index = i\
                break\
\
    p.terminate()\
    return input_index, output_index\
\
\
def record_audio(\
    filename="input.wav",\
    threshold=2400,\
    silence_duration=0.6,\
    max_duration=15.0,\
):\
    """\
    Voice-activated recording:\
      - Waits for volume above threshold\
      - Records until silence for `silence_duration` seconds\
      - Stops after `max_duration` seconds just in case\
\
    Returns filename or None if nothing captured.\
    """\
    p = pyaudio.PyAudio()\
    RATE = 44100\
    CHUNK = 1024\
\
    input_index, _ = find_audio_devices()\
    if input_index is None:\
        print("\uc0\u10060  No microphone found.")\
        p.terminate()\
        return None\
\
    print(f"\uc0\u55356 \u57252  Using input device index \{input_index\}")\
    stream = p.open(\
        format=pyaudio.paInt16,\
        channels=1,\
        rate=RATE,\
        input=True,\
        frames_per_buffer=CHUNK,\
        input_device_index=input_index,\
    )\
\
    print("\uc0\u55357 \u56386  Waiting for speech...")\
    frames = []\
    recording_started = False\
    silence_start = None\
    start_time = time.time()\
\
    try:\
        while True:\
            if time.time() - start_time > max_duration:\
                print("\uc0\u9201 \u65039  Max recording duration reached.")\
                break\
\
            data = stream.read(CHUNK, exception_on_overflow=False)\
            rms = audioop.rms(data, 2)\
\
            if not recording_started:\
                # Wait for voice above threshold\
                if rms >= threshold:\
                    print("\uc0\u55356 \u57241 \u65039  Recording started!")\
                    recording_started = True\
                    frames.append(data)\
                continue\
\
            # We are now recording\
            frames.append(data)\
\
            if rms < threshold:\
                if silence_start is None:\
                    silence_start = time.time()\
                elif time.time() - silence_start >= silence_duration:\
                    print("\uc0\u55357 \u57041  Silence detected, stopping.")\
                    break\
            else:\
                silence_start = None\
\
    except KeyboardInterrupt:\
        print("\\n\uc0\u55357 \u57041  Recording interrupted.")\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  Recording error: \{e\}")\
    finally:\
        stream.stop_stream()\
        stream.close()\
        p.terminate()\
\
    if not frames:\
        print("\uc0\u9888 \u65039  No audio captured.")\
        return None\
\
    with wave.open(filename, "wb") as wf:\
        wf.setnchannels(1)\
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))\
        wf.setframerate(RATE)\
        wf.writeframes(b"".join(frames))\
\
    return filename\
\
\
def transcribe_audio(filename: str) -> str:\
    """Send audio file to Whisper and return transcribed text."""\
    if not filename or not os.path.exists(filename):\
        return ""\
\
    print("\uc0\u55358 \u56800  Transcribing with Whisper...")\
    try:\
        with open(filename, "rb") as f:\
            result = client.audio.transcriptions.create(\
                model=STT_MODEL,\
                file=f,\
            )\
        text = result.text.strip()\
        print(f"\uc0\u55357 \u56541  Heard: \{text\}")\
        return text\
    except APIConnectionError:\
        print("\uc0\u10060  Cannot reach OpenAI (network issue).")\
        return ""\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  Transcription error: \{e\}")\
        return ""\
    finally:\
        try:\
            os.remove(filename)\
        except OSError:\
            pass\
\
\
def speak_text(text: str):\
    """Use OpenAI TTS to speak the text through the speaker."""\
    if not text:\
        return\
\
    mp3_path = "tts_output.mp3"\
    wav_path = "tts_output.wav"\
\
    print("\uc0\u55357 \u56586  Generating speech...")\
    try:\
        # Stream TTS to MP3 file\
        with client.audio.speech.with_streaming_response.create(\
            model=TTS_MODEL,\
            voice=TTS_VOICE,\
            input=text,\
        ) as response:\
            response.stream_to_file(mp3_path)\
    except APIConnectionError:\
        print("\uc0\u10060  Cannot reach OpenAI for TTS.")\
        return\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  TTS generation error: \{e\}")\
        return\
\
    # Convert MP3 \uc0\u8594  WAV (mono) for easy playback\
    try:\
        subprocess.run(\
            ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", wav_path],\
            stdout=subprocess.DEVNULL,\
            stderr=subprocess.DEVNULL,\
            check=True,\
        )\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  ffmpeg conversion error: \{e\}")\
        return\
    finally:\
        if os.path.exists(mp3_path):\
            os.remove(mp3_path)\
\
    # Play WAV via PyAudio\
    try:\
        wf = wave.open(wav_path, "rb")\
        p = pyaudio.PyAudio()\
\
        _, output_index = find_audio_devices()\
        if output_index is None:\
            print("\uc0\u10060  No speaker device found.")\
            wf.close()\
            p.terminate()\
            return\
\
        print(f"\uc0\u55357 \u56586  Using output device index \{output_index\}")\
        stream = p.open(\
            format=p.get_format_from_width(wf.getsampwidth()),\
            channels=wf.getnchannels(),\
            rate=wf.getframerate(),\
            output=True,\
            output_device_index=output_index,\
        )\
\
        data = wf.readframes(1024)\
        while data:\
            stream.write(data)\
            data = wf.readframes(1024)\
\
        stream.stop_stream()\
        stream.close()\
        p.terminate()\
        wf.close()\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  Playback error: \{e\}")\
    finally:\
        if os.path.exists(wav_path):\
            os.remove(wav_path)\
\
# ---------------------------------------------------------\
# E-Paper display utilities\
# ---------------------------------------------------------\
def init_epd():\
    """Initialize the 2.13\\" V4 e-Paper and clear it."""\
    epd = epd2in13_V4.EPD()\
    print("\uc0\u55357 \u56764  Initializing e-Paper...")\
    epd.init()\
    epd.Clear(0xFF)  # 0xFF = white\
    return epd\
\
\
def draw_text_on_epd(epd, text: str):\
    """\
    Clear the screen and draw wrapped text on the 250x122 display.\
    """\
    if not text:\
        text = "(no text)"\
\
    # Waveshare driver exposes width/height\
    H = epd.height    # 122\
    W = epd.width     # 250\
\
    # Create a 1-bit (black/white) image\
    image = Image.new("1", (W, H), 255)  # 255 = white\
    draw = ImageDraw.Draw(image)\
\
    # Choose a font\
    try:\
        font = ImageFont.truetype(\
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16\
        )\
    except Exception:\
        font = ImageFont.load_default()\
\
    # Simple word-wrapping based on pixel width\
    def wrap_text(t, max_width):\
        words = t.split()\
        lines = []\
        line = ""\
\
        for word in words:\
            test_line = (line + " " + word).strip()\
            # textlength is available in newer Pillow; fallback to textsize\
            try:\
                width = draw.textlength(test_line, font=font)\
            except AttributeError:\
                width, _ = draw.textsize(test_line, font=font)\
\
            if width <= max_width or not line:\
                line = test_line\
            else:\
                lines.append(line)\
                line = word\
\
        if line:\
            lines.append(line)\
        return lines\
\
    max_width = W - 4   # small margin\
    lines = wrap_text(text, max_width)\
\
    # Compute line height\
    try:\
        bbox = font.getbbox("Ay")\
        line_height = bbox[3] - bbox[1] + 2\
    except Exception:\
        _, line_height = draw.textsize("Ay", font=font)\
        line_height += 2\
\
    y = 0\
    for line in lines:\
        if y + line_height > H:\
            break\
        draw.text((2, y), line, font=font, fill=0)  # 0 = black\
        y += line_height\
\
    print("\uc0\u55357 \u56540  Updating e-Paper display...")\
    epd.display(epd.getbuffer(image))\
\
\
# ---------------------------------------------------------\
# Main loop\
# ---------------------------------------------------------\
def main():\
    epd = None\
    try:\
        epd = init_epd()\
        draw_text_on_epd(epd, "Ready.\\nSpeak after the beep.")\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  Failed to init e-Paper: \{e\}")\
        epd = None\
\
    print("\uc0\u9989  testbot is running. Ctrl+C to exit.\\n")\
\
    try:\
        while True:\
            # Beep via terminal bell so you know it's listening\
            print("\\a", end="", flush=True)\
            print("\uc0\u55356 \u57252  Say something...")\
\
            audio_path = record_audio()\
            if audio_path is None:\
                print("\uc0\u9888 \u65039  Nothing recorded, trying again.\\n")\
                time.sleep(0.5)\
                continue\
\
            text = transcribe_audio(audio_path)\
            if not text:\
                print("\uc0\u9888 \u65039  No transcription, trying again.\\n")\
                time.sleep(0.5)\
                continue\
\
            # Show text on e-paper\
            if epd is not None:\
                try:\
                    draw_text_on_epd(epd, text)\
                except Exception as e:\
                    print(f"\uc0\u9888 \u65039  Display error: \{e\}")\
\
            # Speak it back\
            speak_text(text)\
\
            print("\\n--- Ready for the next phrase ---\\n")\
            # Small delay to avoid hammering the e-paper\
            time.sleep(2.0)\
\
    except KeyboardInterrupt:\
        print("\\n\uc0\u55357 \u56395  Exiting testbot...")\
\
    finally:\
        if epd is not None:\
            try:\
                print("\uc0\u55357 \u56484  Putting e-Paper to sleep.")\
                epd.sleep()\
            except Exception:\
                pass\
\
\
if __name__ == "__main__":\
    main()\
}