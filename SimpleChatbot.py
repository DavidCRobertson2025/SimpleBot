{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env python3\
import os\
import time\
import threading\
import wave\
import pyaudio\
import audioop\
import subprocess\
import re\
import digitalio\
import sys\
\
from dotenv import load_dotenv\
load_dotenv()\
\
from openai import OpenAI, APIConnectionError\
import board\
\
# ================================================================\
#  OPENAI CONFIGURATION\
# ================================================================\
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))\
\
CHAT_MODEL = "gpt-4.1-mini"\
TTS_MODEL = "gpt-4o-mini-tts"\
VOICE_NAME = "echo"\
\
# ================================================================\
#  GLOBAL STATE\
# ================================================================\
is_running = True\
is_speaking = False\
is_thinking = False\
is_armed = False      # True when listening is enabled\
is_offline = False    # True when OpenAI can't be reached\
\
# ================================================================\
#  BUTTON + LED CONFIGURATION (same as your existing wiring)\
# ================================================================\
# GPIO22 for button, GPIO23 for LED\
BUTTON_PIN = board.D22      # Physical pin 15\
LISTEN_LED_PIN = board.D23  # Physical pin 16\
\
# Button: input with pull-up, pressed/latched to GND = False\
listen_button = digitalio.DigitalInOut(BUTTON_PIN)\
listen_button.direction = digitalio.Direction.INPUT\
listen_button.pull = digitalio.Pull.UP  # not pressed = True, pressed/latched = False\
\
# LED: output, off by default\
listen_led = digitalio.DigitalInOut(LISTEN_LED_PIN)\
listen_led.direction = digitalio.Direction.OUTPUT\
listen_led.value = False\
\
# ================================================================\
#  AUDIO DEVICE DISCOVERY\
# ================================================================\
def find_audio_devices():\
    """\
    Find a USB microphone and USB speaker (or any reasonable input/output).\
    Returns (input_index, output_index).\
    """\
    p = pyaudio.PyAudio()\
    input_index = None\
    output_index = None\
\
    for i in range(p.get_device_count()):\
        info = p.get_device_info_by_index(i)\
        name = info.get("name", "").lower()\
\
        # Input: any device with input channels, prefer USB mic\
        if input_index is None and info.get("maxInputChannels") > 0:\
            if "usb" in name or "mic" in name or "microphone" in name:\
                input_index = i\
\
        # Output: any device with output channels, prefer USB speaker/audio\
        if output_index is None and info.get("maxOutputChannels") > 0:\
            if "usb" in name or "audio" in name or "speaker" in name:\
                output_index = i\
\
    p.terminate()\
    return input_index, output_index\
\
# ================================================================\
#  NOISE / GARBAGE FILTER FOR TEXT\
# ================================================================\
def is_meaningful_text(text: str) -> bool:\
    """\
    Heuristic filter to ignore background noise / nonsense.\
    Allows specific short commands like 'stop', 'exit', 'quit'.\
    """\
    if not text:\
        return False\
\
    t = text.strip().lower()\
\
    # Always allow these short commands\
    ALWAYS_ALLOW = \{"stop", "exit", "quit"\}\
    if t in ALWAYS_ALLOW:\
        return True\
\
    # Too short overall (e.g., "uh", "ok")\
    if len(t) < 5:\
        return False\
\
    # Very few alphabetic characters (mostly symbols / numbers)\
    letters = sum(1 for c in t if c.isalpha())\
    non_space = sum(1 for c in t if not c.isspace())\
    if non_space > 0 and letters / non_space < 0.5:\
        return False\
\
    # If there are no vowels, it's probably not real speech\
    if not any(ch in "aeiou" for ch in t):\
        return False\
\
    # Junk filler patterns\
    NOISE_PATTERNS = \{"uh", "umm", "mm", "hmm"\}\
    if t in NOISE_PATTERNS:\
        return False\
\
    return True\
\
# ================================================================\
#  TRANSCRIPTION (WHISPER)\
# ================================================================\
def transcribe_audio(filename: str) -> str:\
    """Use Whisper API to convert audio to text."""\
    global is_offline\
    print("\uc0\u55358 \u56800  Transcribing...")\
\
    try:\
        with open(filename, "rb") as audio_file:\
            result = client.audio.transcriptions.create(\
                model="whisper-1",\
                file=audio_file\
            )\
        is_offline = False\
        return result.text.strip()\
    except APIConnectionError:\
        print("\uc0\u10060  No internet: cannot reach OpenAI for transcription. Check Wi-Fi.")\
        is_offline = True\
        return ""\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  Transcription error: \{e\}")\
        return ""\
\
# ================================================================\
#  AUDIO RECORDING (VOICE-ACTIVATED)\
# ================================================================\
def record_audio(\
    filename: str = "input.wav",\
    threshold: int = 2400,\
    silence_duration: float = 0.6,\
    max_duration: float = 20.0,\
):\
    """\
    Voice-activated audio recorder:\
      - Starts when RMS > threshold\
      - Stops when RMS < threshold for silence_duration seconds\
      - Aborts immediately if the listen button is turned OFF\
    """\
    global is_armed\
\
    audio_interface = pyaudio.PyAudio()\
    RATE = 44100\
    CHUNK = 1024\
\
    input_index, _ = find_audio_devices()\
    if input_index is None:\
        print("\uc0\u10060  No microphone found.")\
        return None\
\
    print(f"\uc0\u55356 \u57252  Using input device index \{input_index\}")\
\
    stream = audio_interface.open(\
        format=pyaudio.paInt16,\
        channels=1,\
        rate=RATE,\
        input=True,\
        frames_per_buffer=CHUNK,\
        input_device_index=input_index,\
    )\
\
    print("\uc0\u55357 \u56386  Waiting for speech...")\
\
    frames = []\
    recording_started = False\
    silence_start = None\
    start_time = time.time()\
\
    try:\
        while True:\
            # If button is OFF, abort immediately\
            button_on = not listen_button.value  # True when latched ON\
            if not button_on:\
                print("\uc0\u55357 \u57041  Button turned OFF \'97 cancelling recording.")\
                is_armed = False\
                break\
\
            # Time limit safety\
            if time.time() - start_time > max_duration:\
                print("\uc0\u9201 \u65039  Max recording duration reached.")\
                break\
\
            data = stream.read(CHUNK, exception_on_overflow=False)\
            rms = audioop.rms(data, 2)\
\
            if not recording_started:\
                # Wait for speech above noise floor\
                if rms >= threshold:\
                    print("\uc0\u55356 \u57241 \u65039  Recording started!")\
                    recording_started = True\
                    frames.append(data)\
                continue\
\
            # Once recording has started:\
            frames.append(data)\
\
            # Detect silence\
            if rms < threshold:\
                if silence_start is None:\
                    silence_start = time.time()\
                elif time.time() - silence_start >= silence_duration:\
                    print("\uc0\u55357 \u57041  Silence detected \'97 stopping.")\
                    break\
            else:\
                silence_start = None\
\
    except KeyboardInterrupt:\
        print("\\n\uc0\u55357 \u57041  Recording interrupted from keyboard.")\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  Recording error: \{e\}")\
    finally:\
        print("\uc0\u55357 \u57041  Finished recording.")\
        stream.stop_stream()\
        stream.close()\
        audio_interface.terminate()\
\
    if not frames:\
        print("\uc0\u9888 \u65039  No audio captured.")\
        return None\
\
    with wave.open(filename, 'wb') as wf:\
        wf.setnchannels(1)\
        wf.setsampwidth(audio_interface.get_sample_size(pyaudio.paInt16))\
        wf.setframerate(RATE)\
        wf.writeframes(b"".join(frames))\
\
    return filename\
\
# ================================================================\
#  TTS + PLAYBACK (NO MOUTH / SERVOS)\
# ================================================================\
def speak_text(text: str):\
    """Speak via TTS. Button OFF will abort speech and stop listening."""\
    global is_speaking, is_armed, is_offline\
\
    if not text:\
        return\
\
    is_speaking = True\
    mp3_path = "speech_output.mp3"\
    wav_path = "speech_output.wav"\
\
    # Generate TTS to MP3\
    try:\
        with client.audio.speech.with_streaming_response.create(\
            model=TTS_MODEL,\
            voice=VOICE_NAME,\
            input=text,\
        ) as response:\
            response.stream_to_file(mp3_path)\
        is_offline = False\
    except APIConnectionError:\
        print("\uc0\u10060  No internet: cannot reach OpenAI for TTS. Check Wi-Fi.")\
        is_offline = True\
        is_speaking = False\
        return\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  TTS error: \{e\}")\
        is_speaking = False\
        return\
\
    # Convert MP3 to WAV using ffmpeg for easier playback via PyAudio\
    try:\
        subprocess.run(\
            ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", wav_path],\
            stdout=subprocess.DEVNULL,\
            stderr=subprocess.DEVNULL,\
            check=True,\
        )\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  ffmpeg conversion error: \{e\}")\
        is_speaking = False\
        return\
\
    try:\
        wave_file = wave.open(wav_path, 'rb')\
        audio_interface = pyaudio.PyAudio()\
\
        _, output_index = find_audio_devices()\
        if output_index is None:\
            print("\uc0\u10060  No speaker found.")\
            wave_file.close()\
            audio_interface.terminate()\
            is_speaking = False\
            return\
\
        print(f"\uc0\u55357 \u56586  Using output device index \{output_index\}")\
\
        output_stream = audio_interface.open(\
            format=audio_interface.get_format_from_width(wave_file.getsampwidth()),\
            channels=wave_file.getnchannels(),\
            rate=wave_file.getframerate(),\
            output=True,\
            output_device_index=output_index,\
        )\
\
        chunk_size = 1024\
        data = wave_file.readframes(chunk_size)\
\
        while data:\
            # If button turned OFF while speaking, abort and stop listening\
            button_on = not listen_button.value  # True when latched ON\
            if not button_on:\
                print("\uc0\u55357 \u56597  Button turned OFF \'97 stopping speech and listening.")\
                is_armed = False\
                break\
\
            output_stream.write(data)\
            data = wave_file.readframes(chunk_size)\
\
        output_stream.stop_stream()\
        output_stream.close()\
        audio_interface.terminate()\
        wave_file.close()\
\
    except Exception as e:\
        print(f"\uc0\u9888 \u65039  Playback error: \{e\}")\
\
    # Clean up files\
    if os.path.exists(mp3_path):\
        os.remove(mp3_path)\
    if os.path.exists(wav_path):\
        os.remove(wav_path)\
\
    is_speaking = False\
\
# ================================================================\
#  LED STATE\
# ================================================================\
def update_listen_led_state():\
    """\
    Simple LED logic:\
      - OFF when not armed\
      - ON when ready / thinking / speaking\
    """\
    global listen_led, is_armed\
\
    listen_led.value = bool(is_armed)\
\
# ================================================================\
#  MAIN LOOP\
# ================================================================\
def main():\
    global is_running, is_thinking, is_armed\
\
    print("\uc0\u55358 \u56800  Simple Chatbot (no servos, no NeoPixels) ready.")\
    time.sleep(1.0)\
\
    # Initialize armed state based on current button\
    button_on = not listen_button.value  # True when latched ON\
    is_armed = button_on\
    last_armed = button_on\
    update_listen_led_state()\
\
    if is_armed:\
        try:\
            speak_text("I'm ready. Ask me a question.")\
        except Exception:\
            pass\
\
    try:\
        while is_running:\
            # Poll button\
            button_on = not listen_button.value  # True when latched ON\
\
            # Edge detect\
            if button_on != last_armed:\
                if button_on:\
                    print("\uc0\u55357 \u56634  Button latched ON \'97 listening enabled.")\
                    is_armed = True\
                    try:\
                        speak_text("Okay, I'm listening.")\
                    except Exception:\
                        pass\
                else:\
                    print("\uc0\u55357 \u56635  Button switched OFF \'97 stopping listening.")\
                    is_armed = False\
                last_armed = button_on\
\
            update_listen_led_state()\
\
            # If not armed, just idle\
            if not is_armed:\
                time.sleep(0.05)\
                continue\
\
            # ====================================================\
            #  LISTEN\
            # ====================================================\
            print("\uc0\u55356 \u57252  Listening for speech...")\
            audio_path = record_audio()\
\
            if audio_path is None:\
                # Maybe button OFF or no audio; loop again\
                continue\
\
            is_thinking = True\
            update_listen_led_state()\
\
            user_text = transcribe_audio(audio_path)\
            try:\
                os.remove(audio_path)\
            except Exception:\
                pass\
\
            is_thinking = False\
            update_listen_led_state()\
\
            if not user_text:\
                print("\uc0\u9888 \u65039  Empty transcription.")\
                time.sleep(0.3)\
                continue\
\
            print(f"\uc0\u55358 \u56785  You said: \{user_text\}")\
\
            norm = user_text.lower().strip()\
\
            # Voice commands to stop/shutdown\
            if norm in \{"stop", "exit", "quit"\}:\
                print("\uc0\u55357 \u57041  Stop command received \'97 shutting down.")\
                speak_text("Okay, stopping now.")\
                is_running = False\
                break\
\
            # Ignore noisy / meaningless input\
            if not is_meaningful_text(user_text):\
                print("\uc0\u9888 \u65039  Transcription looks like noise; ignoring.")\
                time.sleep(0.5)\
                continue\
\
            # ====================================================\
            #  CHAT COMPLETION\
            # ====================================================\
            print("\uc0\u55358 \u56596  Thinking...")\
\
            try:\
                response = client.chat.completions.create(\
                    model=CHAT_MODEL,\
                    messages=[\
                        \{\
                            "role": "system",\
                            "content": (\
                                "You are a calm, expressive AI assistant. "\
                                "Respond concisely in 1 sentence unless more detail is truly necessary. "\
                                "Do NOT start with greetings like 'Hello', 'Hi', or 'How can I help you today?'. "\
                                "Just answer the user's request directly. "\
                                "Also output emotion as one of: happy, sad, neutral, angry, surprised. "\
                                "Format: <text> [emotion: <label>]"\
                            ),\
                        \},\
                        \{"role": "user", "content": user_text\},\
                    ],\
                )\
                is_offline = False\
            except APIConnectionError:\
                print("\uc0\u10060  No internet: cannot reach OpenAI for chat completion. Check Wi-Fi.")\
                is_offline = True\
                try:\
                    speak_text("I can't reach the internet right now.")\
                except Exception:\
                    pass\
                continue\
            except Exception as e:\
                print(f"\uc0\u9888 \u65039  Chat completion error: \{e\}")\
                continue\
\
            full_reply = response.choices[0].message.content.strip()\
\
            # Extract emotion tag\
            match = re.search(r"\\[emotion:\\s*(\\w+)\\]", full_reply, re.IGNORECASE)\
            emotion = match.group(1).lower() if match else "neutral"\
\
            # Remove emotion tag from spoken text\
            reply_text = re.sub(r"\\[emotion:.*\\]", "", full_reply).strip()\
\
            print(f"\uc0\u55358 \u56598  \{reply_text\}  [\{emotion\}]")\
            speak_text(reply_text)\
\
            # Loop back for the next turn\
\
    except KeyboardInterrupt:\
        print("\\n\uc0\u55357 \u56395  Program interrupted from keyboard.")\
        is_running = False\
\
    finally:\
        # Global shutdown cleanup\
        print("\uc0\u55357 \u56635  Shutting down...")\
        is_running = False\
        is_armed = False\
\
        try:\
            listen_led.value = False\
        except Exception:\
            pass\
\
        try:\
            listen_button.deinit()\
        except Exception:\
            pass\
\
        try:\
            listen_led.deinit()\
        except Exception:\
            pass\
\
        print("\uc0\u55357 \u56588  Chatbot shut down cleanly.")\
\
if __name__ == "__main__":\
    main()\
}