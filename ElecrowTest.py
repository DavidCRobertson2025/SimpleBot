import math
import time
import numpy as np
import pygame

def make_tone(freq_hz=440.0, duration_s=0.4, sample_rate=44100, volume=0.25):
    """Return a pygame Sound containing a sine tone."""
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    wave = (volume * np.sin(2.0 * math.pi * freq_hz * t)).astype(np.float32)

    # Convert to 16-bit stereo
    audio = (wave * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(stereo)

def main():
    pygame.init()

    # ---- AUDIO ----
    # Some systems need mixer init before display; do it early.
    # If this fails, we'll keep going and just skip audio.
    sound = None
    audio_ok = False
    try:
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        sound = make_tone()
        audio_ok = True
    except Exception as e:
        print("[audio] mixer init failed:", e)

    # ---- VIDEO ----
    # Fullscreen on the active display (your HDMI panel should be the active output).
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    pygame.display.set_caption("HDMI + Touch + Audio Test")

    font_big = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 28)

    clock = pygame.time.Clock()
    start = time.time()

    last_touch = None
    taps = 0

    # Play a tone on startup
    if audio_ok and sound:
        sound.play()

    running = True
    while running:
        # ---- EVENTS (touch usually appears as mouse events) ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ESC to quit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            if event.type == pygame.MOUSEMOTION:
                last_touch = event.pos

            if event.type == pygame.MOUSEBUTTONDOWN:
                taps += 1
                last_touch = event.pos
                # Short click tone on tap
                if audio_ok and sound:
                    sound.play()

        # ---- DRAW ----
        t = time.time() - start

        # Animated background: moving stripes
        screen.fill((0, 0, 0))
        stripe_w = 40
        offset = int((t * 120) % (stripe_w * 2))
        for x in range(-stripe_w * 2, w + stripe_w * 2, stripe_w * 2):
            pygame.draw.rect(screen, (40, 40, 40), (x + offset, 0, stripe_w, h))

        # Crosshair at last touch/mouse
        if last_touch:
            x, y = last_touch
            pygame.draw.line(screen, (0, 255, 0), (x - 25, y), (x + 25, y), 3)
            pygame.draw.line(screen, (0, 255, 0), (x, y - 25), (x, y + 25), 3)

        title = font_big.render("HDMI + Touch + Audio Test", True, (255, 255, 255))
        screen.blit(title, (20, 20))

        info_lines = [
            f"Resolution: {w} x {h}",
            f"Touch/mouse position: {last_touch if last_touch else '(move/finger tap to show)'}",
            f"Taps detected: {taps}",
            f"Audio: {'OK (tone plays on start + tap)' if audio_ok else 'NOT OK (mixer init failed)'}",
            "Press ESC to quit.",
        ]

        y0 = 90
        for line in info_lines:
            surf = font_small.render(line, True, (255, 255, 255))
            screen.blit(surf, (20, y0))
            y0 += 34

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
