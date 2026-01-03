import time
import board

USE_PI5 = True

# ---- CHANGE THESE ----
NUM_PIXELS = 16
PIN = board.D13   # GPIO13 / physical pin 33

# Try "GRB" first (most WS2812 rings), if colors look wrong try "BRG" or "RGB"
BYTEORDER = "GRB"   # options: "GRB", "BRG", "RGB"
# ----------------------

def clamp(x): 
    return max(0, min(255, int(x)))

def scale(color, brightness):
    r, g, b = color
    return (clamp(r * brightness), clamp(g * brightness), clamp(b * brightness))

def make_pixels():
    if USE_PI5:
        import adafruit_pixelbuf
        from adafruit_raspberry_pi5_neopixel_write import neopixel_write

        class Pi5PixelBuf(adafruit_pixelbuf.PixelBuf):
            def __init__(self, pin, size, **kwargs):
                self._pin = pin
                super().__init__(size=size, **kwargs)

            def _transmit(self, buf):
                neopixel_write(self._pin, buf)

        return Pi5PixelBuf(
            PIN,
            NUM_PIXELS,
            auto_write=True,
            byteorder=BYTEORDER,
        )
    else:
        import neopixel
        return neopixel.NeoPixel(
            PIN,
            NUM_PIXELS,
            auto_write=True,
            pixel_order=getattr(neopixel, BYTEORDER, neopixel.GRB),
        )

def fill(pixels, color, brightness=0.15):
    pixels.fill(scale(color, brightness))
    pixels.show()

def wipe(pixels, color, brightness=0.2, delay=0.05):
    pixels.fill((0,0,0))
    pixels.show()
    c = scale(color, brightness)
    for i in range(NUM_PIXELS):
        pixels[i] = c
        pixels.show()
        time.sleep(delay)

def chase(pixels, color, brightness=0.2, delay=0.05, cycles=3):
    pixels.fill((0,0,0))
    pixels.show()
    c = scale(color, brightness)
    for _ in range(cycles * NUM_PIXELS):
        i = _ % NUM_PIXELS
        pixels.fill((0,0,0))
        pixels[i] = c
        pixels.show()
        time.sleep(delay)

def brightness_ramp(pixels, color, steps=20):
    for k in range(steps + 1):
        b = k / steps * 0.4  # cap at 40% to reduce brownouts on Pi 5V
        fill(pixels, color, brightness=b)
        time.sleep(0.05)
    for k in range(steps, -1, -1):
        b = k / steps * 0.4
        fill(pixels, color, brightness=b)
        time.sleep(0.05)

def main():
    pixels = make_pixels()
    pixels.fill((0,0,0))
    pixels.show()

    print(f"NeoPixel test: pin={PIN} count={NUM_PIXELS} byteorder={BYTEORDER}")
    print("ESC/Ctrl+C to quit.\n")

    try:
        # Solid fills (confirm all 16 light, confirm order)
        print("Solid red")
        fill(pixels, (255,0,0), brightness=0.12); time.sleep(3)
        print("Solid green")
        fill(pixels, (0,255,0), brightness=0.12); time.sleep(3)
        print("Solid blue")
        fill(pixels, (0,0,255), brightness=0.12); time.sleep(3)

        # Wipes (confirm indexing)
        print("Wipe white")
        wipe(pixels, (255,255,255), brightness=0.10, delay=0.04); time.sleep(0.5)

        # Chase
        print("Chase cyan")
        chase(pixels, (0,255,255), brightness=0.15, delay=0.05, cycles=2); time.sleep(0.3)

        # Brightness ramp (power stability)
        print("Brightness ramp (blue)")
        brightness_ramp(pixels, (0,0,255), steps=25); time.sleep(0.5)

        print("Done. Leaving LEDs off.")
        pixels.fill((0,0,0)); pixels.show()

    except KeyboardInterrupt:
        pixels.fill((0,0,0)); pixels.show()
        print("\nStopped; LEDs off.")

if __name__ == "__main__":
    main()