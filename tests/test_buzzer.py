"""Manual hardware test — run on the Pi to verify the buzzer plays a melody."""

import argparse

from staystation.buzzer import Buzzer, mario_coin

# Note frequencies (Hz)
C4 = 261
G3 = 196
A3 = 220
B3 = 247

# Melody: sequence of (frequency, duration_ms) tuples; frequency=0 is a rest
melody = [
    (C4, 400),
    (G3, 200),
    (G3, 200),
    (A3, 400),
    (G3, 400),
    (0, 400),  # rest
    (B3, 400),
    (C4, 400),
]

PAUSE_MS = 300


def play_melody(buzzer: Buzzer) -> None:
    for freq, duration_ms in melody:
        duration = duration_ms / 1000.0
        if freq == 0:
            buzzer.silence(duration)
        else:
            buzzer.tone(freq, duration)
        buzzer.silence(PAUSE_MS / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Buzzer test")
    parser.add_argument("sound", choices=["melody", "coin"], help="Sound to play")
    args = parser.parse_args()

    buzzer = Buzzer()
    try:
        if args.sound == "melody":
            play_melody(buzzer)
        else:
            mario_coin(buzzer)
        buzzer.off()
    finally:
        buzzer.cleanup()


if __name__ == "__main__":
    main()
