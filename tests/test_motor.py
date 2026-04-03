"""Manual hardware test — run on the Pi to verify the stepper motor dispenses correctly."""

from staystation.motor import Motor


def main() -> None:
    motor = Motor()
    try:
        print("Dispensing one treat...")
        motor.dispense()
        print("Done.")
    finally:
        motor.cleanup()


if __name__ == "__main__":
    main()
