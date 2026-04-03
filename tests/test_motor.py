"""Manual hardware test — run on the Pi to verify the stepper motor dispenses correctly."""

from staystation.motor import Motor


def main() -> None:
    motor = Motor()
    print("Press Enter to dispense a treat. Ctrl+C to exit.")
    try:
        while True:
            input()
            print("Dispensing...")
            motor.dispense()
            print("Done.")
    except KeyboardInterrupt:
        pass
    finally:
        motor.cleanup()


if __name__ == "__main__":
    main()
