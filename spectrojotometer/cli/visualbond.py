"""
visualbond: graphical interface for spectrojotometer.
"""
import sys

from spectrojotometer.gui import ApplicationGUI


def main():
    """Open the main window; returns when it is closed."""
    ApplicationGUI()
    return 0


if __name__ == "__main__":
    sys.exit(main())
