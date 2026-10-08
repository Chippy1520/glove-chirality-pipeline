"""Frozen worker entrypoint; no UI toolkit or model weights embedded here."""
from glove_chirality.desktop_worker import main

if __name__ == "__main__":
    main()
