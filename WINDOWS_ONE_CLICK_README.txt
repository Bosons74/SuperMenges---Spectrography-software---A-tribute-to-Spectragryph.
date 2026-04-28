OpenSpectra Workbench V1.9 — Windows one-click launch
=======================================================

Recommended file to double-click:

    START_OPENSPECTRA_WINDOWS.bat

What it does:
1. Finds Python.
2. Creates a local .venv environment if it does not exist.
3. Installs required packages from requirements.txt.
4. Launches OpenSpectra Workbench.

After the first launch, you can also use:

    START_OPENSPECTRA_FAST.bat

This second launcher starts faster, but it assumes the .venv environment has already been created.

Important:
- Install Python 3.10 or newer first.
- During Python installation, tick:
      Add python.exe to PATH
- The first launch needs internet access to download dependencies.

If Windows SmartScreen warns you:
- This is a plain text .bat file, not an unknown executable.
- Right-click > Edit to inspect it before running if needed.
