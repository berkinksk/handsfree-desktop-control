# Hands-Free Desktop Control

**Graduation Project — Eastern Mediterranean University**  
· Berkin Kaynar · Cıvan Deniz Doğan · Orçun Altınel  
Supervisor: Assoc. Prof. Dr. Adnan Acan

Hands-Free Desktop Control lets physically disabled users operate a desktop computer **without a mouse or keyboard**.  
A webcam tracks **head orientation** to move the cursor, and detects **eyelid blinks** to perform clicks.  
The system is implemented in Python, using OpenCV for real-time video processing, TensorFlow/Keras CNNs for enhanced gesture recognition (in development), PyAutoGUI for OS-level cursor control, a PyQt5 GUI for the interface, and SQLite for persistent settings.

## How to Use

### For End-Users (Recommended)
The easiest way to use the application is to download the pre-built executable from the **[Releases](https://github.com/berkinksk/handsfree-desktop-control/releases)** page on GitHub. No installation is required.

### For Developers

**1. Setup the Environment**
```bash
# Clone the repository and navigate into it
git clone https://github.com/berkinksk/handsfree-desktop-control.git
cd handsfree-desktop-control

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project in editable mode to handles all dependencies and paths)
pip install -e .
```

**2. Run the Application**
```bash
# To launch the GUI application
python app.py
```

**3. Build the Executable (Optional)**
If you want to build the `.exe` file from source, use the following command after setting up the environment. The final executable will be located in the `dist/` directory.
```bash
pyinstaller --onefile --windowed --name "HandsfreeDesktopControl" --icon=src/frontend/assets/sun.png --add-data ".venv/Lib/site-packages/mediapipe;mediapipe" --add-data "src/frontend/assets;assets" app.py
```

## Repository Layout

- `src/backend/` → HeadEyeDetector (head pose & blink detection logic)
- `src/integration/` → HeadEyeController (calibration, cursor control loop, DB interaction)
- `src/frontend/` → PyQt5 GUI (user interface and visual feedback)
- `app.py` → Program entry point (launches GUI and ties components together)
- `setup.py` → Project packaging script (makes the project installable)
- `db/` → SQLite database schema and default data
- `tests/` → PyTest tests (smoke tests, unit tests for logic)

## Branching Model

- **main** – Protected stable branch (demo-ready code).
- **feature/backend-cnn** – Vision & machine learning development (this is where we implement head pose, blink detection, and later CNN integration).
- **feature/integration-system** – System integration development (cursor control, calibration, tying backend to frontend).
- **feature/frontend-gui** – User interface development (building the PyQt5 GUI).

Develop on a feature branch and open a Pull Request to merge into `main` after review:

1. Pull the latest `main`, create or switch to your feature branch.
2. Write code; keep commits focused and use Conventional Commit messages (`feat: ...`, `fix: ...`).
3. Ensure `pytest` passes and the app runs (`python app.py` to sanity-check the GUI).
4. Push and open a PR; get at least one teammate to review and approve.
5. Address any review feedback, then merge.
6. Sync your branch with `main` periodically to minimize conflicts.

## Current Progress and Roadmap

- **Milestone M0** ✅ – Project Scaffold: Basic project structure is set up with stub classes and a simple GUI window. Imports and environment have been tested.
- **Milestone M1** 🛠️ – OpenCV-based Head Control Prototype: The OpenCV pipeline for head pose is now implemented (face detection + landmark-based pose estimation). The cursor control loop and blink-click mechanism are being developed. A developer can test head movement recognition using the provided script. (Blink detection is still basic and will be improved.)
- **Milestone M2** ⌛ – CNN Integration: Coming next, we will introduce trained CNN models to improve head pose accuracy and robust blink detection. This will likely involve training models (using TensorFlow/Keras) on a dataset of face images for various poses and eye states. The models will be integrated into `HeadEyeDetector` (or a subclass) to augment or replace the classical approach when ready. Target accuracies: ≥95% for head pose classification, ≥90% for blink detection (as per project requirements).
- **Milestone M3** ⌛ – Full GUI and Settings: Develop the complete PyQt5 interface, including a calibration wizard (to personalize what is "center" for a user and possibly adjust sensitivity), settings for things like click activation delay, and visual feedback (e.g., showing a pointer or overlay on camera feed). The GUI will also display connection status (camera on/off) and allow the user to start/stop the control.
- **Milestone M4** ⏳ – Testing and Performance Tuning: Rigorously test the system with real users. Optimize performance (ensure low latency, consider using threads so GUI remains responsive while processing video frames). Fine-tune any thresholds or model parameters based on user feedback. Prepare the final report and demonstration. If time permits, add any nice-to-have features or polish.

See the repository's Issues and Project Board for a breakdown of tasks and progress. Each feature branch corresponds to a set of issues/user stories tracked there.

## License

This project is released under the MIT License. See LICENSE for details.
