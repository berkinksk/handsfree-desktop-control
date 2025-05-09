# Hands-Free Desktop Control

**Graduation Project — Eastern Mediterranean University**  
· Berkin Kaynar · Cıvan Deniz Doğan · Orçun Altınel  
Supervisor: Assoc. Prof. Dr. Adnan Acan

Hands-Free Desktop Control lets physically disabled users operate a desktop computer **without a mouse or keyboard**.  
A webcam tracks **head orientation** to move the cursor and detects **eyelid blinks** to perform clicks.  
The system is implemented in Python, using OpenCV for real-time video, TensorFlow/Keras CNNs for gesture recognition, PyAutoGUI for OS control, a PyQt5 GUI, and SQLite for persistent settings.

## Quick start (development)

```bash
# 1. clone and create a virtual environment
git clone https://github.com/<berkinksk>/handsfree-desktop-control.git
cd handsfree-desktop-control
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. install minimal dependencies
pip install -r requirements.txt

# 3. smoke-test scaffold
python src/main.py     # opens a stub GUI window
pytest                 # imports all modules


```

**Note:** heavy ML libraries (tensorflow, dlib, etc.) are added later by the backend branch; install them only when you work on that branch.

## Repository layout

```
src/
  backend/        → HeadEyeDetector (CNN / OpenCV)
  integration/    → HeadEyeController (calibration, cursor loop, DB)
  frontend/       → PyQt5 GUI
  main.py         → program entry point
db/               → SQLite schema and future database files
tests/            → pytest smoke/logic tests
```

## Branching model

- `main` – protected, stable demo-ready code (PR required)
- `feature/backend-cnn` – vision & ML
- `feature/integration-system` – calibration, PyAutoGUI, DB
- `feature/frontend-gui` – user interface

Develop on your feature branch, open a Pull Request, and request at least one teammate review before merging into `main`.

## How to contribute (team workflow)

1. Pull latest `main`, create or switch to your feature branch.
2. Write code; keep commits small and use Conventional Commit messages (`feat: …`, `fix: …`).
3. `pytest` must pass locally; run `python src/main.py` to ensure the GUI still opens.
4. Push and open a PR; address review comments, then merge.
5. Sync your branch with `main` regularly to minimise conflicts.

## Roadmap (milestones)

- M0 ✅ Project scaffold & import smoke-test
- M1 Basic OpenCV head/blink detection & cursor control prototype
- M2 CNN models integrated — accuracy ≥ 95 % / ≥ 90 %
- M3 Full PyQt5 GUI with settings & calibration dialog
- M4 Performance tuning, user testing, final report

See the issues and project board for detailed tasks.

## License

Released under the MIT License — see LICENSE.


Deno cok tatlı.