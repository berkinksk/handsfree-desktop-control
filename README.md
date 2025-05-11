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

## Download Pretrained Models

This project relies on two OpenCV model files for face and landmark detection:

1. Haar Cascade for face detection:
   models/haarcascade_frontalface_default.xml
   Download via PowerShell:
   ```powershell
   Invoke-WebRequest \
     -Uri "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" \
     -OutFile "models\haarcascade_frontalface_default.xml"
   ```

2. LBF Facemark model for landmark detection:
   models/lbfmodel.yaml
   Download via PowerShell:
   ```powershell
   Invoke-WebRequest \
     -Uri "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml" \
     -OutFile "models\lbfmodel.yaml"
   ```

Make sure the `models/` directory exists and contains these files before running the app or tests.
