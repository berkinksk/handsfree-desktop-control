"""
Stubbed vision backend.

class HeadEyeDetector:
    detect_head_pose(frame) -> str
        returns one of {"left","right","up","down","center"}
    detect_blink(frame) -> bool
        True if (placeholder) blink detected
"""
class HeadEyeDetector:
    """Detect head pose and blinks from video frames."""
    def __init__(self):
        pass

    def detect_head_pose(self, frame):
        """Return head pose as one of: left, right, up, down, center."""
        return "center"  # TODO replace with real logic

    def detect_blink(self, frame):
        """Return True if placeholder blink detected."""
        return False  # TODO replace with real logic 