"""
Hands-Free Desktop Control – Modern PyQt5 GUI
=============================================
• Rounded "card" panels with drop-shadows
• Animated flat buttons
• Live webcam preview (fills the video card, no black bars)
• Animated moon/sun toggle switch for dark ↔ light mode
"""

import sys
import math
import cv2
import os
from PyQt5 import QtCore, QtGui, QtWidgets

# ─────────────────────────────────────────────────────────────────────────────
#  Backend stub – replace with the real HeadEyeController in your project
# ─────────────────────────────────────────────────────────────────────────────
try:
    from integration.controller import HeadEyeController  # type: ignore
except ModuleNotFoundError:
    class HeadEyeController(QtCore.QObject):
        """Fallback stub so GUI runs even if backend isn't available."""
        blink_detected = QtCore.pyqtSignal()

        def __init__(self):
            super().__init__()
            self.running = False

        def calibrate(self) -> bool:
            return True

        def start_control(self):
            self.running = True

        def stop_control(self):
            self.running = False


# ════════════════════════════════════════════════════════════════════════════
#  Re-usable flat button with colour-fade hover animation
# ════════════════════════════════════════════════════════════════════════════
class AnimatedButton(QtWidgets.QPushButton):
    def __init__(
        self,
        text: str = "",
        *,
        base_color: QtGui.QColor = QtGui.QColor("#C0C0C0"),
        hover_factor: int = 120,
        radius: int = 5,
        parent=None,
    ):
        super().__init__(text, parent)
        self.base_color = base_color
        self.hover_color = base_color.lighter(hover_factor)
        self._current = self.base_color
        self._radius = radius

        self._apply_style()

        shadow = QtWidgets.QGraphicsDropShadowEffect(
            self, blurRadius=15, offset=QtCore.QPointF(0, 3)
        )
        shadow.setColor(QtGui.QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

        self._anim = QtCore.QPropertyAnimation(self, b"bgColor", self, duration=200)

    # ‒‒‒ Styling helpers ‒‒‒
    def _apply_style(self):
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self._current.name()};
                color: white;
                border: none;
                border-radius: {self._radius}px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            """
        )

    # ‒‒‒ Hover handling ‒‒‒
    def enterEvent(self, e):  # noqa: N802
        self._fade_to(self.hover_color)
        super().enterEvent(e)

    def leaveEvent(self, e):  # noqa: N802
        self._fade_to(self.base_color)
        super().leaveEvent(e)

    def _fade_to(self, colour: QtGui.QColor):
        self._anim.stop()
        self._anim.setStartValue(self._current)
        self._anim.setEndValue(colour)
        self._anim.start()

    # ‒‒‒ Property for animation ‒‒‒
    def _get_bg(self):
        return self._current

    def _set_bg(self, colour):
        if isinstance(colour, QtGui.QColor):
            self._current = colour
            self._apply_style()

    bgColor = QtCore.pyqtProperty(QtGui.QColor, fget=_get_bg, fset=_set_bg)  # type: ignore


# ════════════════════════════════════════════════════════════════════════════
#  Animated moon / sun toggle switch (dark ↔ light mode)
# ════════════════════════════════════════════════════════════════════════════
class ToggleSwitch(QtWidgets.QAbstractButton):
    """A pill-shaped toggle with a sliding knob and moon / sun icons."""

    def __init__(self, *, checked=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setChecked(checked)

        # 0.0 → knob at left (light mode) • 1.0 → right (dark mode)
        self._offset = 1.0 if self.isChecked() else 0.0

        self._anim = QtCore.QPropertyAnimation(
            self, b"offset", self, duration=200, easingCurve=QtCore.QEasingCurve.InOutQuad
        )

        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        # Dynamic track colors
        self.track_off = QtGui.QColor(160, 160, 160)
        self.track_on = QtGui.QColor("#0078d7")
        self.knob_color = QtGui.QColor(245, 245, 245)

    # ‒‒‒ Sizing ‒‒‒
    def sizeHint(self):
        return QtCore.QSize(64, 28)

    # ‒‒‒ Animatable property ‒‒‒
    def _get_offset(self):
        return self._offset

    def _set_offset(self, value):
        self._offset = value
        self.update()

    offset = QtCore.pyqtProperty(float, _get_offset, _set_offset)  # type: ignore

    # ‒‒‒ Interaction ‒‒‒
    def mouseReleaseEvent(self, e):  # noqa: N802
        if e.button() == QtCore.Qt.LeftButton:
            # Animate knob from current position to new position after toggle
            old_offset = self._offset
            super().mouseReleaseEvent(e)
            new_offset = 1.0 if self.isChecked() else 0.0
            self._anim.stop()
            self._anim.setStartValue(old_offset)
            self._anim.setEndValue(new_offset)
            self._anim.start()
        else:
            super().mouseReleaseEvent(e)

    # ‒‒‒ Painting ‒‒‒
    def paintEvent(self, _):  # noqa: N802
        w, h = self.width(), self.height()
        knob_r = h * 0.46
        track_rect = QtCore.QRectF(0, 0, w, h)
        track_r = h / 2

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # Colours
        track_off = QtGui.QColor(160, 160, 160)
        track_on = QtGui.QColor("#0078d7")
        knob_col = QtGui.QColor(245, 245, 245)
        moon_bright, moon_dim = QtGui.QColor(255, 255, 255), QtGui.QColor(120, 120, 120)
        sun_bright, sun_dim = QtGui.QColor("#ffcb3d"), QtGui.QColor(120, 120, 120)

        # Track
        p.setBrush(self.track_on if self.isChecked() else self.track_off)
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(track_rect, track_r, track_r)

        # Knob position
        x_knob = (w - 2 * knob_r - 4) * self._offset + 2
        knob_c = QtCore.QPointF(x_knob + knob_r, h / 2)

        # Knob
        p.setBrush(self.knob_color)
        p.drawEllipse(knob_c, knob_r, knob_r)

        # Icons
        p.setPen(moon_bright if not self.isChecked() else moon_dim)
        self._draw_moon(p, QtCore.QPointF(track_rect.left() - h * 0.8, h / 2), h * 0.22)

        p.setPen(sun_bright if self.isChecked() else sun_dim)
        self._draw_sun(p, QtCore.QPointF(track_rect.right() + h * 0.8, h / 2), h * 0.22)

    # ‒‒‒ Icon helpers ‒‒‒
    @staticmethod
    def _draw_moon(p: QtGui.QPainter, c: QtCore.QPointF, r: float):
        base = QtGui.QPainterPath()
        base.addEllipse(c, r, r)
        cut = QtGui.QPainterPath()
        cut.addEllipse(c + QtCore.QPointF(r * 0.6, 0), r, r)
        p.setBrush(p.pen().color())
        p.drawPath(base.subtracted(cut))

    @staticmethod
    def _draw_sun(p: QtGui.QPainter, c: QtCore.QPointF, r: float):
        p.setBrush(p.pen().color())
        p.drawEllipse(c, r * 0.7, r * 0.7)
        for i in range(8):
            angle = math.radians(i * 45)
            inner = QtCore.QPointF(c.x() + r * 0.9 * math.cos(angle),
                                   c.y() + r * 0.9 * math.sin(angle))
            outer = QtCore.QPointF(c.x() + r * 1.35 * math.cos(angle),
                                   c.y() + r * 1.35 * math.sin(angle))
            p.setPen(QtGui.QPen(p.pen().color(), 1.2))
            p.drawLine(inner, outer)


# ════════════════════════════════════════════════════════════════════════════
#  Main application window
# ════════════════════════════════════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow):

    FRAME_MS = 30  # ~33 FPS webcam update interval

    def __init__(self, controller: HeadEyeController):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Hands-Free Desktop Control")
        self.resize(1100, 650)

        # ───────────── Central layout ─────────────
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Video card (left)
        self.video_card = self._round_frame("#000000")
        self._add_shadow(self.video_card, radius=20, alpha=160)
        vbox = QtWidgets.QVBoxLayout(self.video_card)
        vbox.setContentsMargins(6, 6, 6, 6)
        self.video_lbl = QtWidgets.QLabel("Camera Off", alignment=QtCore.Qt.AlignCenter)
        self.video_lbl.setStyleSheet("color: gray;")
        self.video_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        vbox.addWidget(self.video_lbl)

        # Control card (right)
        self.ctrl = self._round_frame("#f0f0f0")
        self.ctrl.setMinimumWidth(260)
        self._add_shadow(self.ctrl, radius=15, alpha=120)
        cbox = QtWidgets.QVBoxLayout(self.ctrl)
        cbox.setContentsMargins(18, 18, 18, 18)
        cbox.setSpacing(14)

        # Labels
        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        self.track_lbl = QtWidgets.QLabel("Tracking: Off")
        self.blink_lbl = QtWidgets.QLabel("Blink: N/A")
        for l in (self.status_lbl, self.track_lbl, self.blink_lbl):
            l.setStyleSheet("font-weight: 500;")
            cbox.addWidget(l)

        cbox.addSpacing(4)

        # Slider
        cbox.addWidget(QtWidgets.QLabel("Cursor Sensitivity:"))
        self.sens_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=1, maximum=10, value=5)
        self._style_slider(self.sens_slider)
        cbox.addWidget(self.sens_slider)

        # Buttons
        self.calib_btn = AnimatedButton("Calibrate")
        self.calib_btn.clicked.connect(self.calibrate)
        cbox.addWidget(self.calib_btn)

        self.start_btn = AnimatedButton("Start")
        self.start_btn.clicked.connect(self.toggle_start)
        cbox.addWidget(self.start_btn)

        cbox.addStretch(1)

        # Theme toggle with sun/moon icons
        toggle_layout = QtWidgets.QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")

        # Sun icon on left
        sun_path = os.path.join(assets_dir, "sun.png")
        sun_pix = QtGui.QPixmap(sun_path).scaled(24, 24, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        sun_label = QtWidgets.QLabel()
        sun_label.setPixmap(sun_pix)
        sun_label.setContentsMargins(50, 0, 0, 0)
        toggle_layout.addWidget(sun_label)

        # Toggle switch
        self.theme_switch = ToggleSwitch(checked=False)
        self.theme_switch.toggled.connect(self.apply_theme)
        toggle_layout.addWidget(self.theme_switch)

        # Moon icon on right
        moon_path = os.path.join(assets_dir, "moon.png")
        moon_pix = QtGui.QPixmap(moon_path).scaled(24, 20, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        moon_label = QtWidgets.QLabel()
        moon_label.setPixmap(moon_pix)
        moon_label.setContentsMargins(5, 0, 0, 0)
        toggle_layout.addWidget(moon_label)

        cbox.addLayout(toggle_layout)

        # Assemble main layout
        root.addWidget(self.video_card, 1)
        root.addWidget(self.ctrl, 0, QtCore.Qt.AlignTop)

        # Initial palette: apply dark mode by default (toggle unchecked)
        self.apply_theme(self.theme_switch.isChecked())

        # Backend ↔ GUI signals
        self.controller.blink_detected.connect(self.flash_blink)

        # Webcam timer
        self.cap = None
        self.timer = QtCore.QTimer(self, timeout=self.update_frame)

    # ‒‒‒ Helper widgets / styles ‒‒‒
    def _round_frame(self, color: str) -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setStyleSheet(f"background-color:{color};border-radius:12px;")
        return f

    @staticmethod
    def _add_shadow(widget, *, radius: int, alpha: int):
        eff = QtWidgets.QGraphicsDropShadowEffect(
            widget, blurRadius=radius, offset=QtCore.QPointF(0, 0)
        )
        eff.setColor(QtGui.QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(eff)

    @staticmethod
    def _style_slider(slider):
        accent = "#0078d7"
        slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                height:6px; background:#ccc; border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{accent}; width:14px; height:14px;
                margin:-4px 0; border-radius:7px;
            }}"""
        )

    # ‒‒‒ Palette handling ‒‒‒
    def apply_dark_palette(self):
        app = QtWidgets.QApplication.instance()
        QtWidgets.QApplication.setStyle("Fusion")
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(40, 40, 40))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor(40, 40, 40))
        pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#0078d7"))
        pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
        app.setPalette(pal)
        self.ctrl.setStyleSheet(
            "background:#2c2c2c;border-radius:12px;")  # labels inherit palette

    def apply_light_palette(self):
        app = QtWidgets.QApplication.instance()
        QtWidgets.QApplication.setStyle("Fusion")
        app.setPalette(app.style().standardPalette())
        self.ctrl.setStyleSheet("background:#f0f0f0;border-radius:12px;")

    @QtCore.pyqtSlot(bool)
    def apply_theme(self, dark_enabled: bool):
        if dark_enabled:
            self.apply_dark_palette()
            # Change toggle track color on state ON to dark grey
            self.theme_switch.track_on = QtGui.QColor("#A9A9A9")
            self.theme_switch.update()
        else:
            self.apply_light_palette()
            # Reset toggle track ON color to default blue
            self.theme_switch.track_on = QtGui.QColor("#0078d7")
            self.theme_switch.update()

    # ‒‒‒ Button slots ‒‒‒
    def calibrate(self):
        self.status_lbl.setText("Status: Calibrating…")
        QtWidgets.QApplication.processEvents()
        success = self.controller.calibrate()
        self.status_lbl.setText("Status: Calibrated" if success else "Status: Calib. Failed")

    def toggle_start(self):
        if not self.controller.running:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.critical(self, "Camera Error", "Unable to open webcam.")
                self.cap = None
                return
            self.controller.start_control()
            self.controller.running = True
            self.timer.start(self.FRAME_MS)
            self.status_lbl.setText("Status: Running")
            self.track_lbl.setText("Tracking: On")
            self.start_btn.setText("Stop")
        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
            self.controller.stop_control()
            self.controller.running = False
            self.video_lbl.clear()
            self.video_lbl.setText("Camera Off")
            self.status_lbl.setText("Status: Stopped")
            self.track_lbl.setText("Tracking: Off")
            self.start_btn.setText("Start")

    def flash_blink(self):
        self.blink_lbl.setText("Blink: Detected!")
        QtCore.QTimer.singleShot(600, lambda: self.blink_lbl.setText("Blink: N/A"))

    # ‒‒‒ Webcam update loop ‒‒‒
    def update_frame(self):
        if not (self.cap and self.cap.isOpened()):
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        lbl_w, lbl_h = self.video_lbl.width(), self.video_lbl.height()

        # Scale to COVER and crop centre (removes black bars)
        pixmap = QtGui.QPixmap.fromImage(
            QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        )
        scaled = pixmap.scaled(
            lbl_w,
            lbl_h,
            QtCore.Qt.KeepAspectRatioByExpanding,
            QtCore.Qt.SmoothTransformation,
        )
        x = (scaled.width() - lbl_w) // 2
        y = (scaled.height() - lbl_h) // 2
        self.video_lbl.setPixmap(scaled.copy(x, y, lbl_w, lbl_h))

    # ‒‒‒ Cleanup ‒‒‒
    def closeEvent(self, e):  # noqa: N802
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(e)


# ════════════════════════════════════════════════════════════════════════════
def launch_app():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(HeadEyeController())
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_app()   