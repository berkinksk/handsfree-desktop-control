"""
This module contains basic smoke tests to ensure that the main components
of the application can be imported correctly after installation.
"""

def test_backend_import():
    """Tests that the core detector class can be imported."""
    from backend.detector import HeadEyeDetector
    assert HeadEyeDetector is not None

def test_frontend_import():
    """Tests that the GUI launch function can be imported."""
    from frontend.gui import launch_app
    assert launch_app is not None

def test_integration_import():
    """Tests that the main controller class can be imported."""
    from integration.controller import HeadEyeController
    assert HeadEyeController is not None