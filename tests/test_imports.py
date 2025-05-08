def test_module_imports():
    """Smoke test to ensure modules import without errors."""
    import importlib
    for module in ['backend.detector', 'integration.controller', 'frontend.gui']:
        importlib.import_module(module)

    # Add more imports as needed 