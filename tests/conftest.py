import sys
import os

# Add src directory to sys.path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))) 