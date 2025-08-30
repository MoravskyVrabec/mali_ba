"""
Visualization and user interface components for Mali-Ba.
"""

# Export the main visualizer class from the core file
from .visualizer import BoardVisualizer

# Export other UI components if they are needed externally
from .gui_other import (
    InteractiveObject,
    InteractiveObjectManager,
    ControlPanel,
    Sidebar
    # Add DialogBox if you integrate it later
)

# Do NOT export the drawing or parsing functions unless specifically needed elsewhere
# from .visualizer_drawing import ...
# from .visualizer_parsing import ...

# --- END OF FILE mali_ba/ui/__init__.py ---