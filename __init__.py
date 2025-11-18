"""
ComfyStereoViewer - Stereoscopic 3D viewer for ComfyUI
Provides VR headset support via WebXR and native PyOpenXR
"""

from .nodes import NODE_CLASS_MAPPINGS as WEB_NODES, NODE_DISPLAY_NAME_MAPPINGS as WEB_NAMES

# Try to import native nodes (optional - requires PyOpenXR)
try:
    from .native_nodes import NODE_CLASS_MAPPINGS as NATIVE_NODES, NODE_DISPLAY_NAME_MAPPINGS as NATIVE_NAMES
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_NODES = {}
    NATIVE_NAMES = {}
    NATIVE_AVAILABLE = False
    print("PyOpenXR not available. Native VR viewer nodes disabled.")
    print("Install with: pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw")

# Combine all nodes
NODE_CLASS_MAPPINGS = {**WEB_NODES, **NATIVE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**WEB_NAMES, **NATIVE_NAMES}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Web directory for the viewer interface
WEB_DIRECTORY = "./web"
