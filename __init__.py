"""
ComfyStereoViewer - Stereoscopic 3D viewer for ComfyUI
Provides VR headset support via native PyOpenXR
"""

# Import native nodes (requires PyOpenXR)
try:
    from .native_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NATIVE_AVAILABLE = True
except ImportError as e:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    NATIVE_AVAILABLE = False
    print("PyOpenXR not available. Native VR viewer nodes disabled.")
    print("Install with: pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw pillow")
    print(f"Error: {e}")

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
