"""
ComfyStereoViewer - Stereoscopic 3D viewer for ComfyUI
Provides VR headset support via WebXR and multiple stereo viewing modes
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Web directory for the viewer interface
WEB_DIRECTORY = "./web"
