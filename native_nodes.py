"""
Native VR viewer nodes for ComfyUI using PyOpenXR
Auto-launches directly into VR headset without browser
"""

import torch
import numpy as np
from PIL import Image
import io
import os
import json
import folder_paths
import threading
import subprocess
import sys

from .native_viewer import (
    check_openxr_available,
    launch_native_viewer,
    StereoFormat,
    PYOPENXR_AVAILABLE
)


class NativeStereoImageViewer:
    """
    Native VR stereo image viewer using PyOpenXR.
    Auto-launches directly into VR headset without browser.
    Requires OpenXR runtime (SteamVR, Oculus, WMR).
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = "native_stereo_view"

    @classmethod
    def INPUT_TYPES(cls):
        # Check if OpenXR is available
        available, message = check_openxr_available()

        return {
            "required": {
                "image": ("IMAGE",),
                "stereo_format": (["Side-by-Side", "Over-Under", "Anaglyph", "Mono"],),
                "swap_eyes": ("BOOLEAN", {"default": False}),
                "auto_launch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "right_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "view_stereo_native"
    OUTPUT_NODE = True
    CATEGORY = "stereo/native"

    def view_stereo_native(self, image, stereo_format, swap_eyes=False, auto_launch=True, right_image=None):
        """
        View stereo image in VR headset using native PyOpenXR viewer.
        Auto-launches directly into headset.
        """
        # Check if PyOpenXR is available
        available, message = check_openxr_available()

        if not available:
            print(f"\n{'='*60}")
            print("NATIVE VR VIEWER NOT AVAILABLE")
            print(f"{'='*60}")
            print(f"Reason: {message}")
            print("\nTo enable native VR viewing:")
            print("1. Install PyOpenXR dependencies:")
            print("   pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw")
            print("2. Install a VR runtime:")
            print("   - SteamVR (recommended, supports most headsets)")
            print("   - Oculus Runtime (for Oculus headsets)")
            print("   - Windows Mixed Reality (built into Windows)")
            print("3. Connect your VR headset")
            print(f"{'='*60}\n")
            return (image,)

        # Convert tensor to numpy array
        if isinstance(image, torch.Tensor):
            img_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = image

        # Handle separate L/R images
        if right_image is not None:
            if isinstance(right_image, torch.Tensor):
                right_np = (right_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            else:
                right_np = right_image

            # Create side-by-side layout from separate images
            if swap_eyes:
                sbs_img = np.concatenate([right_np, img_np], axis=1)
            else:
                sbs_img = np.concatenate([img_np, right_np], axis=1)

            img_np = sbs_img
            stereo_format = "Side-by-Side"

        # Save the stereo image
        filename = f"{self.prefix_append}_{hash(str(img_np.tobytes()))}.png"
        filepath = os.path.join(self.output_dir, filename)

        pil_img = Image.fromarray(img_np)
        pil_img.save(filepath)

        print(f"\n{'='*60}")
        print("NATIVE VR VIEWER")
        print(f"{'='*60}")
        print(f"Image saved: {filepath}")
        print(f"Stereo format: {stereo_format}")
        print(f"Swap eyes: {swap_eyes}")

        # Map format to internal format strings
        format_map = {
            "Side-by-Side": StereoFormat.SIDE_BY_SIDE,
            "Over-Under": StereoFormat.OVER_UNDER,
            "Anaglyph": StereoFormat.ANAGLYPH,
            "Mono": StereoFormat.MONO,
        }

        internal_format = format_map.get(stereo_format, StereoFormat.SIDE_BY_SIDE)

        if auto_launch:
            print("\nLaunching VR viewer...")
            print("PUT ON YOUR HEADSET NOW!")
            print("Press Ctrl+C in the console to exit VR mode")
            print(f"{'='*60}\n")

            # Launch in a separate thread to avoid blocking ComfyUI
            def launch_thread():
                try:
                    launch_native_viewer(filepath, internal_format, swap_eyes)
                except Exception as e:
                    print(f"Error in VR viewer: {e}")
                    import traceback
                    traceback.print_exc()

            # Launch in background thread
            thread = threading.Thread(target=launch_thread, daemon=True)
            thread.start()

            print("VR viewer launched in background.")
            print("The viewer will continue running until you press Ctrl+C")
        else:
            print("\nAuto-launch disabled.")
            print("To manually launch, run:")
            print(f"python -m ComfyStereoViewer.native_viewer {filepath} {internal_format} {swap_eyes}")
            print(f"{'='*60}\n")

        # Return original image as passthrough
        return (image,)


class NativeVRStatus:
    """
    Check if native VR viewing is available and show status.
    Useful for debugging VR setup issues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status_message", "is_available")
    FUNCTION = "check_status"
    OUTPUT_NODE = True
    CATEGORY = "stereo/native"

    def check_status(self):
        """Check native VR viewer availability"""
        available, message = check_openxr_available()

        status_lines = [
            "="*60,
            "NATIVE VR VIEWER STATUS",
            "="*60,
        ]

        if available:
            status_lines.extend([
                "✓ PyOpenXR: Installed",
                "✓ OpenXR Runtime: Available",
                "✓ Status: Ready for VR viewing",
                "",
                "Detected runtime: " + message,
                "",
                "You can use native VR viewer nodes!",
            ])
        else:
            status_lines.extend([
                "✗ Status: Not available",
                "✗ Reason: " + message,
                "",
                "To enable native VR viewing:",
                "1. Install dependencies:",
                "   pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw",
                "2. Install a VR runtime:",
                "   - SteamVR (recommended)",
                "   - Oculus Runtime",
                "   - Windows Mixed Reality",
                "3. Connect your VR headset and start the runtime",
            ])

        status_lines.append("="*60)

        status_text = "\n".join(status_lines)
        print(status_text)

        return (status_text, available)


class HybridStereoImageViewer:
    """
    Hybrid stereo viewer that automatically chooses the best viewing method:
    - Native PyOpenXR viewer if available (best performance, auto-launch)
    - WebXR browser viewer as fallback (maximum compatibility)
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = "hybrid_stereo_view"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stereo_format": (["Side-by-Side", "Over-Under", "Anaglyph", "Mono"],),
                "swap_eyes": ("BOOLEAN", {"default": False}),
                "prefer_native": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "right_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("passthrough", "viewer_type")
    FUNCTION = "view_stereo_hybrid"
    OUTPUT_NODE = True
    CATEGORY = "stereo"

    def view_stereo_hybrid(self, image, stereo_format, swap_eyes=False, prefer_native=True, right_image=None):
        """
        Automatically select best viewer: native PyOpenXR or WebXR fallback
        """
        # Check if native viewer is available
        native_available, message = check_openxr_available()

        viewer_type = "unknown"

        if native_available and prefer_native:
            # Use native viewer
            print("\n✓ Using NATIVE PyOpenXR viewer (best performance)")
            from .native_nodes import NativeStereoImageViewer
            viewer = NativeStereoImageViewer()
            result = viewer.view_stereo_native(
                image=image,
                stereo_format=stereo_format,
                swap_eyes=swap_eyes,
                auto_launch=True,
                right_image=right_image
            )
            viewer_type = "native_pyopenxr"
        else:
            # Fall back to WebXR viewer
            if not native_available:
                print(f"\n→ Native viewer not available: {message}")
            print("→ Falling back to WebXR browser viewer")
            from .nodes import StereoImageViewer
            viewer = StereoImageViewer()
            result = viewer.view_stereo(
                image=image,
                stereo_format=stereo_format,
                swap_eyes=swap_eyes,
                right_image=right_image
            )
            viewer_type = "webxr_browser"

        return (result[0], viewer_type)


# Node class mappings for native nodes
NODE_CLASS_MAPPINGS = {
    "NativeStereoImageViewer": NativeStereoImageViewer,
    "NativeVRStatus": NativeVRStatus,
    "HybridStereoImageViewer": HybridStereoImageViewer,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeStereoImageViewer": "Native Stereo Viewer (PyOpenXR)",
    "NativeVRStatus": "Check Native VR Status",
    "HybridStereoImageViewer": "Hybrid Stereo Viewer (Auto)",
}
