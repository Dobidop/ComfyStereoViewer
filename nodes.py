"""
ComfyStereoViewer - Custom nodes for stereoscopic 3D viewing in ComfyUI
Supports VR headsets via WebXR and multiple stereo formats
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import json
import folder_paths
import server


class StereoImageViewer:
    """
    A node for viewing stereoscopic 3D images with VR headset support.
    Supports multiple stereo formats: Side-by-Side, Over-Under, Anaglyph, and Separate L/R.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = "stereo_view"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stereo_format": (["Side-by-Side", "Over-Under", "Anaglyph", "Left Only", "Right Only"],),
                "swap_eyes": ("BOOLEAN", {"default": False}),
                "ipd_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "right_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "view_stereo"
    OUTPUT_NODE = True
    CATEGORY = "stereo"

    def view_stereo(self, image, stereo_format, swap_eyes=False, ipd_scale=1.0, right_image=None):
        """
        Process and prepare stereo images for viewing.
        Returns the original image as passthrough.
        """
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
            height = img_np.shape[0]
            width = img_np.shape[1]

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

        # Create metadata for the viewer
        metadata = {
            "filename": filename,
            "filepath": filepath,
            "stereo_format": stereo_format,
            "swap_eyes": swap_eyes,
            "ipd_scale": ipd_scale,
            "width": img_np.shape[1],
            "height": img_np.shape[0],
            "type": "image"
        }

        # Store metadata for web viewer
        metadata_file = os.path.join(self.output_dir, f"{filename}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Return original image as passthrough
        return (image,)


class StereoVideoViewer:
    """
    A node for viewing stereoscopic 3D videos with VR headset support.
    Supports video files in Side-by-Side and Over-Under formats.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "stereo_format": (["Side-by-Side", "Over-Under"],),
                "swap_eyes": ("BOOLEAN", {"default": False}),
                "ipd_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "view_stereo_video"
    OUTPUT_NODE = True
    CATEGORY = "stereo"

    def view_stereo_video(self, video_path, stereo_format, swap_eyes=False, ipd_scale=1.0):
        """
        Process and prepare stereo videos for viewing.
        Returns the video path.
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Create metadata for the viewer
        metadata = {
            "video_path": video_path,
            "stereo_format": stereo_format,
            "swap_eyes": swap_eyes,
            "ipd_scale": ipd_scale,
            "type": "video"
        }

        # Store metadata for web viewer
        filename = os.path.basename(video_path)
        metadata_file = os.path.join(self.output_dir, f"{filename}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        return (video_path,)


class StereoCombine:
    """
    Combines two separate images into a stereo pair (SBS or OU format).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
                "layout": (["Side-by-Side", "Over-Under"],),
                "swap_eyes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine_stereo"
    CATEGORY = "stereo"

    def combine_stereo(self, left_image, right_image, layout, swap_eyes=False):
        """
        Combine left and right images into a single stereo image.
        """
        # Convert tensors to numpy arrays
        left_np = (left_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        right_np = (right_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        # Swap if requested
        if swap_eyes:
            left_np, right_np = right_np, left_np

        # Combine based on layout
        if layout == "Side-by-Side":
            combined = np.concatenate([left_np, right_np], axis=1)
        else:  # Over-Under
            combined = np.concatenate([left_np, right_np], axis=0)

        # Convert back to tensor
        combined_tensor = torch.from_numpy(combined.astype(np.float32) / 255.0).unsqueeze(0)

        return (combined_tensor,)


class StereoSplit:
    """
    Splits a stereo image (SBS or OU) into separate left and right images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stereo_image": ("IMAGE",),
                "layout": (["Side-by-Side", "Over-Under"],),
                "swap_eyes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("left_image", "right_image")
    FUNCTION = "split_stereo"
    CATEGORY = "stereo"

    def split_stereo(self, stereo_image, layout, swap_eyes=False):
        """
        Split a stereo image into left and right components.
        """
        # Convert tensor to numpy array
        img_np = stereo_image.squeeze(0).cpu().numpy()

        # Split based on layout
        if layout == "Side-by-Side":
            width = img_np.shape[1] // 2
            left = img_np[:, :width, :]
            right = img_np[:, width:, :]
        else:  # Over-Under
            height = img_np.shape[0] // 2
            left = img_np[:height, :, :]
            right = img_np[height:, :, :]

        # Swap if requested
        if swap_eyes:
            left, right = right, left

        # Convert back to tensors
        left_tensor = torch.from_numpy(left).unsqueeze(0)
        right_tensor = torch.from_numpy(right).unsqueeze(0)

        return (left_tensor, right_tensor)


class StereoAnaglyph:
    """
    Creates red-cyan anaglyph 3D images from stereo pairs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
                "anaglyph_type": (["Red-Cyan", "Green-Magenta", "Amber-Blue"],),
                "swap_eyes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_anaglyph"
    CATEGORY = "stereo"

    def create_anaglyph(self, left_image, right_image, anaglyph_type, swap_eyes=False):
        """
        Create an anaglyph image from left and right images.
        """
        # Convert tensors to numpy arrays
        left_np = left_image.squeeze(0).cpu().numpy()
        right_np = right_image.squeeze(0).cpu().numpy()

        # Swap if requested
        if swap_eyes:
            left_np, right_np = right_np, left_np

        # Create anaglyph based on type
        anaglyph = np.zeros_like(left_np)

        if anaglyph_type == "Red-Cyan":
            # Red channel from left, green and blue from right
            anaglyph[:, :, 0] = left_np[:, :, 0]
            anaglyph[:, :, 1] = right_np[:, :, 1]
            anaglyph[:, :, 2] = right_np[:, :, 2]
        elif anaglyph_type == "Green-Magenta":
            # Green from left, red and blue from right
            anaglyph[:, :, 0] = right_np[:, :, 0]
            anaglyph[:, :, 1] = left_np[:, :, 1]
            anaglyph[:, :, 2] = right_np[:, :, 2]
        else:  # Amber-Blue
            # Red and green from left, blue from right
            anaglyph[:, :, 0] = left_np[:, :, 0]
            anaglyph[:, :, 1] = left_np[:, :, 1]
            anaglyph[:, :, 2] = right_np[:, :, 2]

        # Convert back to tensor
        anaglyph_tensor = torch.from_numpy(anaglyph).unsqueeze(0)

        return (anaglyph_tensor,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "StereoImageViewer": StereoImageViewer,
    "StereoVideoViewer": StereoVideoViewer,
    "StereoCombine": StereoCombine,
    "StereoSplit": StereoSplit,
    "StereoAnaglyph": StereoAnaglyph,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoImageViewer": "Stereo Image Viewer (VR)",
    "StereoVideoViewer": "Stereo Video Viewer (VR)",
    "StereoCombine": "Combine Stereo Images",
    "StereoSplit": "Split Stereo Image",
    "StereoAnaglyph": "Create Anaglyph 3D",
}
