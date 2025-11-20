"""
Native VR viewer using PyOpenXR
Direct rendering to VR headsets without browser dependency
Persistent viewer with image update queue
"""

import sys
import os
import math
import ctypes  # FIXED: Added missing import
import threading
import queue
import time
import numpy as np
from pathlib import Path

try:
    import xr
    from xr.utils import Matrix4x4f, GraphicsAPI  # FIXED: Correct import location
    from xr.utils.gl import ContextObject
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider
    from OpenGL import GL
    from PIL import Image
    import cv2  # For video playback
    import glfw  # For keyboard input
    import pygame  # For audio playback
    import subprocess  # For ffmpeg audio extraction
    import tempfile
    PYOPENXR_AVAILABLE = True
except ImportError as e:
    PYOPENXR_AVAILABLE = False
    print(f"PyOpenXR not available. Install with: pip install pyopenxr PyOpenGL pillow opencv-python pygame")
    print(f"Error: {e}")


class GLFWVisibleContextProvider(GLFWOffscreenContextProvider):
    """
    GLFW context provider that creates a VISIBLE window for keyboard input.
    Extends the offscreen provider but makes the window visible.
    """
    def __init__(self):
        # Initialize GLFW if not already done
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Set window hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, True)  # Make window VISIBLE
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.FLOATING, True)  # Keep on top

        # Create a small visible window for controls
        # Use _window to match parent class attribute
        self._window = glfw.create_window(400, 300, "VR Video Controls", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        # Make context current
        glfw.make_context_current(self._window)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._window:
            glfw.destroy_window(self._window)
        glfw.terminate()


class StereoFormat:
    """Stereo format constants"""
    SIDE_BY_SIDE = "sbs"
    OVER_UNDER = "ou"
    ANAGLYPH = "anaglyph"
    MONO = "mono"
    SEPARATE = "separate"


class MediaUpdate:
    """Represents a media (image or video) update for the viewer"""
    def __init__(self, media_path, stereo_format, swap_eyes, projection_type="flat", screen_size=3.0, screen_distance=3.0, is_video=False, loop_video=True):
        self.media_path = media_path
        self.stereo_format = stereo_format
        self.swap_eyes = swap_eyes
        self.projection_type = projection_type
        self.screen_size = screen_size
        self.screen_distance = screen_distance
        self.is_video = is_video
        self.loop_video = loop_video


class PersistentNativeViewer:
    """
    Persistent VR viewer that stays running and can receive new images.
    Prevents multiple OpenXR instances and allows continuous updates.
    """

    def __init__(self):
        if not PYOPENXR_AVAILABLE:
            raise ImportError(
                "PyOpenXR is not available. Install with: pip install pyopenxr PyOpenGL pillow"
            )

        self.texture_id = None
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.sphere_vertices = None
        self.sphere_indices = None

        # Media update queue
        self.media_queue = queue.Queue()
        self.current_media = None
        self.current_format = StereoFormat.SIDE_BY_SIDE
        self.current_swap = False
        self.current_projection = "flat"
        self.current_screen_size = 3.0
        self.current_screen_distance = 3.0
        self.current_aspect_ratio = 16.0 / 9.0  # Default aspect ratio

        # Video playback state
        self.is_video = False
        self.video_capture = None
        self.video_playing = True
        self.video_loop = True
        self.video_fps = 30.0
        self.video_frame_time = 1.0 / 30.0
        self.last_frame_time = 0.0
        self.current_frame_number = 0
        self.total_frames = 0

        # Audio playback state
        self.audio_temp_file = None
        self.audio_initialized = False
        self.audio_paused = False

        # Viewer state
        self.running = False
        self.should_stop = False
        self.geometry_needs_update = False
        self.glfw_window = None  # Store GLFW window for keyboard input

        # Alignment offsets for real-time adjustment
        self.horizontal_offset = 0.0
        self.vertical_offset_adjustment = 0.0  # Additional offset beyond auto-centering

    def create_sphere_mesh(self, radius=100.0, segments=60, rings=40):
        """Create sphere geometry for 360° viewing"""
        vertices = []
        indices = []

        # Generate vertices
        for ring in range(rings + 1):
            theta = ring * math.pi / rings
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            for seg in range(segments + 1):
                phi = seg * 2 * math.pi / segments
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)

                # Position
                x = radius * sin_theta * cos_phi
                y = radius * cos_theta
                z = radius * sin_theta * sin_phi

                # UV coordinates (flipped for inside viewing)
                u = 1.0 - (seg / segments)
                v = ring / rings

                vertices.extend([x, y, z, u, v])

        # Generate indices
        for ring in range(rings):
            for seg in range(segments):
                first = ring * (segments + 1) + seg
                second = first + segments + 1

                # Two triangles per quad
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        self.sphere_vertices = np.array(vertices, dtype=np.float32)
        self.sphere_indices = np.array(indices, dtype=np.uint32)

    def create_flat_screen(self, width=3.0, height=2.25, distance=3.0):
        """Create flat screen geometry (like a cinema screen in VR)"""
        vertices = []
        indices = []

        # Calculate aspect ratio preserving dimensions
        half_width = width / 2.0
        half_height = height / 2.0

        # Create a simple quad facing the viewer
        # Position the screen at the given distance
        z = -distance

        # Vertical offset to center screen at comfortable viewing height
        # Offset upward by half height to center the screen at eye level
        # Plus additional user adjustment
        y_offset = half_height + self.vertical_offset_adjustment

        # Horizontal offset for user adjustment
        x_offset = self.horizontal_offset

        # Four corners of the screen (centered vertically and horizontally)
        positions = [
            [-half_width + x_offset, -half_height + y_offset, z],  # Bottom left
            [half_width + x_offset, -half_height + y_offset, z],   # Bottom right
            [half_width + x_offset, half_height + y_offset, z],    # Top right
            [-half_width + x_offset, half_height + y_offset, z],   # Top left
        ]

        # UV coordinates
        uvs = [
            [0.0, 1.0],  # Bottom left
            [1.0, 1.0],  # Bottom right
            [1.0, 0.0],  # Top right
            [0.0, 0.0],  # Top left
        ]

        # Build vertices
        for i in range(4):
            vertices.extend(positions[i])
            vertices.extend(uvs[i])

        # Two triangles to form the quad
        indices = [0, 1, 2, 0, 2, 3]

        self.sphere_vertices = np.array(vertices, dtype=np.float32)
        self.sphere_indices = np.array(indices, dtype=np.uint32)

    def create_curved_screen(self, width=3.0, height=2.25, distance=3.0, curve_amount=0.3):
        """Create curved screen geometry (gently curved like IMAX)"""
        vertices = []
        indices = []

        segments_h = 20  # Horizontal segments for curvature
        segments_v = 10  # Vertical segments

        half_height = height / 2.0

        # Vertical offset to center screen at comfortable eye level
        # Plus additional user adjustment
        y_offset = half_height + self.vertical_offset_adjustment

        # Horizontal offset for user adjustment
        x_offset = self.horizontal_offset

        for v in range(segments_v + 1):
            y = -half_height + (v / segments_v) * height + y_offset
            v_uv = 1.0 - (v / segments_v)

            for h in range(segments_h + 1):
                # Create curve using an arc
                angle = (h / segments_h - 0.5) * math.pi * curve_amount
                x = distance * math.sin(angle)
                z = -distance * math.cos(angle)

                # Scale x based on desired width
                x = x * (width / (2.0 * distance * math.sin(math.pi * curve_amount / 2.0)))

                # Apply horizontal offset
                x = x + x_offset

                u = h / segments_h

                vertices.extend([x, y, z, u, v_uv])

        # Generate indices
        for v in range(segments_v):
            for h in range(segments_h):
                first = v * (segments_h + 1) + h
                second = first + segments_h + 1

                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        self.sphere_vertices = np.array(vertices, dtype=np.float32)
        self.sphere_indices = np.array(indices, dtype=np.uint32)

    def create_dome_180(self, radius=10.0, segments=60):
        """Create 180° dome/hemisphere geometry"""
        vertices = []
        indices = []

        rings = segments // 2

        # Generate only the front hemisphere
        for ring in range(rings + 1):
            theta = ring * (math.pi / 2) / rings  # 0 to π/2 (front hemisphere)
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            for seg in range(segments + 1):
                phi = seg * math.pi / segments  # 0 to π (180 degrees horizontally)
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)

                # Position
                x = radius * sin_theta * sin_phi
                y = radius * cos_theta
                z = -radius * sin_theta * cos_phi

                # UV coordinates
                u = seg / segments
                v = ring / rings

                vertices.extend([x, y, z, u, v])

        # Generate indices
        for ring in range(rings):
            for seg in range(segments):
                first = ring * (segments + 1) + seg
                second = first + segments + 1

                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        self.sphere_vertices = np.array(vertices, dtype=np.float32)
        self.sphere_indices = np.array(indices, dtype=np.uint32)

    def create_geometry(self):
        """Create geometry based on current projection type"""
        if self.current_projection == "flat":
            # Use actual image aspect ratio instead of hardcoded 4:3
            height = self.current_screen_size / self.current_aspect_ratio
            self.create_flat_screen(
                width=self.current_screen_size,
                height=height,
                distance=self.current_screen_distance
            )
        elif self.current_projection == "curved":
            # Use actual image aspect ratio
            height = self.current_screen_size / self.current_aspect_ratio
            self.create_curved_screen(
                width=self.current_screen_size,
                height=height,
                distance=self.current_screen_distance,
                curve_amount=0.4
            )
        elif self.current_projection == "dome180":
            self.create_dome_180(radius=self.current_screen_distance * 2)
        else:  # sphere360
            self.create_sphere_mesh(radius=100.0)

    def create_shaders(self):
        """Create OpenGL shaders for rendering"""
        vertex_shader_source = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 texCoord;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        out vec2 TexCoord;

        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
            TexCoord = texCoord;
        }
        """

        fragment_shader_source = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;

        uniform sampler2D texture1;
        uniform int stereoFormat;
        uniform int eyeIndex;
        uniform bool swapEyes;

        void main() {
            vec2 uv = TexCoord;

            // Adjust UV based on stereo format
            if (stereoFormat == 0) {  // Side-by-Side
                uv.x = uv.x * 0.5;
                if (eyeIndex == 1) {
                    uv.x += 0.5;
                }
                if (swapEyes) {
                    uv.x = uv.x < 0.5 ? uv.x + 0.5 : uv.x - 0.5;
                }
            } else if (stereoFormat == 1) {  // Over-Under
                uv.y = uv.y * 0.5;
                if (eyeIndex == 1) {
                    uv.y += 0.5;
                }
                if (swapEyes) {
                    uv.y = uv.y < 0.5 ? uv.y + 0.5 : uv.y - 0.5;
                }
            }
            // Mono and anaglyph use full texture

            FragColor = texture(texture1, uv);
        }
        """

        # Compile vertex shader
        vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vertex_shader, vertex_shader_source)
        GL.glCompileShader(vertex_shader)

        if not GL.glGetShaderiv(vertex_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(vertex_shader).decode()
            raise RuntimeError(f"Vertex shader compilation failed: {error}")

        # Compile fragment shader
        fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fragment_shader, fragment_shader_source)
        GL.glCompileShader(fragment_shader)

        if not GL.glGetShaderiv(fragment_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(fragment_shader).decode()
            raise RuntimeError(f"Fragment shader compilation failed: {error}")

        # Link shader program
        self.shader_program = GL.glCreateProgram()
        GL.glAttachShader(self.shader_program, vertex_shader)
        GL.glAttachShader(self.shader_program, fragment_shader)
        GL.glLinkProgram(self.shader_program)

        if not GL.glGetProgramiv(self.shader_program, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(self.shader_program).decode()
            raise RuntimeError(f"Shader program linking failed: {error}")

        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)

    def load_video(self, video_path):
        """Load video file and initialize video capture"""
        try:
            print(f"   Loading video from: {video_path}")

            # Close previous video if exists
            if self.video_capture is not None:
                self.video_capture.release()

            # Open video file
            self.video_capture = cv2.VideoCapture(video_path)

            if not self.video_capture.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")

            # Get video properties
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.video_fps <= 0:
                self.video_fps = 30.0  # Default to 30fps if unknown

            self.video_frame_time = 1.0 / self.video_fps
            self.current_frame_number = 0
            self.last_frame_time = time.time()
            self.is_video = True

            print(f"   Video loaded: {width}x{height}, {self.video_fps} fps, {self.total_frames} frames")

            # Load audio track
            self.load_audio(video_path)

            # Load first frame
            ret, frame = self.video_capture.read()
            if ret:
                self.update_texture_from_frame(frame)

        except Exception as e:
            print(f"   ✗ Error loading video: {e}")
            raise

    def update_texture_from_frame(self, frame):
        """Update OpenGL texture with a video frame"""
        try:
            # OpenCV uses BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            height, width = frame_rgb.shape[:2]

            # Calculate aspect ratio from actual frame
            if self.current_format == StereoFormat.SIDE_BY_SIDE:
                aspect_ratio = (width / 2.0) / height
            elif self.current_format == StereoFormat.OVER_UNDER:
                aspect_ratio = width / (height / 2.0)
            else:
                aspect_ratio = width / height

            # Check if aspect ratio changed
            if abs(self.current_aspect_ratio - aspect_ratio) > 0.01:
                self.current_aspect_ratio = aspect_ratio
                if self.current_projection in ["flat", "curved"]:
                    self.geometry_needs_update = True

            if self.texture_id is None:
                self.texture_id = GL.glGenTextures(1)

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                width, height, 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, frame_rgb
            )

        except Exception as e:
            print(f"   ✗ Error updating texture from frame: {e}")

    def get_next_video_frame(self):
        """Get next frame from video, handling looping"""
        if not self.is_video or self.video_capture is None:
            return False

        ret, frame = self.video_capture.read()

        if ret:
            self.update_texture_from_frame(frame)
            self.current_frame_number += 1
            return True
        else:
            # End of video
            if self.video_loop:
                # Loop back to start
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_number = 0
                ret, frame = self.video_capture.read()
                if ret:
                    self.update_texture_from_frame(frame)
                    return True
            return False

    def seek_video(self, frames):
        """Seek video by number of frames (positive or negative)"""
        if not self.is_video or self.video_capture is None:
            return

        new_frame = max(0, min(self.current_frame_number + frames, self.total_frames - 1))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame_number = new_frame

        # Read and display the frame
        ret, frame = self.video_capture.read()
        if ret:
            self.update_texture_from_frame(frame)

        print(f"   Seeked to frame {self.current_frame_number}/{self.total_frames}")

    def restart_video(self):
        """Restart video from beginning"""
        if not self.is_video or self.video_capture is None:
            return

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_number = 0
        ret, frame = self.video_capture.read()
        if ret:
            self.update_texture_from_frame(frame)

        # Restart audio too
        if self.audio_initialized:
            pygame.mixer.music.rewind()
            if self.video_playing:
                pygame.mixer.music.unpause()

        print("   Video restarted")

    def load_audio(self, video_path):
        """Extract and load audio from video file using pygame"""
        try:
            # Initialize pygame mixer if not done
            if not self.audio_initialized:
                pygame.mixer.init()
                self.audio_initialized = True

            # Clean up previous audio temp file
            if self.audio_temp_file and os.path.exists(self.audio_temp_file):
                try:
                    os.remove(self.audio_temp_file)
                except:
                    pass

            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            self.audio_temp_file = temp_audio.name
            temp_audio.close()

            # Try to extract audio using ffmpeg (if available)
            try:
                subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-vn',  # No video
                    '-acodec', 'libmp3lame',  # MP3 codec
                    '-y',  # Overwrite
                    self.audio_temp_file
                ], capture_output=True, check=True, timeout=30)

                # Load audio with pygame
                pygame.mixer.music.load(self.audio_temp_file)
                pygame.mixer.music.play(loops=-1 if self.video_loop else 0)
                print("   ✓ Audio loaded and playing")

            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                print("   ⚠️  Could not extract audio (ffmpeg not found or no audio track)")
                self.audio_temp_file = None

        except Exception as e:
            print(f"   ⚠️  Audio loading failed: {e}")
            self.audio_temp_file = None

    def toggle_audio_playback(self):
        """Toggle audio play/pause"""
        if not self.audio_initialized:
            return

        if self.audio_paused:
            pygame.mixer.music.unpause()
            self.audio_paused = False
        else:
            pygame.mixer.music.pause()
            self.audio_paused = True

    def stop_audio(self):
        """Stop audio playback"""
        if self.audio_initialized:
            pygame.mixer.music.stop()

        # Clean up temp audio file
        if self.audio_temp_file and os.path.exists(self.audio_temp_file):
            try:
                os.remove(self.audio_temp_file)
            except:
                pass
            self.audio_temp_file = None

    def load_texture(self, image_path):
        """Load or update image as OpenGL texture"""
        try:
            print(f"   Loading texture from: {image_path}")

            # Mark as static image (not video)
            self.is_video = False
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None

            # Stop audio when switching to image
            self.stop_audio()

            img = Image.open(image_path)
            img = img.convert('RGB')
            img_data = np.array(img, dtype=np.uint8)

            print(f"   Image size: {img.width}x{img.height}, channels: {img_data.shape}")

            # Calculate aspect ratio from actual image
            # For stereo side-by-side, divide width by 2 to get per-eye aspect ratio
            if self.current_format == StereoFormat.SIDE_BY_SIDE:
                aspect_ratio = (img.width / 2.0) / img.height
            elif self.current_format == StereoFormat.OVER_UNDER:
                aspect_ratio = img.width / (img.height / 2.0)
            else:
                # Mono or anaglyph use full image
                aspect_ratio = img.width / img.height

            print(f"   Calculated aspect ratio: {aspect_ratio:.3f} ({img.width}x{img.height})")

            # Check if aspect ratio changed significantly (trigger geometry rebuild)
            if abs(self.current_aspect_ratio - aspect_ratio) > 0.01:
                print(f"   Aspect ratio changed: {self.current_aspect_ratio:.3f} → {aspect_ratio:.3f}")
                self.current_aspect_ratio = aspect_ratio
                # Only update geometry for flat/curved projections that respect aspect ratio
                if self.current_projection in ["flat", "curved"]:
                    self.geometry_needs_update = True

            if self.texture_id is None:
                # Create new texture
                self.texture_id = GL.glGenTextures(1)
                print(f"   Created texture ID: {self.texture_id}")

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                img.width, img.height, 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data
            )

            print(f"   ✓ Texture loaded successfully!")
        except Exception as e:
            print(f"   ✗ Error loading texture: {e}")
            raise

    def setup_geometry(self):
        """Set up VAO and VBO for geometry based on projection type"""
        self.create_geometry()

        # Create VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Create VBO
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self.sphere_vertices.nbytes,
            self.sphere_vertices,
            GL.GL_STATIC_DRAW
        )

        # Create EBO (Element Buffer Object)
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            self.sphere_indices.nbytes,
            self.sphere_indices,
            GL.GL_STATIC_DRAW
        )

        # Position attribute
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, None)
        GL.glEnableVertexAttribArray(0)

        # Texture coordinate attribute
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4,
            ctypes.c_void_p(3 * 4)
        )
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)

    def keyboard_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input for video controls and viewer adjustments"""
        if action != glfw.PRESS and action != glfw.REPEAT:
            return

        # Video playback controls
        if key == glfw.KEY_SPACE:
            # Toggle play/pause
            self.video_playing = not self.video_playing
            # Also toggle audio
            self.toggle_audio_playback()
            status = "Playing" if self.video_playing else "Paused"
            print(f"   Video {status}")
        elif key == glfw.KEY_R:
            # Restart video
            self.restart_video()
        elif key == glfw.KEY_LEFT and not (mods & glfw.MOD_SHIFT):
            # Seek backward (1 second)
            frames_to_seek = -int(self.video_fps)
            self.seek_video(frames_to_seek)
        elif key == glfw.KEY_RIGHT and not (mods & glfw.MOD_SHIFT):
            # Seek forward (1 second)
            frames_to_seek = int(self.video_fps)
            self.seek_video(frames_to_seek)
        elif key == glfw.KEY_DOWN and not (mods & glfw.MOD_SHIFT):
            # Seek backward (5 seconds)
            frames_to_seek = -int(self.video_fps * 5)
            self.seek_video(frames_to_seek)
        elif key == glfw.KEY_UP and not (mods & glfw.MOD_SHIFT):
            # Seek forward (5 seconds)
            frames_to_seek = int(self.video_fps * 5)
            self.seek_video(frames_to_seek)
        elif key == glfw.KEY_L:
            # Toggle loop
            self.video_loop = not self.video_loop
            status = "enabled" if self.video_loop else "disabled"
            print(f"   Video loop {status}")

        # Viewer adjustment controls
        elif key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
            # Quit viewer
            print("\n👋 Quitting VR viewer (ComfyUI continues running)...")
            self.should_stop = True

        elif key == glfw.KEY_P:
            # Cycle projection type
            projections = ["flat", "curved", "dome180", "sphere360"]
            current_idx = projections.index(self.current_projection) if self.current_projection in projections else 0
            next_idx = (current_idx + 1) % len(projections)
            self.current_projection = projections[next_idx]
            self.geometry_needs_update = True
            proj_names = {"flat": "Flat Screen", "curved": "Curved Screen", "dome180": "180° Dome", "sphere360": "360° Sphere"}
            print(f"   📐 Projection: {proj_names.get(self.current_projection, self.current_projection)}")

        elif key == glfw.KEY_LEFT_BRACKET:  # [
            # Decrease screen distance
            self.current_screen_distance = max(1.0, self.current_screen_distance - 0.5)
            self.geometry_needs_update = True
            print(f"   📏 Screen distance: {self.current_screen_distance:.1f}m")

        elif key == glfw.KEY_RIGHT_BRACKET:  # ]
            # Increase screen distance
            self.current_screen_distance = min(10.0, self.current_screen_distance + 0.5)
            self.geometry_needs_update = True
            print(f"   📏 Screen distance: {self.current_screen_distance:.1f}m")

        elif key == glfw.KEY_MINUS:
            # Decrease screen size
            self.current_screen_size = max(1.0, self.current_screen_size - 0.5)
            self.geometry_needs_update = True
            print(f"   📺 Screen size: {self.current_screen_size:.1f}m")

        elif key == glfw.KEY_EQUAL:  # = (same key as +)
            # Increase screen size
            self.current_screen_size = min(10.0, self.current_screen_size + 0.5)
            self.geometry_needs_update = True
            print(f"   📺 Screen size: {self.current_screen_size:.1f}m")

        elif key == glfw.KEY_S and (mods & glfw.MOD_SHIFT):
            # Cycle stereo format (Shift+S)
            formats = ["sbs", "ou", "mono"]
            format_names = {"sbs": "Side-by-Side", "ou": "Over-Under", "mono": "Mono"}
            current_idx = formats.index(self.current_format) if self.current_format in formats else 0
            next_idx = (current_idx + 1) % len(formats)
            self.current_format = formats[next_idx]
            self.geometry_needs_update = True  # May affect aspect ratio
            print(f"   🎬 Stereo format: {format_names.get(self.current_format, self.current_format)}")

        elif key == glfw.KEY_E:
            # Toggle swap eyes
            self.current_swap = not self.current_swap
            status = "ON" if self.current_swap else "OFF"
            print(f"   👁️  Swap eyes: {status}")

        elif key == glfw.KEY_W:
            # Move screen up
            self.vertical_offset_adjustment += 0.1
            self.geometry_needs_update = True
            print(f"   ⬆️  Vertical offset: {self.vertical_offset_adjustment:+.2f}m")

        elif key == glfw.KEY_S and not (mods & glfw.MOD_SHIFT):
            # Move screen down
            self.vertical_offset_adjustment -= 0.1
            self.geometry_needs_update = True
            print(f"   ⬇️  Vertical offset: {self.vertical_offset_adjustment:+.2f}m")

        elif key == glfw.KEY_A:
            # Move screen left
            self.horizontal_offset -= 0.1
            self.geometry_needs_update = True
            print(f"   ⬅️  Horizontal offset: {self.horizontal_offset:+.2f}m")

        elif key == glfw.KEY_D:
            # Move screen right
            self.horizontal_offset += 0.1
            self.geometry_needs_update = True
            print(f"   ➡️  Horizontal offset: {self.horizontal_offset:+.2f}m")

        elif key == glfw.KEY_0:
            # Reset all adjustments
            self.vertical_offset_adjustment = 0.0
            self.horizontal_offset = 0.0
            self.geometry_needs_update = True
            print(f"   🔄 Reset alignment offsets to center")

    def check_for_updates(self):
        """Check if there's a new media (image or video) to display"""
        try:
            while not self.media_queue.empty():
                update = self.media_queue.get_nowait()

                # Check if projection type or size changed (requires geometry rebuild)
                if (update.projection_type != self.current_projection or
                    update.screen_size != self.current_screen_size or
                    update.screen_distance != self.current_screen_distance):
                    print(f"\n🔄 Projection changed: {self.current_projection} → {update.projection_type}")
                    self.current_projection = update.projection_type
                    self.current_screen_size = update.screen_size
                    self.current_screen_distance = update.screen_distance
                    self.geometry_needs_update = True

                self.current_media = update.media_path
                self.current_format = update.stereo_format
                self.current_swap = update.swap_eyes
                self.video_loop = update.loop_video

                media_type = "video" if update.is_video else "image"
                print(f"\n📷 Updating VR view with new {media_type}: {update.media_path}")
                print(f"   Format: {update.stereo_format}, Swap: {update.swap_eyes}")
                print(f"   Projection: {update.projection_type}, Size: {update.screen_size}m, Distance: {update.screen_distance}m")

                # Load video or image
                if update.is_video:
                    self.load_video(self.current_media)
                    self.video_playing = True
                else:
                    self.load_texture(self.current_media)
        except queue.Empty:
            pass

    def run(self):
        """Main viewer loop - runs in background thread"""
        self.running = True
        self.should_stop = False

        print("\n" + "="*60)
        print("🥽 NATIVE VR VIEWER STARTING")
        print("="*60)
        print("PUT ON YOUR HEADSET NOW!")
        print("\n📖 KEYBOARD CONTROLS (focus control window):")
        print("="*60)
        print("\n🎬 VIDEO PLAYBACK:")
        print("  SPACE      - Play/Pause")
        print("  R          - Restart video")
        print("  LEFT/RIGHT - Seek backward/forward 1 second")
        print("  DOWN/UP    - Seek backward/forward 5 seconds")
        print("  L          - Toggle loop")
        print("\n📐 VIEWER ADJUSTMENTS:")
        print("  P          - Cycle projection type")
        print("  [ / ]      - Decrease/Increase screen distance")
        print("  - / =      - Decrease/Increase screen size")
        print("  Shift+S    - Cycle stereo format")
        print("  E          - Toggle swap eyes")
        print("\n🎯 ALIGNMENT:")
        print("  W / S      - Move screen up/down")
        print("  A / D      - Move screen left/right")
        print("  0          - Reset alignment to center")
        print("\n🚪 OTHER:")
        print("  Q or ESC   - Quit viewer (ComfyUI keeps running)")
        print("="*60 + "\n")

        # Map format strings to integers for shader
        format_map = {
            StereoFormat.SIDE_BY_SIDE: 0,
            StereoFormat.OVER_UNDER: 1,
            StereoFormat.ANAGLYPH: 2,
            StereoFormat.MONO: 2,
        }

        try:
            # Create visible context provider for keyboard input
            context_provider = GLFWVisibleContextProvider()

            with ContextObject(
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[
                        xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                    ],
                ),
                context_provider=context_provider,
            ) as context:

                # Get GLFW window and set up keyboard callback
                self.glfw_window = context_provider._window
                glfw.set_key_callback(self.glfw_window, self.keyboard_callback)
                print("✓ Keyboard controls enabled (focus the control window to use keys)")

                # Initialize OpenGL resources
                self.create_shaders()
                self.setup_geometry()

                # Load initial media if available
                if self.current_media:
                    if self.is_video:
                        self.load_video(self.current_media)
                    else:
                        self.load_texture(self.current_media)

                # Enable depth testing
                GL.glEnable(GL.GL_DEPTH_TEST)
                # Disable culling temporarily for debugging
                # GL.glEnable(GL.GL_CULL_FACE)
                # GL.glCullFace(GL.GL_FRONT)

                print("✓ VR session started successfully!")
                print("✓ Headset is ready for viewing")
                print(f"✓ Texture ID: {self.texture_id}")
                print(f"✓ Sphere vertices: {len(self.sphere_vertices) // 5}")
                print(f"✓ Sphere indices: {len(self.sphere_indices)}\n")

                frame_count = 0
                frames_rendered = 0

                for frame_index, frame_state in enumerate(context.frame_loop()):
                    # Check for stop signal
                    if self.should_stop:
                        print("\n🛑 Stopping VR viewer...")
                        break

                    # Check for media updates every few frames
                    if frame_count % 30 == 0:
                        self.check_for_updates()

                        # Rebuild geometry if projection type changed
                        if self.geometry_needs_update:
                            print("🔨 Rebuilding geometry...")
                            self.setup_geometry()
                            self.geometry_needs_update = False
                            print("✓ Geometry updated!")

                    # Poll keyboard events
                    if self.glfw_window:
                        glfw.poll_events()

                    # Advance video frame if playing
                    if self.is_video and self.video_playing:
                        current_time = time.time()
                        elapsed = current_time - self.last_frame_time

                        # Check if it's time for next frame
                        if elapsed >= self.video_frame_time:
                            self.get_next_video_frame()
                            self.last_frame_time = current_time

                    format_int = format_map.get(self.current_format, 0)

                    # Render to each eye
                    for view_index, view in enumerate(context.view_loop(frame_state)):

                        # Clear buffers to black
                        GL.glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
                        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

                        if self.texture_id is None:
                            # No image loaded yet, show black background
                            if frame_count == 0:
                                print("⚠️  Waiting for media to load...")
                            continue

                        # Debug: Print first frame render
                        if frames_rendered == 0:
                            print(f"🎬 Rendering first frame (eye {view_index})")

                        frames_rendered += 1

                        # Use shader
                        GL.glUseProgram(self.shader_program)

                        # Set up projection matrix
                        projection = Matrix4x4f.create_projection_fov(
                            graphics_api=GraphicsAPI.OPENGL,
                            fov=view.fov,
                            near_z=0.1,
                            far_z=1000.0,  # Increased for larger sphere
                        )

                        # Set up view matrix
                        to_view = Matrix4x4f.create_translation_rotation_scale(
                            translation=view.pose.position,
                            rotation=view.pose.orientation,
                            scale=(1, 1, 1),
                        )
                        view_matrix = Matrix4x4f.invert_rigid_body(to_view)

                        # Model matrix
                        model_matrix = np.eye(4, dtype=np.float32)

                        # Set uniforms
                        proj_loc = GL.glGetUniformLocation(self.shader_program, "projection")
                        view_loc = GL.glGetUniformLocation(self.shader_program, "view")
                        model_loc = GL.glGetUniformLocation(self.shader_program, "model")
                        format_loc = GL.glGetUniformLocation(self.shader_program, "stereoFormat")
                        eye_loc = GL.glGetUniformLocation(self.shader_program, "eyeIndex")
                        swap_loc = GL.glGetUniformLocation(self.shader_program, "swapEyes")

                        GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_FALSE, projection.as_numpy().flatten("F"))
                        GL.glUniformMatrix4fv(view_loc, 1, GL.GL_FALSE, view_matrix.as_numpy().flatten("F"))
                        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, model_matrix.flatten("F"))
                        GL.glUniform1i(format_loc, format_int)
                        GL.glUniform1i(eye_loc, view_index)
                        GL.glUniform1i(swap_loc, 1 if self.current_swap else 0)

                        # Bind texture
                        GL.glActiveTexture(GL.GL_TEXTURE0)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
                        GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "texture1"), 0)

                        # Draw sphere
                        GL.glBindVertexArray(self.vao)
                        GL.glDrawElements(
                            GL.GL_TRIANGLES,
                            len(self.sphere_indices),
                            GL.GL_UNSIGNED_INT,
                            None
                        )
                        GL.glBindVertexArray(0)

                    frame_count += 1

        except KeyboardInterrupt:
            # FIXED: Catch KeyboardInterrupt so it doesn't propagate to ComfyUI
            print("\n⚠️ Received interrupt signal (Ctrl+C)")
            print("Note: To stop the viewer, close ComfyUI instead")
        except Exception as e:
            print(f"\n❌ Error in VR viewer: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup video capture
            if self.video_capture is not None:
                self.video_capture.release()

            # Cleanup audio
            self.stop_audio()

            # Cleanup OpenGL resources
            # Wrap in try-except as OpenGL context may already be destroyed
            try:
                if self.texture_id:
                    GL.glDeleteTextures([self.texture_id])
                if self.vao:
                    GL.glDeleteVertexArrays(1, [self.vao])
                if self.vbo:
                    GL.glDeleteBuffers(1, [self.vbo])
                if self.ebo:
                    GL.glDeleteBuffers(1, [self.ebo])
                if self.shader_program:
                    GL.glDeleteProgram(self.shader_program)
            except Exception as cleanup_error:
                # Ignore cleanup errors (context likely already destroyed)
                pass

            self.running = False
            print("\n✓ VR viewer stopped cleanly")

    def stop(self):
        """Stop the viewer"""
        self.should_stop = True

    def update_media(self, media_path, stereo_format, swap_eyes, projection_type="flat", screen_size=3.0, screen_distance=3.0, is_video=False, loop_video=True):
        """Queue a new media (image or video) for display"""
        update = MediaUpdate(media_path, stereo_format, swap_eyes, projection_type, screen_size, screen_distance, is_video, loop_video)
        self.media_queue.put(update)


# Global persistent viewer instance
_global_viewer = None
_viewer_thread = None
_viewer_lock = threading.Lock()


def get_or_create_viewer():
    """Get existing viewer or create new one (singleton pattern)"""
    global _global_viewer, _viewer_thread

    with _viewer_lock:
        # If there's a viewer that's still running, return it
        if _global_viewer is not None and _global_viewer.running:
            return _global_viewer

        # If there's a thread that's still alive, wait for it to finish
        if _viewer_thread is not None and _viewer_thread.is_alive():
            print("⏳ Waiting for previous viewer instance to terminate...")
            _viewer_thread.join(timeout=5.0)
            if _viewer_thread.is_alive():
                print("⚠️  Previous viewer did not terminate cleanly")
                # Force create a new viewer anyway, might cause issues
            else:
                print("✓ Previous viewer terminated")

        # Create new viewer instance
        print("🔨 Creating new VR viewer instance...")
        _global_viewer = PersistentNativeViewer()
        _viewer_thread = threading.Thread(target=_global_viewer.run, daemon=True)
        _viewer_thread.start()

        # Give it a moment to initialize
        time.sleep(1.0)  # Increased from 0.5 to 1.0 for better initialization

        return _global_viewer


def stop_global_viewer():
    """Stop the global viewer"""
    global _global_viewer
    if _global_viewer:
        _global_viewer.stop()


def check_openxr_available():
    """Check if OpenXR runtime is available"""
    if not PYOPENXR_AVAILABLE:
        return False, "PyOpenXR not installed"

    try:
        # Try to enumerate available runtimes
        xr.enumerate_instance_extension_properties()
        return True, "OpenXR runtime available"
    except Exception as e:
        return False, f"OpenXR runtime not available: {str(e)}"


def launch_native_viewer(media_path, stereo_format="sbs", swap_eyes=False, projection_type="flat", screen_size=3.0, screen_distance=3.0, is_video=False, loop_video=True):
    """
    Launch or update the native viewer with a new image or video.
    If viewer is already running, updates it with the new media.
    Otherwise, starts a new viewer.

    Args:
        media_path: Path to stereo image or video
        stereo_format: Stereo format (sbs, ou, mono)
        swap_eyes: Whether to swap eyes
        projection_type: Projection type (flat, curved, dome180, sphere360)
        screen_size: Screen size in meters (for flat/curved/dome)
        screen_distance: Distance from viewer in meters
        is_video: Whether the media is a video file
        loop_video: Whether to loop video playback

    Returns:
        bool: True if successful, False if error
    """
    available, message = check_openxr_available()

    if not available:
        print(f"ERROR: {message}")
        print("\nTo use native VR viewer:")
        print("1. Install PyOpenXR: pip install pyopenxr PyOpenGL pillow opencv-python")
        print("2. Install SteamVR or Oculus runtime")
        print("3. Make sure your VR headset is connected")
        return False

    try:
        viewer = get_or_create_viewer()

        # Queue the new media
        viewer.update_media(media_path, stereo_format, swap_eyes, projection_type, screen_size, screen_distance, is_video, loop_video)

        # If this is the first media, set it as current
        if viewer.current_media is None:
            viewer.current_media = media_path
            viewer.current_format = stereo_format
            viewer.current_swap = swap_eyes
            viewer.current_projection = projection_type
            viewer.current_screen_size = screen_size
            viewer.current_screen_distance = screen_distance
            viewer.is_video = is_video
            viewer.video_loop = loop_video

        return True

    except Exception as e:
        print(f"Error launching native viewer: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the viewer
    if len(sys.argv) < 2:
        print("Usage: python native_viewer.py <image_path> [stereo_format] [swap_eyes]")
        print("Example: python native_viewer.py test.png sbs false")
        sys.exit(1)

    image_path = sys.argv[1]
    stereo_format = sys.argv[2] if len(sys.argv) > 2 else "sbs"
    swap_eyes = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False

    success = launch_native_viewer(image_path, stereo_format, swap_eyes)

    if success:
        print("\nViewer is running. Press Enter to exit...")
        input()
        stop_global_viewer()
