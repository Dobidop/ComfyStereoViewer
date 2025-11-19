"""
Native VR viewer using PyOpenXR
Direct rendering to VR headsets without browser dependency
Persistent viewer with image update queue
"""

import sys
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
    PYOPENXR_AVAILABLE = True
except ImportError:
    PYOPENXR_AVAILABLE = False
    print("PyOpenXR not available. Install with: pip install pyopenxr PyOpenGL pillow")


class StereoFormat:
    """Stereo format constants"""
    SIDE_BY_SIDE = "sbs"
    OVER_UNDER = "ou"
    ANAGLYPH = "anaglyph"
    MONO = "mono"
    SEPARATE = "separate"


class ImageUpdate:
    """Represents an image update for the viewer"""
    def __init__(self, image_path, stereo_format, swap_eyes):
        self.image_path = image_path
        self.stereo_format = stereo_format
        self.swap_eyes = swap_eyes


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

        # Image update queue
        self.image_queue = queue.Queue()
        self.current_image = None
        self.current_format = StereoFormat.SIDE_BY_SIDE
        self.current_swap = False

        # Viewer state
        self.running = False
        self.should_stop = False

    def create_sphere_mesh(self, radius=10.0, segments=60, rings=40):
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

    def load_texture(self, image_path):
        """Load or update image as OpenGL texture"""
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_data = np.array(img, dtype=np.uint8)

        if self.texture_id is None:
            # Create new texture
            self.texture_id = GL.glGenTextures(1)

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

    def setup_geometry(self):
        """Set up VAO and VBO for sphere mesh"""
        self.create_sphere_mesh()

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

    def check_for_updates(self):
        """Check if there's a new image to display"""
        try:
            while not self.image_queue.empty():
                update = self.image_queue.get_nowait()
                self.current_image = update.image_path
                self.current_format = update.stereo_format
                self.current_swap = update.swap_eyes
                print(f"\n📷 Updating VR view with new image: {update.image_path}")
                print(f"   Format: {update.stereo_format}, Swap: {update.swap_eyes}")
                # Reload texture with new image
                self.load_texture(self.current_image)
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
        print("\nControls:")
        print("  - Look around naturally with your headset")
        print("  - Run the workflow again to update the image")
        print("  - Close ComfyUI to stop the viewer")
        print("="*60 + "\n")

        # Map format strings to integers for shader
        format_map = {
            StereoFormat.SIDE_BY_SIDE: 0,
            StereoFormat.OVER_UNDER: 1,
            StereoFormat.ANAGLYPH: 2,
            StereoFormat.MONO: 2,
        }

        try:
            with ContextObject(
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[
                        xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                    ],
                ),
                context_provider=GLFWOffscreenContextProvider(),
            ) as context:

                # Initialize OpenGL resources
                self.create_shaders()
                self.setup_geometry()

                # Load initial image if available
                if self.current_image:
                    self.load_texture(self.current_image)

                # Enable depth testing
                GL.glEnable(GL.GL_DEPTH_TEST)
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glCullFace(GL.GL_FRONT)

                print("✓ VR session started successfully!")
                print("✓ Headset is ready for viewing\n")

                frame_count = 0

                for frame_index, frame_state in enumerate(context.frame_loop()):
                    # Check for stop signal
                    if self.should_stop:
                        print("\n🛑 Stopping VR viewer...")
                        break

                    # Check for image updates every few frames
                    if frame_count % 30 == 0:
                        self.check_for_updates()

                    format_int = format_map.get(self.current_format, 0)

                    # Render to each eye
                    for view_index, view in enumerate(context.view_loop(frame_state)):

                        # Clear buffers
                        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
                        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

                        if self.texture_id is None:
                            # No image loaded yet, show black
                            continue

                        # Use shader
                        GL.glUseProgram(self.shader_program)

                        # Set up projection matrix
                        projection = Matrix4x4f.create_projection_fov(
                            graphics_api=GraphicsAPI.OPENGL,
                            fov=view.fov,
                            near_z=0.1,
                            far_z=100.0,
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

    def update_image(self, image_path, stereo_format, swap_eyes):
        """Queue a new image for display"""
        update = ImageUpdate(image_path, stereo_format, swap_eyes)
        self.image_queue.put(update)


# Global persistent viewer instance
_global_viewer = None
_viewer_thread = None
_viewer_lock = threading.Lock()


def get_or_create_viewer():
    """Get existing viewer or create new one (singleton pattern)"""
    global _global_viewer, _viewer_thread

    with _viewer_lock:
        if _global_viewer is None or not _global_viewer.running:
            _global_viewer = PersistentNativeViewer()
            _viewer_thread = threading.Thread(target=_global_viewer.run, daemon=True)
            _viewer_thread.start()
            # Give it a moment to initialize
            time.sleep(0.5)

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


def launch_native_viewer(image_path, stereo_format="sbs", swap_eyes=False):
    """
    Launch or update the native viewer with a new image.
    If viewer is already running, updates it with the new image.
    Otherwise, starts a new viewer.

    Args:
        image_path: Path to stereo image
        stereo_format: Stereo format (sbs, ou, anaglyph, mono)
        swap_eyes: Whether to swap eyes

    Returns:
        bool: True if successful, False if error
    """
    available, message = check_openxr_available()

    if not available:
        print(f"ERROR: {message}")
        print("\nTo use native VR viewer:")
        print("1. Install PyOpenXR: pip install pyopenxr PyOpenGL pillow")
        print("2. Install SteamVR or Oculus runtime")
        print("3. Make sure your VR headset is connected")
        return False

    try:
        viewer = get_or_create_viewer()

        # Queue the new image
        viewer.update_image(image_path, stereo_format, swap_eyes)

        # If this is the first image, set it as current
        if viewer.current_image is None:
            viewer.current_image = image_path
            viewer.current_format = stereo_format
            viewer.current_swap = swap_eyes

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
