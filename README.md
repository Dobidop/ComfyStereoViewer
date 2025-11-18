# ComfyStereoViewer

A comprehensive stereoscopic 3D viewer for ComfyUI with VR headset support via WebXR. View stereo images and videos in immersive 3D with support for Quest, Vive, Index, and all WebXR-compatible devices.

## Features

- **VR Headset Support**: Full WebXR integration for immersive viewing
- **Multiple Stereo Formats**: Side-by-Side, Over-Under, Anaglyph, and Separate L/R
- **Image & Video Support**: View both stereo images and videos
- **Cross-Platform**: Works on desktop, mobile, and VR headsets
- **Real-time Controls**: Adjust IPD, FOV, and stereo settings on the fly
- **ComfyUI Integration**: Seamless workflow integration with custom nodes

## Supported VR Headsets

- Meta Quest (1, 2, 3, Pro)
- HTC Vive / Vive Pro
- Valve Index
- Windows Mixed Reality headsets
- Any WebXR-compatible VR device

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "ComfyStereoViewer"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/Dobidop/ComfyStereoViewer.git
   ```

3. Install dependencies:
   ```bash
   cd ComfyStereoViewer
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## Available Nodes

### 1. Stereo Image Viewer (VR)

Views stereoscopic images with VR support.

**Inputs:**
- `image` (IMAGE): The stereo image to view
- `stereo_format`: Side-by-Side, Over-Under, Anaglyph, Left Only, or Right Only
- `swap_eyes` (BOOLEAN): Swap left and right eye views
- `ipd_scale` (FLOAT): Interpupillary distance scale (0.1-2.0)
- `right_image` (IMAGE, optional): Separate right eye image

**Outputs:**
- `passthrough` (IMAGE): Original image passed through for further processing

### 2. Stereo Video Viewer (VR)

Views stereoscopic videos with VR support.

**Inputs:**
- `video_path` (STRING): Path to the stereo video file
- `stereo_format`: Side-by-Side or Over-Under
- `swap_eyes` (BOOLEAN): Swap left and right eye views
- `ipd_scale` (FLOAT): Interpupillary distance scale (0.1-2.0)

**Outputs:**
- `video_path` (STRING): Original path passed through

### 3. Combine Stereo Images

Combines separate left and right images into a single stereo image.

**Inputs:**
- `left_image` (IMAGE): Left eye image
- `right_image` (IMAGE): Right eye image
- `layout`: Side-by-Side or Over-Under
- `swap_eyes` (BOOLEAN): Swap left and right positions

**Outputs:**
- Stereo combined image (IMAGE)

### 4. Split Stereo Image

Splits a stereo image into separate left and right images.

**Inputs:**
- `stereo_image` (IMAGE): The stereo image to split
- `layout`: Side-by-Side or Over-Under
- `swap_eyes` (BOOLEAN): Swap the split eyes

**Outputs:**
- `left_image` (IMAGE)
- `right_image` (IMAGE)

### 5. Create Anaglyph 3D

Creates red-cyan anaglyph images from stereo pairs.

**Inputs:**
- `left_image` (IMAGE): Left eye image
- `right_image` (IMAGE): Right eye image
- `anaglyph_type`: Red-Cyan, Green-Magenta, or Amber-Blue
- `swap_eyes` (BOOLEAN): Swap eyes

**Outputs:**
- Anaglyph image (IMAGE)

## Usage Examples

### Example 1: Basic Stereo Viewing

1. Load two images (left and right eye views)
2. Connect them to the **Combine Stereo Images** node
3. Connect output to **Stereo Image Viewer (VR)**
4. Click "Enter VR" in the web viewer to view in your headset

### Example 2: Video Playback

1. Add a **Stereo Video Viewer (VR)** node
2. Set the video path to your stereo video file
3. Choose the appropriate stereo format (SBS or OU)
4. View in VR or on desktop

### Example 3: Creating Anaglyph Images

1. Load left and right eye images
2. Connect to **Create Anaglyph 3D** node
3. Choose anaglyph type (Red-Cyan recommended)
4. View with 3D glasses or connect to viewer

### Example 4: Processing Existing Stereo Images

1. Load a side-by-side stereo image
2. Connect to **Split Stereo Image** node
3. Process each eye separately (apply filters, etc.)
4. Recombine with **Combine Stereo Images** node
5. View with **Stereo Image Viewer (VR)**

## Web Viewer Controls

### Desktop Controls
- **Click and Drag**: Pan the view
- **Mouse Wheel**: Zoom in/out (adjust FOV)
- **VR Button**: Enter VR mode (if headset connected)
- **Fullscreen**: Toggle fullscreen mode

### VR Controls
- **Head Movement**: Look around naturally
- **Controller Buttons**: (depends on headset)
- Exit VR: Use headset menu or browser VR controls

### Settings Panel
- **Stereo Format**: Choose viewing format
- **Swap Eyes**: Reverse left/right if crosseyed
- **IPD Scale**: Adjust stereo separation
- **Field of View**: Adjust viewing angle

## Stereo Format Guide

### Side-by-Side (SBS)
- Left and right images placed horizontally
- Most common format for VR content
- Full or half resolution variants

### Over-Under (OU)
- Left and right images stacked vertically
- Common for 3D movies
- Also called "top-bottom" format

### Anaglyph
- Red-cyan colored glasses required
- Works on any 2D display
- Lower quality but universally compatible

### Separate L/R
- Highest quality option
- Each eye gets full resolution image
- Best for professional workflows

## Tips for Best Results

1. **Image Quality**: Use high-resolution images for clearer VR viewing
2. **IPD Adjustment**: Adjust IPD scale if the 3D effect feels wrong
3. **Swap Eyes**: If the depth looks inverted, try swapping eyes
4. **Format Selection**: Use SBS for most VR content, OU for videos
5. **Browser Compatibility**: Use Chrome or Edge for best WebXR support

## Browser Requirements

- **Desktop**: Chrome 79+, Edge 79+, Firefox 98+ (with WebXR enabled)
- **VR**: Chrome on Quest, SteamVR browser, Viveport browser
- **Mobile**: Chrome on Android (for mobile VR)

## Troubleshooting

### VR Button Disabled
- Check that your browser supports WebXR
- Ensure your VR headset is properly connected
- Try using Chrome or Edge browsers

### Images Not Loading
- Check that images are in ComfyUI's output directory
- Verify file permissions
- Check browser console for errors

### Wrong Stereo Format
- Try different format options
- Use "Swap Eyes" if depth appears inverted
- Verify source content format

### Poor VR Performance
- Reduce image resolution
- Close other browser tabs
- Update graphics drivers
- Try lower quality settings

## Technical Details

### Architecture
- **Backend**: Python nodes integrated with ComfyUI
- **Frontend**: Three.js for 3D rendering
- **VR**: WebXR Device API for headset support
- **Video**: HTML5 video with texture mapping

### File Structure
```
ComfyStereoViewer/
├── __init__.py              # Node registration
├── nodes.py                 # Custom node implementations
├── requirements.txt         # Python dependencies
├── web/                     # Web viewer interface
│   ├── stereo_viewer.html   # Main viewer page
│   ├── stereo_viewer.js     # WebXR and rendering logic
│   └── styles.css           # Interface styling
└── README.md               # This file
```

### Supported Media Formats

**Images:**
- PNG
- JPEG
- WebP
- Any format supported by PIL/Pillow

**Videos:**
- MP4 (H.264, H.265)
- WebM (VP8, VP9)
- OGV (Theora)

## Development

### Building from Source

```bash
git clone https://github.com/Dobidop/ComfyStereoViewer.git
cd ComfyStereoViewer
pip install -r requirements.txt
```

### Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - see LICENSE file for details

## Credits

Created by [Dobidop](https://github.com/Dobidop)

## Support

- **Issues**: [GitHub Issues](https://github.com/Dobidop/ComfyStereoViewer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Dobidop/ComfyStereoViewer/discussions)

## Roadmap

- [ ] Real-time depth map generation
- [ ] Support for more stereo formats (checkerboard, line-interleaved)
- [ ] 360° stereo video support
- [ ] Spatial audio integration
- [ ] Hand tracking support
- [ ] Multi-user viewing sessions
- [ ] VR UI overlays for controls

## Changelog

### Version 1.0.0 (Initial Release)
- Basic stereo image viewing
- VR headset support via WebXR
- Multiple stereo formats (SBS, OU, Anaglyph)
- Video playback support
- Desktop and mobile viewing
- Real-time control adjustments

---

**Enjoy viewing your stereo content in immersive 3D!**
