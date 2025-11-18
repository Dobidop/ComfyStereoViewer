# ComfyStereoViewer Examples

This directory contains example workflows demonstrating how to use ComfyStereoViewer nodes.

## Example Workflows

### 1. Basic Stereo Viewing
**File:** `basic_stereo_viewing.json`

A simple workflow that:
1. Loads two separate images (left and right eye)
2. Combines them into a side-by-side stereo image
3. Displays in the VR viewer

### 2. Stereo Image Processing
**File:** `stereo_processing.json`

Demonstrates how to:
1. Load a stereo side-by-side image
2. Split it into separate left/right images
3. Apply different processing to each eye
4. Recombine and view in VR

### 3. Anaglyph Creation
**File:** `anaglyph_creation.json`

Shows how to:
1. Take two standard images
2. Create a red-cyan anaglyph
3. Save for viewing with 3D glasses

### 4. Video Viewing
**File:** `video_viewing.json`

Demonstrates:
1. Loading a stereo video file
2. Configuring stereo format
3. Viewing in VR with controls

## How to Use

1. Open ComfyUI
2. Click "Load" button
3. Navigate to this examples directory
4. Select the desired workflow JSON file
5. Adjust node parameters as needed
6. Run the workflow

## Creating Your Own Workflows

### Essential Nodes

- **StereoCombine**: Combine left/right images
- **StereoSplit**: Split stereo images
- **StereoAnaglyph**: Create anaglyph images
- **StereoImageViewer**: View stereo images in VR
- **StereoVideoViewer**: View stereo videos in VR

### Tips

1. Always ensure left and right images have the same dimensions
2. Use "swap_eyes" parameter if the depth feels inverted
3. Adjust IPD scale for comfortable viewing (typically 0.8-1.2)
4. Test in desktop mode before entering VR
5. Use high-resolution images for best VR quality

## Example Node Connections

```
LoadImage (Left) ─┐
                  ├─> StereoCombine ─> StereoImageViewer
LoadImage (Right) ┘

LoadImage (Stereo) ─> StereoSplit ─┬─> ProcessNode (Left) ─┐
                                   │                        ├─> StereoCombine ─> StereoImageViewer
                                   └─> ProcessNode (Right) ─┘

LoadImage (Left) ─┐
                  ├─> StereoAnaglyph ─> SaveImage
LoadImage (Right) ┘
```

## Common Use Cases

1. **AI-Generated Stereo Pairs**: Generate two slightly different views from AI and combine
2. **Depth-based Stereo**: Use depth maps to create stereo pairs
3. **3D Photography**: Process stereo camera images
4. **Video Conversion**: Convert 2D to 3D or change stereo formats
5. **Quality Control**: Check stereo alignment and quality

## Support

For more information, see the main [README.md](../README.md) or open an issue on GitHub.
