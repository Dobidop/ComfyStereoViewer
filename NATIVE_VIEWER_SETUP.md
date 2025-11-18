# Native VR Viewer Setup Guide

This guide explains how to set up and use the native PyOpenXR viewer for ComfyStereoViewer, which provides auto-launch VR viewing without a browser.

---

## Why Use Native Viewer?

### Native PyOpenXR Viewer
✅ **Auto-launches** directly into VR headset
✅ **Better performance** (no browser overhead)
✅ **Lower latency**
✅ **Seamless experience** (no "Enter VR" button needed)
✅ **Offline support**

### WebXR Browser Viewer (Default)
✅ Works immediately without setup
✅ Maximum compatibility
❌ Requires manual "Enter VR" button click
❌ Browser performance overhead

---

## Prerequisites

### 1. VR Runtime

You need an OpenXR-compatible VR runtime installed:

#### **SteamVR** (Recommended - supports most headsets)
- Download: https://store.steampowered.com/app/250820/SteamVR/
- Supports: Valve Index, HTC Vive, Quest (via Link), WMR, and more
- Free and widely compatible

#### **Oculus Runtime** (For Oculus/Meta headsets)
- Included with Oculus software
- Best for native Oculus Rift/Quest via Link

#### **Windows Mixed Reality** (For WMR headsets)
- Built into Windows 10/11
- Automatically available for WMR headsets

### 2. Python Dependencies

Install PyOpenXR and OpenGL libraries:

```bash
pip install -r requirements-native.txt
```

Or manually:

```bash
pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw
```

### 3. VR Headset

Connect and set up your VR headset:
- **Quest**: Use Link cable or Air Link
- **Index/Vive**: Connect via SteamVR
- **WMR**: Connect and enable Mixed Reality Portal

---

## Installation Steps

### Step 1: Install ComfyStereoViewer

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Dobidop/ComfyStereoViewer.git
cd ComfyStereoViewer
pip install -r requirements.txt
```

### Step 2: Install Native VR Dependencies (Optional)

```bash
pip install -r requirements-native.txt
```

### Step 3: Install and Start VR Runtime

**For SteamVR:**
1. Install SteamVR from Steam
2. Connect your headset
3. Start SteamVR (it should auto-start when headset is connected)

**For Quest via Link:**
1. Install Oculus software
2. Connect Quest via USB or enable Air Link
3. Start Oculus Link from Quest menu

**For WMR:**
1. Connect headset
2. Mixed Reality Portal should start automatically

### Step 4: Verify Setup

In ComfyUI, add the **"Check Native VR Status"** node to verify everything is working:

```
Add Node → stereo → native → Check Native VR Status
```

This will show:
- ✓ PyOpenXR installed
- ✓ OpenXR runtime detected
- ✓ Ready for native VR viewing

---

## Available Native Nodes

### 1. Native Stereo Viewer (PyOpenXR)

**Path:** `stereo → native → Native Stereo Viewer (PyOpenXR)`

Auto-launches stereo images directly into VR headset.

**Inputs:**
- `image` - Stereo image to view
- `stereo_format` - Side-by-Side, Over-Under, Anaglyph, or Mono
- `swap_eyes` - Swap left/right if needed
- `auto_launch` - Automatically launch into VR (default: true)
- `right_image` (optional) - Separate right eye image

**How it works:**
1. Saves the image
2. Launches PyOpenXR viewer
3. **Automatically enters VR mode** (put on headset!)
4. Press Ctrl+C in console to exit

### 2. Check Native VR Status

**Path:** `stereo → native → Check Native VR Status`

Diagnostic node to verify native VR setup.

**Outputs:**
- `status_message` - Detailed status information
- `is_available` - Boolean indicating if native VR is ready

### 3. Hybrid Stereo Viewer (Auto)

**Path:** `stereo → Hybrid Stereo Viewer (Auto)`

**Automatically selects best viewer:**
- Native PyOpenXR if available
- Falls back to WebXR if not

**This is the recommended node for most users!**

**Inputs:**
- `image` - Stereo image
- `stereo_format` - Format selection
- `swap_eyes` - Eye swap option
- `prefer_native` - Prefer native viewer if available (default: true)
- `right_image` (optional) - Separate right eye

**Outputs:**
- `passthrough` - Original image
- `viewer_type` - Which viewer was used ("native_pyopenxr" or "webxr_browser")

---

## Usage Examples

### Example 1: Basic Native Viewing

```
LoadImage → Native Stereo Viewer (PyOpenXR)
```

Set stereo format and run workflow. **Headset will auto-launch!**

### Example 2: Hybrid Auto-Select

```
LoadImage → Hybrid Stereo Viewer (Auto)
```

Automatically uses native viewer if available, otherwise falls back to WebXR.

### Example 3: Separate L/R Images

```
LoadImage (Left)  ┐
                  ├→ Native Stereo Viewer
LoadImage (Right) ┘
```

Connect left image to main input, right image to optional input.

### Example 4: Check Status First

```
Check Native VR Status → Display Text

LoadImage → Native Stereo Viewer (PyOpenXR)
```

Verify setup before attempting to view.

---

## Troubleshooting

### "PyOpenXR not available"

**Solution:**
```bash
pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw
```

### "OpenXR runtime not available"

**Possible causes:**
1. **SteamVR not running** → Start SteamVR
2. **Headset not connected** → Connect and power on headset
3. **No runtime installed** → Install SteamVR or Oculus software

**To check:**
- Open SteamVR settings
- Verify headset is detected
- Check for any SteamVR errors

### "Viewer launches but black screen"

**Solutions:**
1. Make sure headset is on and tracking
2. Check that SteamVR shows headset as ready
3. Try restarting SteamVR
4. Verify image file exists and is valid

### Performance Issues

**Solutions:**
1. Close unnecessary applications
2. Update GPU drivers
3. Reduce image resolution
4. Check SteamVR performance settings

### Quest-Specific Issues

**Quest via Link not working:**
1. Enable Link in Quest settings
2. Allow USB debugging
3. Check cable connection (or Air Link connection)
4. Verify Oculus software recognizes Quest

### "Viewer won't exit"

Press **Ctrl+C** in the ComfyUI console to stop the viewer.

---

## Performance Tips

1. **Image Resolution:**
   - Recommended: 3840x2160 or lower for SBS
   - Larger images may cause lag

2. **VR Runtime Settings:**
   - SteamVR: Set render resolution to 100%
   - Disable supersampling if experiencing lag

3. **GPU:**
   - Native viewer uses OpenGL
   - Ensure GPU drivers are up to date
   - Close GPU-intensive applications

---

## Comparison: Native vs WebXR

| Feature | Native PyOpenXR | WebXR Browser |
|---------|----------------|---------------|
| **Setup** | Requires PyOpenXR + VR runtime | Works immediately |
| **Launch** | Auto-launch to headset | Manual "Enter VR" click |
| **Performance** | Better (direct rendering) | Good (browser overhead) |
| **Latency** | Lower | Slightly higher |
| **Compatibility** | PC VR (SteamVR, Oculus) | All WebXR devices |
| **Quest Standalone** | No (needs Link) | Yes |
| **Offline** | Yes | Limited |

---

## Recommended Workflow

### For PC VR Users (Index, Vive, Rift, Quest+Link):

1. Install native dependencies: `pip install -r requirements-native.txt`
2. Use **Hybrid Stereo Viewer (Auto)** node
3. Enjoy auto-launch VR viewing!

### For Quest Standalone Users:

- Use **WebXR browser viewer** (works natively on Quest browser)
- Native viewer requires Quest Link to PC

### For Maximum Compatibility:

- Use **Hybrid Stereo Viewer (Auto)**
- It automatically picks the best option

---

## Advanced: Manual Launch

You can also launch the native viewer from command line:

```bash
python -m ComfyStereoViewer.native_viewer image.png sbs false
```

Arguments:
- `image.png` - Path to stereo image
- `sbs` - Format: sbs, ou, anaglyph, mono
- `false` - Swap eyes: true or false

---

## System Requirements

### Minimum:
- Python 3.8+
- OpenXR-compatible VR runtime (SteamVR, Oculus, WMR)
- GPU with OpenGL 3.3+ support
- VR headset (Index, Vive, Quest, Rift, WMR, etc.)

### Recommended:
- Python 3.10+
- Dedicated GPU (NVIDIA/AMD)
- SteamVR installed
- Updated GPU drivers

---

## Support

If you encounter issues:

1. Run **Check Native VR Status** node
2. Check console output for error messages
3. Verify VR runtime is running
4. Check GitHub issues: https://github.com/Dobidop/ComfyStereoViewer/issues

---

## Next Steps

Once native VR is working:

1. Try the **Hybrid Stereo Viewer** for automatic selection
2. Experiment with different stereo formats
3. Create workflows combining stereo processing with native viewing
4. Adjust IPD and swap eyes for comfortable viewing

**Enjoy seamless VR viewing!** 🥽
