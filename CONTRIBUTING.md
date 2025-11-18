# Contributing to ComfyStereoViewer

Thank you for your interest in contributing to ComfyStereoViewer! This document provides guidelines for contributing to the project.

## Ways to Contribute

- Report bugs and issues
- Suggest new features or enhancements
- Improve documentation
- Submit bug fixes
- Add new stereo formats support
- Improve VR experience
- Add tests

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ComfyStereoViewer.git
   cd ComfyStereoViewer
   ```
3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install ComfyUI if you haven't already
2. Navigate to the custom_nodes directory
3. Clone your fork into the custom_nodes directory
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Restart ComfyUI

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Comment complex logic
- Keep functions focused and concise

### Python Example

```python
def process_stereo_image(left_image, right_image, format="sbs"):
    """
    Process stereo images into the specified format.

    Args:
        left_image: Left eye image tensor
        right_image: Right eye image tensor
        format: Output format ("sbs", "ou", "anaglyph")

    Returns:
        Processed stereo image tensor
    """
    # Implementation here
    pass
```

### JavaScript Example

```javascript
/**
 * Update stereo material based on current settings
 * @param {THREE.Texture} texture - The texture to apply
 * @param {string} format - Stereo format (sbs, ou, anaglyph)
 */
updateStereoMaterial(texture, format) {
    // Implementation here
}
```

## Testing

Before submitting a pull request:

1. Test your changes in ComfyUI
2. Test on both desktop and VR (if applicable)
3. Test with different stereo formats
4. Check for console errors
5. Verify backwards compatibility

### Testing Checklist

- [ ] Code runs without errors
- [ ] Nodes appear correctly in ComfyUI
- [ ] Viewer loads properly
- [ ] VR mode works (if applicable)
- [ ] All stereo formats work
- [ ] No console errors
- [ ] Documentation is updated

## Submitting Changes

1. Commit your changes with clear messages:
   ```bash
   git commit -m "Add support for new stereo format"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a Pull Request on GitHub

### Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include screenshots/videos for UI changes
- List testing performed
- Update documentation if needed

### Good PR Description Example

```markdown
## Description
Adds support for checkerboard stereo format

## Changes
- Added checkerboard format option to viewer
- Implemented UV mapping for checkerboard layout
- Updated documentation with new format

## Testing
- [x] Tested in desktop mode
- [x] Tested in VR mode
- [x] Verified with sample images
- [x] No console errors

## Screenshots
[Include relevant screenshots]

Fixes #123
```

## Reporting Bugs

When reporting bugs, include:

- ComfyUI version
- Browser and version (for viewer issues)
- VR headset model (for VR issues)
- Steps to reproduce
- Expected behavior
- Actual behavior
- Screenshots or error messages
- Console logs if applicable

### Bug Report Template

```markdown
**Environment:**
- ComfyUI Version:
- Browser:
- VR Headset:
- OS:

**Description:**
A clear description of the bug

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Screenshots/Logs:**
[Include relevant images or logs]
```

## Feature Requests

When requesting features:

- Explain the use case
- Describe the desired behavior
- Provide examples if possible
- Consider implementation complexity

## Code Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited

## Areas for Contribution

### High Priority

- Additional stereo formats (checkerboard, line-interleaved)
- 360° video support
- Performance optimizations
- Better error handling
- Test coverage

### Medium Priority

- Spatial audio support
- Hand tracking integration
- UI improvements
- More example workflows
- Video format support

### Documentation

- Tutorial videos
- More detailed examples
- Translation to other languages
- API documentation

## Questions?

Feel free to:
- Open an issue for discussion
- Ask in pull request comments
- Check existing issues and discussions

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the project goals
- Accept constructive criticism

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to ComfyStereoViewer!
