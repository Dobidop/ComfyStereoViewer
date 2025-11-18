/**
 * ComfyStereoViewer - WebXR-enabled stereoscopic viewer
 * Supports VR headsets and multiple stereo formats
 */

class StereoViewer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.sphere = null;
        this.texture = null;
        this.videoElement = null;
        this.isVRMode = false;
        this.stereoFormat = 'sbs';
        this.swapEyes = false;
        this.ipdScale = 1.0;
        this.fov = 90;
        this.currentMedia = null;
        this.mediaType = 'image';

        this.mouseDown = false;
        this.mouseX = 0;
        this.mouseY = 0;
        this.lon = 0;
        this.lat = 0;
        this.phi = 0;
        this.theta = 0;

        this.init();
    }

    init() {
        // Initialize Three.js scene
        this.scene = new THREE.Scene();

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            this.fov,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 0);

        // Renderer setup
        const canvas = document.getElementById('viewer-canvas');
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.xr.enabled = true;

        // Create sphere for 360° viewing
        const geometry = new THREE.SphereGeometry(500, 60, 40);
        geometry.scale(-1, 1, 1); // Invert for inside viewing

        // Material will be set when media is loaded
        this.material = new THREE.MeshBasicMaterial();
        this.sphere = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.sphere);

        // Set up VR button
        this.setupVRButton();

        // Set up event listeners
        this.setupEventListeners();

        // Start animation loop
        this.animate();

        // Load media list
        this.loadMediaList();
    }

    setupVRButton() {
        const vrButton = document.getElementById('vr-button');
        const vrStatus = document.getElementById('vr-status');

        if ('xr' in navigator) {
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                if (supported) {
                    vrButton.addEventListener('click', () => {
                        if (!this.isVRMode) {
                            this.enterVR();
                        } else {
                            this.exitVR();
                        }
                    });
                    vrStatus.textContent = 'VR Ready';
                    vrStatus.className = 'vr-ready';
                } else {
                    vrButton.disabled = true;
                    vrButton.textContent = 'VR Not Supported';
                    vrStatus.textContent = 'WebXR not supported on this device';
                }
            });
        } else {
            vrButton.disabled = true;
            vrButton.textContent = 'VR Not Available';
            vrStatus.textContent = 'WebXR not available in this browser';
        }
    }

    async enterVR() {
        if (!this.renderer.xr.enabled) return;

        try {
            const session = await navigator.xr.requestSession('immersive-vr', {
                optionalFeatures: ['local-floor', 'bounded-floor']
            });

            await this.renderer.xr.setSession(session);
            this.isVRMode = true;
            document.getElementById('vr-button').textContent = 'Exit VR';
            document.getElementById('controls-panel').style.display = 'none';

            session.addEventListener('end', () => {
                this.exitVR();
            });

            // Adjust rendering for stereo in VR
            this.updateStereoMaterial();

        } catch (error) {
            console.error('Failed to enter VR:', error);
            alert('Failed to enter VR mode. Make sure your headset is connected.');
        }
    }

    exitVR() {
        this.isVRMode = false;
        document.getElementById('vr-button').textContent = 'Enter VR';
        document.getElementById('controls-panel').style.display = 'block';
        if (this.renderer.xr.getSession()) {
            this.renderer.xr.getSession().end();
        }
    }

    setupEventListeners() {
        const canvas = document.getElementById('viewer-canvas');

        // Mouse controls for desktop
        canvas.addEventListener('mousedown', (e) => {
            this.mouseDown = true;
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (this.mouseDown && !this.isVRMode) {
                const deltaX = e.clientX - this.mouseX;
                const deltaY = e.clientY - this.mouseY;

                this.lon -= deltaX * 0.1;
                this.lat += deltaY * 0.1;

                this.lat = Math.max(-85, Math.min(85, this.lat));

                this.mouseX = e.clientX;
                this.mouseY = e.clientY;
            }
        });

        canvas.addEventListener('mouseup', () => {
            this.mouseDown = false;
        });

        // Touch controls for mobile
        let touchStartX = 0;
        let touchStartY = 0;

        canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
            }
        });

        canvas.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1 && !this.isVRMode) {
                e.preventDefault();
                const deltaX = e.touches[0].clientX - touchStartX;
                const deltaY = e.touches[0].clientY - touchStartY;

                this.lon -= deltaX * 0.1;
                this.lat += deltaY * 0.1;

                this.lat = Math.max(-85, Math.min(85, this.lat));

                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
            }
        }, { passive: false });

        // Scroll to zoom
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.fov += e.deltaY * 0.05;
            this.fov = Math.max(30, Math.min(120, this.fov));
            this.camera.fov = this.fov;
            this.camera.updateProjectionMatrix();
            document.getElementById('fov-value').textContent = Math.round(this.fov) + '°';
        }, { passive: false });

        // UI controls
        document.getElementById('media-select').addEventListener('change', (e) => {
            this.loadMedia(e.target.value);
        });

        document.getElementById('format-select').addEventListener('change', (e) => {
            this.stereoFormat = e.target.value;
            this.updateStereoMaterial();
        });

        document.getElementById('swap-eyes').addEventListener('change', (e) => {
            this.swapEyes = e.target.checked;
            this.updateStereoMaterial();
        });

        document.getElementById('ipd-slider').addEventListener('input', (e) => {
            this.ipdScale = parseFloat(e.target.value);
            document.getElementById('ipd-value').textContent = this.ipdScale.toFixed(1);
            this.updateStereoMaterial();
        });

        document.getElementById('fov-slider').addEventListener('input', (e) => {
            this.fov = parseInt(e.target.value);
            this.camera.fov = this.fov;
            this.camera.updateProjectionMatrix();
            document.getElementById('fov-value').textContent = this.fov + '°';
        });

        document.getElementById('fullscreen-button').addEventListener('click', () => {
            if (!document.fullscreenElement) {
                document.body.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    loadMediaList() {
        // In a real implementation, this would fetch from the ComfyUI API
        // For now, we'll use a placeholder
        const select = document.getElementById('media-select');
        select.innerHTML = '<option value="">Select a stereo image or video...</option>';

        // This would be populated by the ComfyUI backend
        // Example entries:
        // select.innerHTML += '<option value="stereo_image_1.png">Stereo Image 1</option>';
    }

    loadMedia(filename) {
        if (!filename) return;

        document.getElementById('loading').style.display = 'block';

        // Determine if it's a video or image
        const ext = filename.split('.').pop().toLowerCase();
        const videoFormats = ['mp4', 'webm', 'ogv'];

        if (videoFormats.includes(ext)) {
            this.loadVideo(filename);
        } else {
            this.loadImage(filename);
        }
    }

    loadImage(filename) {
        const loader = new THREE.TextureLoader();
        loader.load(
            filename,
            (texture) => {
                this.texture = texture;
                this.mediaType = 'image';
                this.currentMedia = filename;
                this.updateStereoMaterial();
                document.getElementById('loading').style.display = 'none';
            },
            undefined,
            (error) => {
                console.error('Error loading image:', error);
                document.getElementById('loading').textContent = 'Error loading image';
            }
        );
    }

    loadVideo(filename) {
        if (this.videoElement) {
            this.videoElement.pause();
        }

        this.videoElement = document.createElement('video');
        this.videoElement.src = filename;
        this.videoElement.crossOrigin = 'anonymous';
        this.videoElement.loop = true;
        this.videoElement.muted = false;

        this.videoElement.addEventListener('loadeddata', () => {
            this.texture = new THREE.VideoTexture(this.videoElement);
            this.texture.minFilter = THREE.LinearFilter;
            this.texture.magFilter = THREE.LinearFilter;
            this.mediaType = 'video';
            this.currentMedia = filename;
            this.updateStereoMaterial();
            this.videoElement.play();
            document.getElementById('loading').style.display = 'none';
        });

        this.videoElement.addEventListener('error', (error) => {
            console.error('Error loading video:', error);
            document.getElementById('loading').textContent = 'Error loading video';
        });
    }

    updateStereoMaterial() {
        if (!this.texture) return;

        // Create custom shader for stereo rendering
        const uvTransform = this.getUVTransform();

        this.material.map = this.texture;
        this.material.needsUpdate = true;

        // Apply UV transformation based on stereo format
        if (this.stereoFormat === 'sbs') {
            // Side-by-side: each eye sees half the texture
            this.setupSideBySide(uvTransform);
        } else if (this.stereoFormat === 'ou') {
            // Over-under: each eye sees top or bottom half
            this.setupOverUnder(uvTransform);
        } else if (this.stereoFormat === 'anaglyph') {
            // Anaglyph: red-cyan
            this.setupAnaglyph();
        } else {
            // Mono: full texture for both eyes
            this.setupMono();
        }
    }

    getUVTransform() {
        return {
            offsetX: 0,
            offsetY: 0,
            scaleX: 1,
            scaleY: 1
        };
    }

    setupSideBySide(uvTransform) {
        // For side-by-side, we need to adjust UV coordinates
        // Left eye sees left half, right eye sees right half
        const geometry = this.sphere.geometry;
        const uvAttribute = geometry.attributes.uv;

        for (let i = 0; i < uvAttribute.count; i++) {
            const u = uvAttribute.getX(i);
            const v = uvAttribute.getY(i);

            // Scale U coordinate to use only half the texture
            const newU = this.swapEyes ? u * 0.5 + 0.5 : u * 0.5;
            uvAttribute.setX(i, newU);
        }

        uvAttribute.needsUpdate = true;
    }

    setupOverUnder(uvTransform) {
        const geometry = this.sphere.geometry;
        const uvAttribute = geometry.attributes.uv;

        for (let i = 0; i < uvAttribute.count; i++) {
            const u = uvAttribute.getX(i);
            const v = uvAttribute.getY(i);

            // Scale V coordinate to use only half the texture
            const newV = this.swapEyes ? v * 0.5 + 0.5 : v * 0.5;
            uvAttribute.setY(i, newV);
        }

        uvAttribute.needsUpdate = true;
    }

    setupAnaglyph() {
        // Anaglyph is already processed, just display normally
        this.setupMono();
    }

    setupMono() {
        // Reset UV coordinates to default
        const geometry = this.sphere.geometry;
        const uvAttribute = geometry.attributes.uv;

        // Rebuild geometry if needed
        const originalGeometry = new THREE.SphereGeometry(500, 60, 40);
        originalGeometry.scale(-1, 1, 1);

        geometry.attributes.uv.copy(originalGeometry.attributes.uv);
        geometry.attributes.uv.needsUpdate = true;
    }

    animate() {
        this.renderer.setAnimationLoop(() => {
            this.update();
            this.renderer.render(this.scene, this.camera);
        });
    }

    update() {
        if (!this.isVRMode) {
            // Update camera rotation based on mouse/touch input
            this.lat = Math.max(-85, Math.min(85, this.lat));
            this.phi = THREE.MathUtils.degToRad(90 - this.lat);
            this.theta = THREE.MathUtils.degToRad(this.lon);

            this.camera.position.x = 100 * Math.sin(this.phi) * Math.cos(this.theta);
            this.camera.position.y = 100 * Math.cos(this.phi);
            this.camera.position.z = 100 * Math.sin(this.phi) * Math.sin(this.theta);

            this.camera.lookAt(this.scene.position);
        }
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const viewer = new StereoViewer();
});
