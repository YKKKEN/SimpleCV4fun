# SimpleCV4Fun

> A collection of computer vision projects developed for exploratory and recreational purposes.

## Projects Overview

| File | Description |
|------|-------------|
| `color_detector_gui.py` | Multi-functional GUI tool for HSV/RGB color detection, image processing, and video analysis with dark/light themes |
| `hough_circle_tuner.py` | Real-time parameter adjustment tool for Hough Circle detection with various preprocessing filters |
| `video_circle_detection.py` | Real-time circle detection from video streams with color filtering (blue/red) |
| `image_circle_detection.py` | Static image circle detection combining color masks and Hough transform |

## Features

- Color detection (HSV/RGB color space)
- Hough Circle detection with adjustable parameters
- Multiple preprocessing filters (Median, Gaussian, Bilateral blur)
- Dark/Light theme support
- Real-time video processing

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pillow (`PIL`)
- Tkinter (usually included with Python)

## Quick Start

```bash
git clone https://github.com/ykkken/SimpleCV4fun.git
cd SimpleCV4fun
python src/<project_name>.py
```

## Project Structure

```
SimpleCV4fun/
├── src/                            # Source code
│   ├── color_detector_gui.py       # Vision Analysis System
│   ├── hough_circle_tuner.py       # Parameter Tuner
│   ├── video_circle_detection.py   # Video Detection
│   └── image_circle_detection.py   # Image Detection
├── test_images/                    # Demo images and videos
└── README.md
```

## License

MIT License - Feel free to use and modify for your own projects!