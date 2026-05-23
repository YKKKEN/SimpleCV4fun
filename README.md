# SimpleCV4Fun

> 一个用于探索和娱乐目的的计算机视觉项目集合。

---

## 项目概述

SimpleCV4Fun 是一系列基于 Python 的计算机视觉应用程序集合，旨在用于学习、实验和实际应用。项目包含以下工具：

- **颜色检测**：HSV/RGB 颜色空间分析与过滤
- **圆形检测**：霍夫圆变换与实时参数调节
- **形状识别**：A4纸上的几何形状检测（圆形、三角形、正方形）
- **视频处理**：实时视频流分析与暂停/继续功能

---

## 功能特性

| 功能 | 描述 |
|------|------|
| 颜色检测 | HSV/RGB 颜色空间检测与过滤（蓝色和红色） |
| 霍夫圆检测 | 可调节参数（dp、minDist、param1、param2、minRadius、maxRadius） |
| 预处理滤波器 | 中值滤波、高斯滤波、双边滤波、Canny边缘检测 |
| 形状识别 | 圆形、三角形、正方形检测与尺寸测量 |
| A4纸检测 | 透视变换与基于A4纸的距离计算 |
| 主题支持 | GUI应用支持深色/浅色主题切换 |
| 视频处理 | 实时处理支持暂停/继续（空格键） |
| 全屏模式 | F11 键切换全屏显示 |

---

## 环境要求

### Python 版本
- Python 3.7 或更高版本

### 依赖库

| 包名 | 用途 |
|------|------|
| `opencv-python` (cv2) | 图像处理、视频捕获、霍夫变换 |
| `numpy` | 数值运算、数组处理 |
| `Pillow` (PIL) | 图像格式转换（用于Tkinter显示） |
| `tkinter` | GUI框架（通常随Python安装） |

---

## 安装指南

### 步骤 1：克隆仓库

```bash
git clone https://github.com/ykkken/SimpleCV4fun.git
cd SimpleCV4fun
```

### 步骤 2：创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 步骤 3：安装依赖

```bash
pip install opencv-python numpy Pillow
```

### 步骤 4：验证安装

```bash
python -c "import cv2; import numpy; from PIL import Image; print('所有依赖安装成功！')"
```

---

## 快速开始

直接运行 `src/` 目录下的任意项目：

```bash
# GUI 应用程序
python src/color_detector_gui.py      # 视觉分析系统
python src/hough_circle_tuner.py      # 霍夫圆参数调节工具

# 视频处理
python src/video_circle_detection.py  # 视频圆形检测

# 图像处理
python src/image_circle_detection.py  # 静态图像圆形检测

# 形状检测（需要摄像头）
python src/ShapeDetect.py             # A4纸上的形状检测
python src/MultiSquareDetect.py       # 最小正方形边长检测
```

---

## 项目结构

```
SimpleCV4fun/
├── src/                            # 源代码
│   ├── color_detector_gui.py       # 多功能视觉分析GUI
│   ├── hough_circle_tuner.py       # 霍夫圆参数调节GUI
│   ├── video_circle_detection.py   # 视频圆形检测
│   ├── image_circle_detection.py   # 静态图像圆形检测
│   ├── ShapeDetect.py              # 形状检测（圆形/三角形/正方形）
│   └── MultiSquareDetect.py        # 最小正方形边长检测
├── test_images/                    # 示例图像和视频
│   ├── HoughCircles(VIDEO).mp4     # 圆形检测示例视频
│   └── HoughCircles.jpg            # 圆形检测示例图像
└── README.md                       # 说明文档
```

---

## 详细使用说明

### 1. 颜色检测器 GUI (`color_detector_gui.py`)

一个综合性的GUI工具，用于颜色检测和图像/视频分析。

**功能特性：**
- 打开图像或视频进行分析
- HSV/RGB 颜色空间可视化
- 深色/浅色主题切换
- 视频播放进度条
- 鼠标交互取色

**控制按钮：**
| 按钮 | 功能 |
|------|------|
| 打开图片 | 加载图像文件进行分析 |
| 打开视频 | 加载视频文件进行处理 |
| 切换主题 | 在深色和浅色主题间切换 |

**使用方法：**
```bash
python src/color_detector_gui.py
```

---

### 2. 霍夫圆调节器 (`hough_circle_tuner.py`)

霍夫圆检测的实时参数调节工具。

**功能特性：**
- 可调节的霍夫圆参数
- 多种预处理滤波器
- 对比度、亮度、锐化控制
- 全屏模式（F11）

**参数说明：**
| 参数 | 描述 | 默认值 |
|------|------|--------|
| dp | 图像分辨率倒数 | 1.0 |
| minDist | 圆心最小距离 | 50 |
| param1 | Canny边缘检测高阈值 | 100 |
| param2 | 中心点累加器阈值 | 30 |
| minRadius | 最小半径 | 0 |
| maxRadius | 最大半径 | 0 |

**预处理选项：**
- 中值滤波（去除椒盐噪声）
- 高斯滤波（平滑图像）
- 双边滤波（保边平滑）
- 均值滤波（简单平均）
- Canny边缘检测

**使用方法：**
```bash
python src/hough_circle_tuner.py
```

---

### 3. 视频圆形检测 (`video_circle_detection.py`)

从视频文件中实时检测圆形，支持颜色过滤。

**功能特性：**
- 蓝色和红色圆形检测
- 显示圆心坐标和半径
- 颜色计数统计
- 暂停/继续功能

**快捷键：**
| 按键 | 功能 |
|------|------|
| 空格 | 暂停/继续视频 |
| Esc | 退出程序 |

**配置方法：**
修改第200行设置视频路径：
```python
video_path = r'{your_video_path}'
```

**使用方法：**
```bash
python src/video_circle_detection.py
```

---

### 4. 图像圆形检测 (`image_circle_detection.py`)

静态图像的圆形检测，支持颜色过滤。

**功能特性：**
- 蓝色和红色圆形检测
- 输出圆心坐标和半径
- 每个圆的颜色分类

**配置方法：**
修改第103行设置图像路径：
```python
im_path = r'{your-pic-path}'
```

**可调参数：**
```python
res = circle_detection(im_path, min_radius=100, max_radius=233)
```

**使用方法：**
```bash
python src/image_circle_detection.py
```

---

### 5. 形状检测 (`ShapeDetect.py`)

使用摄像头在A4纸上实时检测形状（圆形、三角形、正方形）。

**功能特性：**
- A4纸检测与跟踪
- 到A4纸的距离计算
- 形状分类（圆形/三角形/正方形）
- 尺寸测量（厘米）

**输出信息：**
| 显示 | 描述 |
|------|------|
| D | 到A4纸的距离（厘米） |
| S | 形状类型（Circle/Triangle/Square） |
| X | 形状尺寸（厘米） |

**快捷键：**
| 按键 | 功能 |
|------|------|
| Esc | 退出程序 |

**使用要求：**
- 摄像头（索引0）
- A4纸作为参考
- 推荐距离范围：100-180厘米

**使用方法：**
```bash
python src/ShapeDetect.py
```

---

### 6. 多正方形检测 (`MultiSquareDetect.py`)

在A4纸上实时检测最小正方形边长。

**功能特性：**
- A4纸检测
- 内部形状轮廓检测
- 最小正方形边长计算
- 稳定性过滤确保测量一致

**输出信息：**
- 最小边长（厘米）

**快捷键：**
| 按键 | 功能 |
|------|------|
| Esc | 退出程序 |

**使用方法：**
```bash
python src/MultiSquareDetect.py
```

---

## 配置参数

### 颜色检测参数

颜色检测的默认HSV范围：

| 颜色 | HSV下限 | HSV上限 |
|------|---------|---------|
| 蓝色 | [95, 40, 40] | [135, 255, 255] |
| 红色（范围1） | [0, 120, 70] | [8, 255, 255] |
| 红色（范围2） | [170, 120, 70] | [180, 255, 255] |

### A4纸常量

```python
A4W = 21.0   # A4纸宽度（厘米）
A4H = 29.7   # A4纸高度（厘米）
A4S = A4W * A4H  # A4纸面积（平方厘米）
FOCAL_LENGTH = 800.0  # 默认焦距（像素）
```

### 霍夫圆检测参数

根据不同场景调整参数以获得最佳效果：

| 场景 | dp | minDist | param1 | param2 | minRadius | maxRadius |
|------|-----|---------|--------|--------|-----------|-----------|
| 大圆形 | 1.0 | 50 | 100 | 30 | 50 | 150 |
| 小圆形 | 1.5 | 30 | 50 | 20 | 10 | 50 |
| 密集圆形 | 1.0 | 20 | 80 | 25 | 20 | 80 |

---

## 常见问题

### 问题排查

#### 1. 摄像头无法打开
```
错误: 相机坏了哦
```
**解决方案：**
- 确保摄像头已连接且未被其他程序占用
- 尝试更改摄像头索引：将 `cv2.VideoCapture(0)` 改为 `cv2.VideoCapture(1)`

#### 2. 模块未找到
```
ModuleNotFoundError: No module named 'cv2'
```
**解决方案：**
```bash
pip install opencv-python
```

#### 3. Tkinter 不可用
```
ModuleNotFoundError: No module named 'tkinter'
```
**解决方案：**
- **Windows**：Tkinter 随 Python 安装程序一起提供
- **Linux**：`sudo apt-get install python3-tk`
- **macOS**：从 python.org 安装的 Python 包含 Tkinter

#### 4. 视频/图像文件未找到
```
文件找不到
```
**解决方案：**
- 修改脚本中的路径变量
- 使用绝对路径：`r'C:\path\to\your\file.mp4'`
- 确保文件存在于指定位置

#### 5. 未检测到圆形
```
未检测到圆形
```
**解决方案：**
- 调整 `minRadius` 和 `maxRadius` 参数
- 降低 `param2` 阈值以提高灵敏度
- 确保光照条件良好
- 尝试不同的预处理滤波器

#### 6. 形状检测效果差
**解决方案：**
- 确保A4纸完全可见且平整
- 保持距离在100-180厘米之间
- 使用良好的光照，避免阴影
- 保持摄像头稳定

### 性能优化建议

1. **视频处理**：使用较低分辨率可加快处理速度
2. **实时检测**：关闭其他应用程序以释放CPU资源
3. **提高准确性**：使用良好光照并避免运动模糊

---

## 许可证

本项目采用 MIT 许可证 - 欢迎自由使用和修改！

