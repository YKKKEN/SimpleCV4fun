import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class HoughCircleTuner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("霍夫圆实时调参工具 1.0")
        self.geometry("1200x900")  
        self.resizable(True, True)  # 允许窗口调整大小
        self.configure(bg="#1e1e1e")
        
        # 全屏状态标志
        self.fullscreen = False

        # 默认参数
        self.dp = tk.DoubleVar(value=1.0)    # 图像分辨率倒数
        self.minDist = tk.IntVar(value=50)   # 圆心最小距离
        self.param1 = tk.IntVar(value=100)   # Canny边缘检测高阈值
        self.param2 = tk.IntVar(value=30)    # 中心点累加器阈值
        self.minRadius = tk.IntVar(value=0)  # 最小半径
        self.maxRadius = tk.IntVar(value=0)  # 最大半径
        
        # 预处理参数
        self.use_median_blur = tk.BooleanVar(value=True)     # 是否使用中值滤波
        self.median_blur_size = tk.IntVar(value=5)           # 中值滤波核大小
        
        self.use_gaussian_blur = tk.BooleanVar(value=False)  # 是否使用高斯滤波
        self.gaussian_blur_size = tk.IntVar(value=5)         # 高斯滤波核大小
        self.gaussian_sigma = tk.DoubleVar(value=1.0)        # 高斯滤波标准差
        
        self.use_bilateral_blur = tk.BooleanVar(value=False) # 是否使用双边滤波
        self.bilateral_d = tk.IntVar(value=9)                # 双边滤波直径
        self.bilateral_sigma_color = tk.IntVar(value=75)     # 颜色空间标准差
        self.bilateral_sigma_space = tk.IntVar(value=75)     # 坐标空间标准差
        
        self.use_blur = tk.BooleanVar(value=False)           # 是否使用均值滤波
        self.blur_size = tk.IntVar(value=5)                  # 均值滤波核大小
        
        self.use_canny = tk.BooleanVar(value=False)          # 是否使用Canny边缘检测
        self.canny_low = tk.IntVar(value=50)                 # Canny边缘检测低阈值
        

        
        self.contrast = tk.DoubleVar(value=1.0)   # 对比度调整
        self.brightness = tk.IntVar(value=0)      # 亮度调整
        self.sharpness = tk.DoubleVar(value=0.0)  # 锐化程度

        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.tk_image = None

        self.create_widgets()
        self.bind("<Configure>", self.on_resize)
        self.bind("<F11>", self.toggle_fullscreen)  # F11切换全屏

    def create_widgets(self):
        # 左侧控制面板 - 预处理参数
        left_control_frame = tk.Frame(self, width=600, bg="#2d2d30")
        left_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 打开图片按钮
        open_btn = tk.Button(left_control_frame, text="打开图片", command=self.open_image, bg="#007acc", fg="white", font=("微软雅黑", 12))
        open_btn.pack(pady=10, fill=tk.X)

        left_column = tk.Frame(left_control_frame, bg="#2d2d30")
        right_column = tk.Frame(left_control_frame, bg="#2d2d30")
        left_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), expand=True)
        right_column.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), expand=True)
        
        # 左侧列：滤波方法
        filter_frame = tk.LabelFrame(left_column, text="滤波处理参数", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 12))
        filter_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # 中值滤波
        median_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        median_frame.pack(fill=tk.X, pady=5)
        median_check = tk.Checkbutton(median_frame, text="使用中值滤波", variable=self.use_median_blur,
                                     command=self.update_image, bg="#3c3c3c", fg="#d4d4d4")
        median_check.pack(anchor="w")
        
        median_size_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        median_size_frame.pack(fill=tk.X, padx=20)
        median_size_label = tk.Label(median_size_frame, text=f"中值滤波核大小: {self.median_blur_size.get()}", 
                                    anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        median_size_label.pack(anchor="w")
        median_size_scale = tk.Scale(median_size_frame, from_=1, to=21, resolution=2, orient=tk.HORIZONTAL,
                                   variable=self.median_blur_size, 
                                   command=lambda v, l=median_size_label, n="中值滤波核大小": self.update_label(l, n, v),
                                   bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        median_size_scale.pack(fill=tk.X)
        
        # 高斯滤波
        gaussian_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        gaussian_frame.pack(fill=tk.X, pady=5)
        gaussian_check = tk.Checkbutton(gaussian_frame, text="使用高斯滤波", variable=self.use_gaussian_blur,
                                       command=self.update_image, bg="#3c3c3c", fg="#d4d4d4")
        gaussian_check.pack(anchor="w")
        
        gaussian_size_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        gaussian_size_frame.pack(fill=tk.X, padx=20)
        gaussian_size_label = tk.Label(gaussian_size_frame, text=f"高斯滤波核大小: {self.gaussian_blur_size.get()}", 
                                     anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        gaussian_size_label.pack(anchor="w")
        gaussian_size_scale = tk.Scale(gaussian_size_frame, from_=1, to=21, resolution=2, orient=tk.HORIZONTAL,
                                     variable=self.gaussian_blur_size, 
                                     command=lambda v, l=gaussian_size_label, n="高斯滤波核大小": self.update_label(l, n, v),
                                     bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        gaussian_size_scale.pack(fill=tk.X)
        
        gaussian_sigma_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        gaussian_sigma_frame.pack(fill=tk.X, padx=20)
        gaussian_sigma_label = tk.Label(gaussian_sigma_frame, text=f"高斯滤波标准差: {self.gaussian_sigma.get():.1f}", 
                                      anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        gaussian_sigma_label.pack(anchor="w")
        gaussian_sigma_scale = tk.Scale(gaussian_sigma_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                                      variable=self.gaussian_sigma, 
                                      command=lambda v, l=gaussian_sigma_label, n="高斯滤波标准差": self.update_label(l, n, v),
                                      bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        gaussian_sigma_scale.pack(fill=tk.X)
        
        # 双边滤波
        bilateral_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        bilateral_frame.pack(fill=tk.X, pady=5)
        bilateral_check = tk.Checkbutton(bilateral_frame, text="使用双边滤波", variable=self.use_bilateral_blur,
                                       command=self.update_image, bg="#3c3c3c", fg="#d4d4d4")
        bilateral_check.pack(anchor="w")
        
        bilateral_d_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        bilateral_d_frame.pack(fill=tk.X, padx=20)
        bilateral_d_label = tk.Label(bilateral_d_frame, text=f"双边滤波直径: {self.bilateral_d.get()}", 
                                   anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        bilateral_d_label.pack(anchor="w")
        bilateral_d_scale = tk.Scale(bilateral_d_frame, from_=1, to=50, resolution=1, orient=tk.HORIZONTAL,
                                   variable=self.bilateral_d, 
                                   command=lambda v, l=bilateral_d_label, n="双边滤波直径": self.update_label(l, n, v),
                                   bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        bilateral_d_scale.pack(fill=tk.X)
        
        # 均值滤波
        blur_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        blur_frame.pack(fill=tk.X, pady=5)
        blur_check = tk.Checkbutton(blur_frame, text="使用均值滤波", variable=self.use_blur,
                                  command=self.update_image, bg="#3c3c3c", fg="#d4d4d4")
        blur_check.pack(anchor="w")
        
        blur_size_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        blur_size_frame.pack(fill=tk.X, padx=20)
        blur_size_label = tk.Label(blur_size_frame, text=f"均值滤波核大小: {self.blur_size.get()}", 
                                 anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        blur_size_label.pack(anchor="w")
        blur_size_scale = tk.Scale(blur_size_frame, from_=1, to=21, resolution=2, orient=tk.HORIZONTAL,
                                 variable=self.blur_size, 
                                 command=lambda v, l=blur_size_label, n="均值滤波核大小": self.update_label(l, n, v),
                                 bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        blur_size_scale.pack(fill=tk.X)
        
        edge_frame1 = tk.LabelFrame(left_column, text="边缘检测参数", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 12))
        edge_frame1.pack(fill=tk.X, pady=10, padx=5)
        
        # Canny边缘检测
        canny_frame = tk.Frame(edge_frame1, bg="#3c3c3c")
        canny_frame.pack(fill=tk.X, pady=5)
        canny_check = tk.Checkbutton(canny_frame, text="使用Canny边缘检测", variable=self.use_canny,
                                   command=self.update_image, bg="#3c3c3c", fg="#d4d4d4")
        canny_check.pack(anchor="w")
        
        canny_low_frame = tk.Frame(edge_frame1, bg="#3c3c3c")
        canny_low_frame.pack(fill=tk.X, padx=20)
        canny_low_label = tk.Label(canny_low_frame, text=f"Canny低阈值: {self.canny_low.get()}", 
                                 anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        canny_low_label.pack(anchor="w")
        canny_low_scale = tk.Scale(canny_low_frame, from_=1, to=200, resolution=1, orient=tk.HORIZONTAL,
                                 variable=self.canny_low, 
                                 command=lambda v, l=canny_low_label, n="Canny低阈值": self.update_label(l, n, v),
                                 bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        canny_low_scale.pack(fill=tk.X)
        
        edge_frame2 = tk.LabelFrame(right_column, text="阈值处理参数", bg="#f0f0f0", font=("微软雅黑", 12))
        edge_frame2.pack(fill=tk.X, pady=10, padx=5)
        
        # 双边滤波参数
        bilateral_sigma_color_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        bilateral_sigma_color_frame.pack(fill=tk.X, padx=20)
        bilateral_sigma_color_label = tk.Label(bilateral_sigma_color_frame, 
                                             text=f"颜色空间标准差: {self.bilateral_sigma_color.get()}", 
                                             anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        bilateral_sigma_color_label.pack(anchor="w")
        bilateral_sigma_color_scale = tk.Scale(bilateral_sigma_color_frame, from_=1, to=200, resolution=1,
                                             orient=tk.HORIZONTAL, variable=self.bilateral_sigma_color, 
                                             command=lambda v, l=bilateral_sigma_color_label, n="颜色空间标准差": self.update_label(l, n, v),
                                             bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        bilateral_sigma_color_scale.pack(fill=tk.X)
        
        bilateral_sigma_space_frame = tk.Frame(filter_frame, bg="#3c3c3c")
        bilateral_sigma_space_frame.pack(fill=tk.X, padx=20)
        bilateral_sigma_space_label = tk.Label(bilateral_sigma_space_frame, 
                                             text=f"坐标空间标准差: {self.bilateral_sigma_space.get()}", 
                                             anchor="w", bg="#3c3c3c", fg="#d4d4d4", font=("微软雅黑", 10))
        bilateral_sigma_space_label.pack(anchor="w")
        bilateral_sigma_space_scale = tk.Scale(bilateral_sigma_space_frame, from_=1, to=200, resolution=1,
                                             orient=tk.HORIZONTAL, variable=self.bilateral_sigma_space, 
                                             command=lambda v, l=bilateral_sigma_space_label, n="坐标空间标准差": self.update_label(l, n, v),
                                             bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
        bilateral_sigma_space_scale.pack(fill=tk.X)
        

        

        
        # 右侧控制面板 - 霍夫圆参数
        right_control_frame = tk.Frame(self, width=300, bg="#2d2d30")
        right_control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # 中间图像显示
        self.image_label = tk.Label(self, bg="#3c3c3c", relief=tk.SUNKEN, bd=2)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        hough_label = tk.Label(right_control_frame, text="霍夫圆参数", font=("微软雅黑", 12, "bold"), bg="#2d2d30", fg="#d4d4d4")
        hough_label.pack(anchor="w", pady=5)
        
        hough_params = [
            ("图像分辨率倒数(dp)", self.dp, 0.1, 5.0, 0.1),
            ("圆心最小距离(minDist)", self.minDist, 1, 200, 1),
            ("Canny高阈值(param1)", self.param1, 1, 300, 1),
            ("中心点阈值(param2)", self.param2, 1, 300, 1),
            ("最小半径(minRadius)", self.minRadius, 0, 200, 1),
            ("最大半径(maxRadius)", self.maxRadius, 0, 200, 1),
        ]

        for name, var, min_val, max_val, step in hough_params:
            frame = tk.Frame(right_control_frame, bg="#2d2d30")
            frame.pack(fill=tk.X, pady=5)
            label = tk.Label(frame, text=f"{name}: {var.get()}", anchor="w", bg="#2d2d30", fg="#d4d4d4", font=("微软雅黑", 10))
            label.pack(anchor="w")
            scale = tk.Scale(frame, from_=min_val, to=max_val, resolution=step, orient=tk.HORIZONTAL,
                             variable=var, command=lambda v, l=label, n=name: self.update_label(l, n, v),
                             bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
            scale.pack(fill=tk.X)
        
        # 图像增强参数 - 放在霍夫圆参数下方
        enhance_label = tk.Label(right_control_frame, text="图像增强参数", font=("微软雅黑", 12, "bold"), bg="#2d2d30", fg="#d4d4d4")
        enhance_label.pack(anchor="w", pady=10)
        
        enhance_params = [
            ("对比度", self.contrast, 0.1, 3.0, 0.1),
            ("亮度调整", self.brightness, -100, 100, 5),
            ("锐化程度", self.sharpness, 0.0, 3.0, 0.1),
        ]
        
        for name, var, min_val, max_val, step in enhance_params:
            frame = tk.Frame(right_control_frame, bg="#2d2d30")
            frame.pack(fill=tk.X, pady=5)
            label = tk.Label(frame, text=f"{name}: {var.get()}", anchor="w", bg="#2d2d30", fg="#d4d4d4", font=("微软雅黑", 10))
            label.pack(anchor="w")
            scale = tk.Scale(frame, from_=min_val, to=max_val, resolution=step, orient=tk.HORIZONTAL,
                             variable=var, command=lambda v, l=label, n=name: self.update_label(l, n, v),
                             bg="#3c3c3c", troughcolor="#5a5a5a", highlightthickness=0, fg="#d4d4d4", activebackground="#007acc")
            scale.pack(fill=tk.X)

    def update_label(self, label, name, value):
        # 根据参数名称判断是否为浮点数
        is_float = any(keyword in name for keyword in ["图像分辨率", "高斯滤波标准差", "对比度", "锐化程度"])
        label.config(text=f"{name}: {float(value):.1f}" if is_float else f"{name}: {int(float(value))}")
        self.update_image()
    
    def toggle_fullscreen(self, event=None):
        # 切换全屏状态
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)
        # 如果退出全屏，恢复窗口大小
        if not self.fullscreen:
            self.geometry("1000x700")

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("错误", "无法加载图片，请选择有效的图片文件！")
                return
            self.update_image()

    def update_image(self):
        if self.original_image is None:
            return

        # 复制原图
        img = self.original_image.copy()
        
        # 调整对比度和亮度
        contrast = self.contrast.get()
        brightness = self.brightness.get()
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        
        # 应用锐化
        sharpness = self.sharpness.get()
        if sharpness > 0:
            kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness * 8, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用所有选中的滤波方法（按顺序）
        # 均值滤波
        if self.use_blur.get():
            # 确保核大小为奇数
            blur_size = max(1, self.blur_size.get() // 2 * 2 + 1)
            gray = cv2.blur(gray, (blur_size, blur_size))
        
        # 高斯滤波
        if self.use_gaussian_blur.get():
            # 确保核大小为奇数
            gaussian_size = max(1, self.gaussian_blur_size.get() // 2 * 2 + 1)
            gray = cv2.GaussianBlur(gray, (gaussian_size, gaussian_size), self.gaussian_sigma.get())
        
        # 中值滤波
        if self.use_median_blur.get():
            # 确保核大小为奇数
            median_size = max(1, self.median_blur_size.get() // 2 * 2 + 1)
            gray = cv2.medianBlur(gray, median_size)
        
        # 双边滤波
        if self.use_bilateral_blur.get():
            gray = cv2.bilateralFilter(gray, self.bilateral_d.get(), 
                                     self.bilateral_sigma_color.get(), 
                                     self.bilateral_sigma_space.get())
        

        
        # 应用Canny边缘检测
        if self.use_canny.get():
            edges = cv2.Canny(gray, self.canny_low.get(), self.param1.get())
            # 将边缘图与原图混合显示
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            img = cv2.addWeighted(img, 0.7, edges_colored, 0.3, 0)
            # 使用边缘图进行霍夫变换
            gray = edges

        # 霍夫圆检测
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp.get(),
            minDist=self.minDist.get(),
            param1=self.param1.get(),
            param2=self.param2.get(),
            minRadius=self.minRadius.get(),
            maxRadius=self.maxRadius.get() if self.maxRadius.get() > 0 else 0
        )

        # 绘制圆
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        # 调整图像大小以适应窗口
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # 获取图像显示区域的可用大小
        if hasattr(self, 'image_label') and self.image_label.winfo_exists():
            # 考虑控件边框和边距
            available_width = self.image_label.winfo_width() - 20
            available_height = self.image_label.winfo_height() - 20
        else:
            # 默认大小
            available_width = 650
            available_height = 650
        
        # 最小尺寸
        available_width = max(100, available_width)
        available_height = max(100, available_height)
        
        scale = min(available_width / w, available_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 转换为PIL格式并显示
        self.display_image = Image.fromarray(img_resized)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.image_label.config(image=self.tk_image)

    def on_resize(self, event):
        if event.widget == self and self.original_image is not None:
            self.update_image()

if __name__ == "__main__":
    app = HoughCircleTuner()
    app.mainloop()
