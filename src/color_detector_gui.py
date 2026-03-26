import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class VisionSoftware(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("多功能视觉分析系统 (Vision Analysis System)")
        self.geometry("1100x700")
        self.minsize(800, 600)
        
        self.current_theme = "dark"
        
        self.image_path = None
        self.cap = None
        self.original_image = None
        self.base_resized_image = None
        self.display_image = None
        
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        
        self.is_video = False
        self.total_frames = 1
        
        self.setup_styles()
        self.build_ui()
        self.apply_theme()
        
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def setup_styles(self):
        self.style = ttk.Style(self)
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')

    def apply_theme(self):
        if self.current_theme == "dark":
            colors = {
                "bg": "#2b2b2b", "fg": "#ffffff", "panel": "#323232",
                "canvas": "#1e1e1e", "btn": "#4b4d4f", "btn_active": "#5b5d5f",
                "border": "#555555"
            }
        else:
            colors = {
                "bg": "#f5f5f5", "fg": "#333333", "panel": "#ffffff",
                "canvas": "#e8e8e8", "btn": "#e0e0e0", "btn_active": "#d0d0d0",
                "border": "#cccccc"
            }
            
        self.configure(bg=colors["bg"])
        self.style.configure('TFrame', background=colors["panel"])
        self.style.configure('Main.TFrame', background=colors["bg"])
        self.style.configure('TLabel', background=colors["panel"], foreground=colors["fg"])
        self.style.configure('TButton', background=colors["btn"], foreground=colors["fg"], borderwidth=0, focuscolor=colors["btn"])
        self.style.map('TButton', background=[('active', colors["btn_active"])])
        self.style.configure('TLabelframe', background=colors["panel"], foreground=colors["fg"], bordercolor=colors["border"])
        self.style.configure('TLabelframe.Label', background=colors["panel"], foreground=colors["fg"], font=('微软雅黑', 10, 'bold'))
        self.style.configure('TRadiobutton', background=colors["panel"], foreground=colors["fg"], focuscolor=colors["panel"])
        self.style.configure('TCheckbutton', background=colors["panel"], foreground=colors["fg"], focuscolor=colors["panel"])
        self.style.configure('TScale', background=colors["panel"], troughcolor=colors["bg"])
        
        self.top_frame.configure(style='Main.TFrame')
        self.center_frame.configure(style='Main.TFrame')
        self.bottom_frame.configure(style='Main.TFrame')
        self.right_frame.configure(style='TFrame')
        self.canvas.configure(bg=colors["canvas"], highlightthickness=0)
        
        self.update_display()

    def build_ui(self):
        self.top_frame = ttk.Frame(self)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(self.top_frame, text="打开图片", command=self.open_image, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.top_frame, text="打开视频", command=self.open_video, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.top_frame, text="切换主题", command=self.toggle_theme, width=15).pack(side=tk.RIGHT, padx=5)
        
        self.bottom_frame = ttk.Frame(self)
        ttk.Label(self.bottom_frame, text="视频进度:").pack(side=tk.LEFT, padx=5)
        self.video_slider = ttk.Scale(self.bottom_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_video_frame)
        self.video_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.center_frame = ttk.Frame(self)
        self.center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.right_frame = ttk.Frame(self.center_frame, width=280)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.right_frame.pack_propagate(False)
        
        self.canvas_frame = ttk.Frame(self.center_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.mode_var = tk.StringVar(value="mouse")
        mode_lf = ttk.LabelFrame(self.right_frame, text="工作模式")
        mode_lf.pack(fill=tk.X, padx=10, pady=10)
        ttk.Radiobutton(mode_lf, text="跟随鼠标 (查看单点信息)", variable=self.mode_var, value="mouse", command=self.on_mode_change).pack(anchor=tk.W, padx=10, pady=8)
        ttk.Radiobutton(mode_lf, text="矩形框选 (查看区域统计)", variable=self.mode_var, value="rect", command=self.on_mode_change).pack(anchor=tk.W, padx=10, pady=8)
        
        info_lf = ttk.LabelFrame(self.right_frame, text="参数信息")
        info_lf.pack(fill=tk.X, padx=10, pady=10)
        self.info_pos = ttk.Label(info_lf, text="位置/区域: -", font=('Consolas', 10))
        self.info_pos.pack(anchor=tk.W, padx=10, pady=5)
        self.info_rgb = ttk.Label(info_lf, text="RGB: -", font=('Consolas', 10), justify=tk.LEFT)
        self.info_rgb.pack(anchor=tk.W, padx=10, pady=5)
        self.info_hsv = ttk.Label(info_lf, text="HSV: -", font=('Consolas', 10), justify=tk.LEFT)
        self.info_hsv.pack(anchor=tk.W, padx=10, pady=5)
        
        self.edge_frame = ttk.LabelFrame(self.right_frame, text="边缘检测 (Canny)")
        self.edge_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.edge_frame, text="启用边缘检测 (仅在框选区域内)", variable=self.edge_var, command=self.update_display).pack(anchor=tk.W, padx=10, pady=5)
        
        ttk.Label(self.edge_frame, text="最小阈值:").pack(anchor=tk.W, padx=10, pady=(5,0))
        self.canny_min = tk.IntVar(value=100)
        ttk.Scale(self.edge_frame, from_=0, to=255, variable=self.canny_min, command=self.on_canny_change).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(self.edge_frame, text="最大阈值:").pack(anchor=tk.W, padx=10, pady=(5,0))
        self.canny_max = tk.IntVar(value=200)
        ttk.Scale(self.edge_frame, from_=0, to=255, variable=self.canny_max, command=self.on_canny_change).pack(fill=tk.X, padx=10, pady=5)

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()

    def on_mode_change(self):
        self.start_x = self.end_x = self.start_y = self.end_y = 0
        if self.mode_var.get() == "mouse":
            self.edge_frame.pack_forget()
        else:
            self.edge_frame.pack(fill=tk.X, padx=10, pady=10)
        self.update_display()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.is_video = False
            self.bottom_frame.pack_forget()
            img = cv2.imread(path)
            if img is not None:
                self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.start_x = self.end_x = self.start_y = self.end_y = 0
                self.update_base_resized()
                self.update_display()

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.mov")])
        if path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if self.cap.isOpened():
                self.is_video = True
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_slider.config(to=max(1, self.total_frames-1))
                self.video_slider.set(0)
                self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
                self.start_x = self.end_x = self.start_y = self.end_y = 0
                self.update_video_frame(0)

    def update_video_frame(self, val):
        if self.is_video and self.cap:
            frame_no = int(float(val))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            if ret:
                self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_base_resized()
                self.update_display()
                if self.mode_var.get() == "rect":
                    self.update_rect_info()

    def on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        if self.original_image is not None:
            self.update_base_resized()
            self.update_display()
        else:
            self.update_display()

    def update_base_resized(self):
        if self.original_image is None or not hasattr(self, 'canvas_width'):
            return
        h, w, _ = self.original_image.shape
        ratio = min(self.canvas_width/w, self.canvas_height/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        if new_w <= 0 or new_h <= 0:
            return
        self.scale_ratio = ratio
        self.display_offset_x = (self.canvas_width - new_w) // 2
        self.display_offset_y = (self.canvas_height - new_h) // 2
        self.base_resized_image = cv2.resize(self.original_image, (new_w, new_h))

    def update_display(self):
        self.canvas.delete("all")
        if self.original_image is None or self.base_resized_image is None:
            text_color = "#888888" if self.current_theme == "dark" else "#aaaaaa"
            w = getattr(self, 'canvas_width', 800)
            h = getattr(self, 'canvas_height', 600)
            self.canvas.create_text(w//2, h//2, text="请点击上方按钮打开图片或视频文件\n(Please open an image or video file)", font=("微软雅黑", 14), fill=text_color, justify=tk.CENTER)
            return

        disp_img = self.base_resized_image.copy()
        
        if self.mode_var.get() == "rect" and self.edge_var.get() and self.start_x != self.end_x and self.start_y != self.end_y:
            rx1 = max(0, min(self.start_x, self.end_x) - self.display_offset_x)
            rx2 = min(disp_img.shape[1], max(self.start_x, self.end_x) - self.display_offset_x)
            ry1 = max(0, min(self.start_y, self.end_y) - self.display_offset_y)
            ry2 = min(disp_img.shape[0], max(self.start_y, self.end_y) - self.display_offset_y)
            
            if rx2 > rx1 and ry2 > ry1:
                roi = disp_img[ry1:ry2, rx1:rx2]
                edges = cv2.Canny(roi, self.canny_min.get(), self.canny_max.get())
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                disp_img[ry1:ry2, rx1:rx2] = edges_colored

        img_pil = Image.fromarray(disp_img)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(self.display_offset_x, self.display_offset_y, anchor=tk.NW, image=self.tk_image)
        
        if self.mode_var.get() == "rect" and self.start_x != self.end_x and self.start_y != self.end_y:
            self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="#ff3333", width=2, tags="rect")

    def on_canny_change(self, val=None):
        if self.mode_var.get() == "rect" and self.edge_var.get():
            self.update_display()

    def screen_to_image(self, sx, sy):
        if not hasattr(self, 'scale_ratio'): return 0, 0
        ix = int((sx - self.display_offset_x) / self.scale_ratio)
        iy = int((sy - self.display_offset_y) / self.scale_ratio)
        return ix, iy

    def on_mouse_move(self, event):
        if self.original_image is None: return
        
        if self.mode_var.get() == "mouse":
            ix, iy = self.screen_to_image(event.x, event.y)
            h, w, _ = self.original_image.shape
            
            self.canvas.delete("crosshair")
            if 0 <= ix < w and 0 <= iy < h:
                r, g, b = self.original_image[iy, ix]
                hsv_img = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)
                h_val, s_val, v_val = hsv_img[0][0]
                
                self.info_pos.config(text=f"位置: X:{ix} Y:{iy}")
                self.info_rgb.config(text=f"RGB: ({r}, {g}, {b})")
                self.info_hsv.config(text=f"HSV: ({h_val}, {s_val}, {v_val})")
                
                self.canvas.create_line(event.x-15, event.y, event.x+15, event.y, fill="#00ff00", width=1, tags="crosshair")
                self.canvas.create_line(event.x, event.y-15, event.x, event.y+15, fill="#00ff00", width=1, tags="crosshair")

    def on_mouse_down(self, event):
        if self.mode_var.get() == "rect" and self.original_image is not None:
            self.start_x = self.end_x = event.x
            self.start_y = self.end_y = event.y

    def on_mouse_drag(self, event):
        if self.mode_var.get() == "rect" and self.original_image is not None:
            self.end_x = event.x
            self.end_y = event.y
            self.update_rect_info()
            if self.edge_var.get():
                self.update_display()
            else:
                self.canvas.delete("rect")
                self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="#ff3333", width=2, tags="rect")

    def on_mouse_up(self, event):
        if self.mode_var.get() == "rect" and self.original_image is not None:
            self.end_x = event.x
            self.end_y = event.y
            self.update_rect_info()
            if self.edge_var.get():
                self.update_display()

    def update_rect_info(self):
        if self.original_image is None: return
        ix1, iy1 = self.screen_to_image(self.start_x, self.start_y)
        ix2, iy2 = self.screen_to_image(self.end_x, self.end_y)
        
        ix1, ix2 = min(ix1, ix2), max(ix1, ix2)
        iy1, iy2 = min(iy1, iy2), max(iy1, iy2)
        
        h, w, _ = self.original_image.shape
        ix1 = max(0, ix1); iy1 = max(0, iy1)
        ix2 = min(w, ix2); iy2 = min(h, iy2)
        
        if ix2 > ix1 and iy2 > iy1:
            roi = self.original_image[iy1:iy2, ix1:ix2]
            
            r_min, r_max, r_mean = roi[:,:,0].min(), roi[:,:,0].max(), int(roi[:,:,0].mean())
            g_min, g_max, g_mean = roi[:,:,1].min(), roi[:,:,1].max(), int(roi[:,:,1].mean())
            b_min, b_max, b_mean = roi[:,:,2].min(), roi[:,:,2].max(), int(roi[:,:,2].mean())
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            h_min, h_max, h_mean = hsv_roi[:,:,0].min(), hsv_roi[:,:,0].max(), int(hsv_roi[:,:,0].mean())
            s_min, s_max, s_mean = hsv_roi[:,:,1].min(), hsv_roi[:,:,1].max(), int(hsv_roi[:,:,1].mean())
            v_min, v_max, v_mean = hsv_roi[:,:,2].min(), hsv_roi[:,:,2].max(), int(hsv_roi[:,:,2].mean())
            
            self.info_pos.config(text=f"区域: {ix2-ix1}x{iy2-iy1} px")
            self.info_rgb.config(text=f"RGB 平均:\n({r_mean}, {g_mean}, {b_mean})\n\n范围 (Min-Max):\nR: {r_min} - {r_max}\nG: {g_min} - {g_max}\nB: {b_min} - {b_max}")
            self.info_hsv.config(text=f"HSV 平均:\n({h_mean}, {s_mean}, {v_mean})\n\n范围 (Min-Max):\nH: {h_min} - {h_max}\nS: {s_min} - {s_max}\nV: {v_min} - {v_max}")

if __name__ == "__main__":
    app = VisionSoftware()
    app.mainloop()
