import cv2
import numpy as np
import math

# 常量定义
A4W = 21.0
A4H = 29.7
A4S = A4W * A4H
FOCAL_LENGTH = 800.0  # 默认焦距（pix）

# 计算短边的长度
def width_cal(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    return min(width, height) 

def contour_det(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度转化
    
    # 自适应阈值处理，增强鲁棒
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # 使用高斯加权平均计算阈值
        cv2.THRESH_BINARY_INV, 77, 10 
    )

    # 预处理
    thresh_copy = thresh.copy()
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)      
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # 轮廓获取
    if contours:
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:  # 过滤太小的轮廓
                continue
            
            # 多边形逼近
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # 矩形检测
            if len(approx) == 4:
                return approx, thresh_copy, area  # 返回A4纸像素面积
    return None, thresh_copy, 0

def distance_cal(focal_length, pixel_width):
    return (focal_length * A4W) / pixel_width

# 将A4纸区域透视变换为标准矩形
def perspective_transform(image, contour):
    # 将轮廓点排序为（左上、右上、右下、左下）
    pts = contour.reshape(4, 2)               # 轮廓点重塑（方便定位）
    rect = np.zeros((4, 2), dtype="float32")  # 初始化目标矩形点
    center = np.mean(pts, axis=0)             # 中性点坐标
    
    # 根据点与中心的相对位置排序
    for point in pts:
        if point[0] < center[0] and point[1] < center[1]:
            rect[0] = point  # 左上
        elif point[0] > center[0] and point[1] < center[1]:
            rect[1] = point  # 右上
        elif point[0] > center[0] and point[1] > center[1]:
            rect[2] = point  # 右下
        else:
            rect[3] = point  # 左下
    
    # 计算目标矩形尺寸（保持A4比例）
    width = max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[2] - rect[3])
    )
    height = max(
        np.linalg.norm(rect[0] - rect[3]),
        np.linalg.norm(rect[1] - rect[2])
    )
    
    # 目标点指向
    dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # 透视变换
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    return warped

# 计算三点形成的角度便于描述其几何特征
def angle_cal(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 处理数值精度
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def circle_det(warped):
    inverted_gray = cv2.bitwise_not(warped)
    binary_img = cv2.threshold(inverted_gray, 160, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    MIN_CIRCULARITY = 0.80 
    MAX_CIRCULARITY = 1.20

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)   # 是圆形度:)
        if MIN_CIRCULARITY <= circularity and circularity <= MAX_CIRCULARITY:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            diameter = 2 * radius
            pixel_diameter = diameter * circularity
            actual_diameter = (pixel_diameter / warped.shape[1]) * A4W
            if actual_diameter < 9:
                continue
            return {
                "type": "circle",
                "size": actual_diameter,
                "center": center,
                "contour": contour
            }
    return None

def triangle_det(warped):
    inverted = cv2.bitwise_not(warped)
    binary_img = cv2.threshold(inverted, 160, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    triangles = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 3 and is_triangle(approx):
            side_length = triangle_size(approx)
            pixel_side = side_length
            cm_side = (pixel_side / warped.shape[1]) * A4W

            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            triangles.append({
                "type": "triangle",
                "size": cm_side,
                "center": (cX, cY),
                "contour": approx
            })
    
    if triangles:
        return max(triangles, key=lambda x: x["size"])
    return None

def square_det(warped, a4_pixel_area):
    inverted_gray = cv2.bitwise_not(warped)
    binary_img = cv2.threshold(inverted_gray, 160, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    squares = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and is_square(approx):
            square_pixel_area = cv2.contourArea(approx)
            if square_pixel_area < 2:
                continue
            
            if a4_pixel_area > 0:
                area_ratio = square_pixel_area / a4_pixel_area
                square_real_area = A4S * area_ratio
                square_real_side = math.sqrt(square_real_area)
            else:
                square_real_side = square_size_2nd(
                    calculate_side_length(approx), warped)
            
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            squares.append({
                "type": "square",
                "size": square_real_side,
                "center": (cX, cY),
                "contour": approx
            })
    
    if squares:
        return max(squares, key = lambda x: x["size"])
    return None

def is_triangle(contour, side_threshold=0.10):
    points = contour.reshape(3, 2)
    sides = [
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[1]),
        np.linalg.norm(points[0] - points[2])
    ]
    max_side, min_side = max(sides), min(sides)
    aspect_ratio = abs(max_side - min_side) / ((max_side + min_side) / 2)
    if aspect_ratio > side_threshold:
        return False
    for i in range(3):
        a, b, c = points[i], points[(i + 1) % 3], points[(i + 2) % 3]
        angle = angle_cal(a, b, c)
        if not (55 <= angle <= 65): 
            return False
    return True

def is_square(contour, aspect_threshold=0.10):
    sides = []
    for i in range(4):
        pt1 = contour[i][0]
        pt2 = contour[(i + 1) % 4][0]
        sides.append(np.linalg.norm(pt2 - pt1))
    max_side, min_side = max(sides), min(sides)
    aspect_ratio = abs(max_side - min_side) / ((max_side + min_side) / 2)
    if aspect_ratio > aspect_threshold:
        return False
    for i in range(4):
        pt1 = contour[i][0]
        pt2 = contour[(i + 1) % 4][0]
        pt3 = contour[(i + 2) % 4][0]
        angle = angle_cal(pt1, pt2, pt3)
        if not (85 <= angle <= 95): 
            return False
    return True

# 基于A4纸比例计算正方形实际尺寸的方法（作为备选）
def square_size_2nd(contour, warped):
    a4_pixel_width = warped.shape[1]
    
    # 计算边长
    sides = []
    for i in range(4):
        pt1 = contour[i][0]
        pt2 = contour[(i + 1) % 4][0]
        sides.append(np.linalg.norm(pt2 - pt1))
    avg_side = np.mean(sides)
    
    pixel_to_cm = A4W / a4_pixel_width
    return avg_side * pixel_to_cm

def triangle_size(contour):
    points = contour.reshape(3, 2)
    sides_direct = [
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[1]),
        np.linalg.norm(points[0] - points[2])
    ]
    avg_side = np.mean(sides_direct)
    radius = cv2.minEnclosingCircle(contour)[1]
    circum_radius = radius
    theoretical_side = circum_radius * np.sqrt(3)
    area = cv2.contourArea(contour)
    area_side = (4 * area / np.sqrt(3)) ** 0.5
    return (avg_side * 0.6 + theoretical_side * 0.3 + area_side * 0.1)

def result_disp(frame, distance, shape_info, a4_contour):
    frame_copy = frame.copy()
    
    # 绘制A4纸边框
    if a4_contour is not None:
        cv2.drawContours(frame_copy, [a4_contour], -1, (0, 255, 0), 2)
    
    # 显示文本信息
    text_y = 40
    line_height = 40
    
    # 距离信息
    cv2.putText(
        frame_copy, f"D = {distance:.1f} cm", (10, text_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
    )
    text_y += line_height
    
    # 图形形状信息
    if shape_info is not None:
        shape_name = shape_info.get('type', 'unknown')
        shape_name_en = {
            'circle': 'Circle',
            'triangle': 'Triangle',
            'square': 'Square'
        }.get(shape_name, 'Unknown')
        
        cv2.putText(
            frame_copy, f"S = {shape_name_en}", 
            (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
        text_y += line_height
        
        # 图形尺寸信息
        size = shape_info.get('size', 0.0)
        cv2.putText(
            frame_copy, f"X = {size + 0.4:.1f} cm",
            (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
    
    return frame_copy

def main():
    # 初始化相机
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("相机坏了哦")
        return
    
    print("按 'Esc' 退出")
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取帧")
            break
        
        # 检测A4纸
        a4_contour, thresh, a4_pixel_area = contour_det(frame)
        
        if a4_contour is not None:
            # 计算距离
            pixel_width = width_cal(a4_contour)
            distance = distance_cal(FOCAL_LENGTH, pixel_width)
            
            # 距离范围在100-180cm之间
            distance = max(100.0, min(180.0, distance))
            
            # 透视变换
            warped = perspective_transform(thresh, a4_contour)
            
            # 检测图形（先检测圆形，再检测三角形，最后检测正方形）
            shape_info = None
            
            # 先检测圆形（圆度检测）
            circle = circle_det(warped)
            if circle is not None:
                shape_info = circle
            else:
                # 检测三角形（有明确的3个顶点）
                triangle = triangle_det(warped)
                if triangle is not None:
                    shape_info = triangle
                else:
                    # 最后检测正方形（有明确的4个顶点）
                    square = square_det(warped, a4_pixel_area)
                    if square is not None:
                        shape_info = square
            
            # 显示结果
            result_frame = result_disp(frame, distance, shape_info, a4_contour)
            cv2.imshow('Result', result_frame)
        else:
            cv2.imshow('Result', frame)
        
        # 处理按键
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()