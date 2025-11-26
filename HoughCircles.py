# 使用前先去100行修改path

import os,cv2,numpy as np

def color_detection(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2

    all_mask = blue_mask | red_mask
    
    color_info = {
        'blue': (blue_mask, "Blue", (255, 0, 0)),
        'red': (red_mask, "Red", (0, 0, 255))
    }
    
    return all_mask, color_info

def circle_detection(image_path, min_radius, max_radius):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return None
    
    result = image.copy()
    color_mask, color_info = color_detection(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=color_mask)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)
    
    # 使用霍夫变换检测圆
    circles = cv2.HoughCircles(
        blurred,
        method = cv2.HOUGH_GRADIENT_ALT,
        dp = 1.5,                   # 降低累加器分辨率，提高检测灵敏度
        minDist = 40,               # 缩小圆心最小距离，允许检测密集分布的圆
        param1 = 25,                # 降低Canny边缘检测阈值
        param2 = 0.4,               # 降低质量阈值，提高检测率
        minRadius = min_radius,     # 减小最小半径，检测更小的圆
        maxRadius = max_radius      # 调整为更合理的最大半径
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))        
        for circle in circles[0, :]:
            center_x, center_y, radius = circle[0], circle[1], circle[2]
            h, w = image.shape[:2]
            x1 = max(0, center_x - radius)
            y1 = max(0, center_y - radius)
            x2 = min(w, center_x + radius)
            y2 = min(h, center_y + radius)
            
            # 获取圆形区域内的颜色掩码
            roi_blue_mask = color_info['blue'][0][y1:y2, x1:x2]
            roi_red_mask = color_info['red'][0][y1:y2, x1:x2]
            
            # 计算颜色像素数量
            blue_amount = cv2.countNonZero(roi_blue_mask)
            red_amount = cv2.countNonZero(roi_red_mask)
            
            if blue_amount > red_amount:
                color_name, color = color_info['blue'][1], color_info['blue'][2]
            else:
                color_name, color = color_info['red'][1], color_info['red'][2]
            
            # 绘制圆形
            cv2.circle(result, (center_x, center_y), radius, color, 2)
            
            # 绘制圆心（绿点）
            cv2.circle(result, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # 计算坐标
            x1, y1 = int(center_x - radius), int(center_y - radius)
            x2, y2 = int(center_x + radius), int(center_y + radius)
            
            # 书写其他参数
            text = f"{color_name} ({center_x},{center_y}), R={radius}"
            cv2.putText(result, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
    else:
        print("未检测到圆形")
    
    return result

def main():

    im_path = 'D:/PYTHON/CV/photo1.jpg'

    if not os.path.exists(im_path):
        print("文件找不到，请按照代码提示修改路径:)")
        return
    res = circle_detection(im_path, min_radius=100, max_radius=233)
    if res is not None:
        cv2.imshow('Result', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()