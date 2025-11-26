import os,cv2,numpy as np

def color_detection(image):
    if image is None or image.size == 0:
        print("警告：无效的图像输入")
        return None, None
        
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print(f"颜色空间转换错误: {e}")
        return None, None
    
    lower_blue = np.array([95, 40, 40])    
    upper_blue = np.array([135, 255, 255]) 
    lower_red1 = np.array([0, 120, 70])   
    upper_red1 = np.array([8, 255, 255])   
    lower_red2 = np.array([170, 120, 70])  
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

def circle_detection(image):
    if image is None or image.size == 0:
        print("无法加载图像或图像为空:(")
        return None
    
    result = image.copy()
    
    try:
        color_mask, color_info = color_detection(image)
        if color_mask is None or color_info is None:
            return result
        
        # 中值滤波去除椒盐噪声
        preprocessed = cv2.medianBlur(image, 5)
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        
        # 将颜色掩码与灰度图结合
        masked_gray = cv2.bitwise_and(gray, gray, mask=color_mask)
        
        # 增加高斯模糊的核大小，平滑图像
        blurred = cv2.GaussianBlur(masked_gray, (13, 13), 5)

        circles = cv2.HoughCircles(
            blurred, method=cv2.HOUGH_GRADIENT_ALT,
            dp=1.1,        # 降低dp提高检测灵敏度
            minDist=60,    # 降低minDist以检测靠得更近的圆
            param1=30,     # Canny边缘检测阈值
            param2=0.2,    # 降低质量阈值，提高检测灵敏度
            minRadius=50,  # 最小半径
            maxRadius=135  # 最大半径
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # amount = len(circles[0, :])
            # print(f"检测到 {amount} 个圆形")
            
            # 当圆心重合时保留较大半径的圆
            if len(circles[0, :]) > 0:
                # 先按半径降序排序，这样在去重时优先保留大半径的圆
                # 使用明确的元组结构存储圆信息，避免数据错乱（这玩意修了好久我服了）
                circles_sorted = sorted([(int(circle[0]), int(circle[1]), int(circle[2])) for circle in circles[0, :]], 
                                       key=lambda x: -x[2])
                unique_circles = []
                
                for circle in circles_sorted:
                    center_x, center_y, radius = circle
                    is_duplicate = False
                    
                    # 检查与已保留圆的距离
                    for uc in unique_circles:
                        uc_x, uc_y, uc_r = uc
                        # 使用平方距离比较，避免平方根计算，提高效率
                        distance_sq = (center_x - uc_x)**2 + (center_y - uc_y)**2
                        min_radius_sq = (min(uc_r, radius) * 0.7)**2
                        
                        # 如果圆心距离小于较小半径的70%，视为重复
                        if distance_sq < min_radius_sq:
                            is_duplicate = True
                            break
                    
                    # 否则添加
                    if not is_duplicate:
                        unique_circles.append((center_x, center_y, radius))
                                
            # 初始化颜色计数器
            blue_count = 0
            red_count = 0
            
            # 使用去重后的圆列表进行绘制
            for circle in unique_circles:
                # 直接解包元组，避免索引访问错误（之前是会把颜色搞错的问题）
                center_x, center_y, radius = circle
                
                h, w = image.shape[:2]
                # 直接计算坐标，避免使用可能导致类型问题的np.clip
                x1 = max(0, center_x - radius)
                y1 = max(0, center_y - radius)
                x2 = min(w, center_x + radius)
                y2 = min(h, center_y + radius)

                # 确保坐标有效
                if x2 <= x1:
                    x2 = x1 + 1
                if y2 <= y1:
                    y2 = y1 + 1
                
                # 提取ROI并计算颜色像素数
                try:
                    roi_blue_mask = color_info['blue'][0][y1:y2, x1:x2]
                    roi_red_mask = color_info['red'][0][y1:y2, x1:x2]
                    blue_amount = cv2.countNonZero(roi_blue_mask)
                    red_amount = cv2.countNonZero(roi_red_mask)
                except Exception as e:
                    print(f"处理ROI时出错: {e}")
                    blue_amount = 0
                    red_amount = 0

                """这个方法...留作纪念算了，不能识别到边框处的
                total_pixels = (y2 - y1) * (x2 - x1)

                # 设置颜色识别阈值
                color_threshold = 0.4 
                
                # 确定颜色
                color_name, color = None, None
                if blue_amount / total_pixels > color_threshold:
                    color_name, color = color_info['blue'][1], color_info['blue'][2]
                elif red_amount / total_pixels > color_threshold:
                    color_name, color = color_info['red'][1], color_info['red'][2]
                """

                if blue_amount > red_amount:
                    color_name, color = color_info['blue'][1], color_info['blue'][2]
                    blue_count += 1
                else:
                    color_name, color = color_info['red'][1], color_info['red'][2]
                    red_count += 1

                if color_name is not None and color_name != "Unknown":
                    # 绘制圆形
                    cv2.circle(result, (center_x, center_y), radius, color, 2)

                    # 绘制圆心（绿点）
                    cv2.circle(result, (center_x, center_y), 3, (0, 255, 0), -1)

                    # 计算坐标
                    x1, y1 = int(center_x - radius), int(center_y - radius)
                    x2, y2 = int(center_x + radius), int(center_y + radius)

                    # 写标记
                    text = f"{color_name} ({center_x},{center_y}), R={radius}"
                    cv2.putText(result, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)
            
            # 在屏幕右上角显示颜色计数
            h, w = result.shape[:2]
            text_blue = f"Blue Balls: {blue_count}"
            text_red = f"Red Balls: {red_count}"
            
            # 绘制计数文本
            cv2.putText(result, text_blue, (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)
            
            cv2.putText(result, text_red, (w - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        else:
            print("未检测到圆形")
    except Exception as e:
        print(f"圆形检测过程中出错: {e}")
        warning_text = "WARNING: No circles detected"
        cv2.putText(result, warning_text, (w - 250, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        return image.copy()

    return result

def main():
    video_path = 'D:/PYTHON/CV/video1.mp4'

    if not os.path.exists(video_path):
        print("文件找不到")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 暂停状态变量
    is_paused = False
    
    while True:
        try:
            # 只有在未暂停状态下才读取新帧
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("视频读取结束或出错")
                    break
                
                # 确保帧有效
                if frame is None or frame.size == 0:
                    print("警告：无效帧")
                    continue
                
                processed_frame = circle_detection(frame)
            
            if 'processed_frame' in locals() and processed_frame is not None and processed_frame.size > 0:
                if is_paused:
                    frame_copy = processed_frame.copy()
                    cv2.putText(frame_copy, "PAUSED", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Result', frame_copy)
                else:
                    cv2.imshow('Result', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("用户中断播放")
                break
            elif key == 32:  # 空格键暂停/继续
                is_paused = not is_paused
                if is_paused:
                    print("视频已暂停，按空格键继续")
                else:
                    print("视频继续播放")
            
        except Exception as e:
            print(f'errno:{e}')
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    