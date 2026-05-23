import cv2
import numpy as np

A4W = 21.0 
A4H = 29.7

def contour_det(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 77, 10 
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)      
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if contours:     # 识别A4纸
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx
    return None

def perspective_transform(image, contour):    #透视变换
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    center = np.mean(pts, axis=0)
    
    for point in pts:
        if point[0] < center[0] and point[1] < center[1]:
            rect[0] = point
        elif point[0] > center[0] and point[1] < center[1]:
            rect[1] = point
        elif point[0] > center[0] and point[1] > center[1]:
            rect[2] = point
        else:
            rect[3] = point
    
    width = max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[2] - rect[3])
    )
    height = max(
        np.linalg.norm(rect[0] - rect[3]),
        np.linalg.norm(rect[1] - rect[2])
    )
    
    dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    M_inv = cv2.getPerspectiveTransform(dst, rect)
    
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    return warped, M_inv

def inner_shapes_det(warped):    # 识别内部形状
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 77, 10
    )
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    h, w = warped.shape[:2]
    margin_x = int(w * 0.095)
    margin_y = int(h * 0.067)
    
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(inner_mask, (margin_x, margin_y), (w - margin_x, h - margin_y), 255, -1)
    
    masked_thresh = cv2.bitwise_and(mask, inner_mask)
    
    contours = cv2.findContours(masked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if len(contours) == 0:
        return []
    
    return contours

def transform_contour_back(contour, M_inv):
    if len(contour.shape) == 2:
        contour = contour.reshape(-1, 1, 2)
    contour = contour.astype(np.float32)
    transformed = cv2.perspectiveTransform(contour, M_inv)
    return transformed.reshape(-1, 1, 2).astype(np.int32)

def min_square_side_det(contour, warped_width):    # 识别最小正方形边长
    # 计算动态去重阈值
    desired_threshold_cm = 0.1  # 期望的实际阈值（厘米）
    pixels_per_cm = warped_width / A4W
    dynamic_threshold = pixels_per_cm * desired_threshold_cm
    
    # 计算不同场景的动态阈值倍数
    endpoint_threshold = dynamic_threshold * 1.5  # 端点判断阈值
    trend_threshold = dynamic_threshold * 3.0      # 趋势判断阈值
    min_edge_threshold = dynamic_threshold * 2.0   # 短边过滤阈值
    
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    
    if len(approx) < 4:
        return None
    
    points = approx.reshape(-1, 2)
    
    clustered_points = []
    used = [False] * len(points)
    
    for i in range(len(points)):
        if used[i]:
            continue
        cluster = [points[i]]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            dist = np.linalg.norm(points[i] - points[j])
            if dist < dynamic_threshold:
                cluster.append(points[j])
                used[j] = True
        avg_point = np.mean(cluster, axis=0)
        clustered_points.append(avg_point)
    
    points = np.array(clustered_points)
    
    x_coords = sorted(set([int(round(p[0])) for p in points]))
    y_coords = sorted(set([int(round(p[1])) for p in points]))
    
    point_dict = {}
    for p in points:
        key = (int(round(p[0])), int(round(p[1])))
        if key not in point_dict:
            point_dict[key] = []
        point_dict[key].append(p)
    
    # 智能去重：保留长边，同时确保凸变点不丢失
    processed_point_dict = {}
    
    # 1. 处理同一x坐标的点，按y坐标去重，保留长边
    x_groups = {}
    for (x, y) in point_dict.keys():
        if x not in x_groups:
            x_groups[x] = []
        x_groups[x].append(y)
    
    for x, y_list in x_groups.items():
        # 按y坐标排序
        sorted_ys = sorted(y_list)
        
        # 识别凸变点，合并普通点，保留长边
        merged_points = []
        if len(sorted_ys) > 0:
            current_start = sorted_ys[0]
            for i in range(1, len(sorted_ys)):
                distance = abs(sorted_ys[i] - current_start)
                if distance > dynamic_threshold * 2:  # 距离较大，是凸变点，保留
                    merged_points.append(current_start)
                    current_start = sorted_ys[i]
            merged_points.append(current_start)
        
        unique_ys = merged_points
        
        # 保留所有去重后的点
        for y in unique_ys:
            key = (x, y)
            processed_point_dict[key] = point_dict[key]
    
    # 2. 处理同一y坐标的点，按x坐标去重，保留长边
    final_point_dict = {}
    y_groups = {}
    for (x, y) in processed_point_dict.keys():
        if y not in y_groups:
            y_groups[y] = []
        y_groups[y].append(x)
    
    for y, x_list in y_groups.items():
        # 按x坐标排序
        sorted_xs = sorted(x_list)
        
        # 识别凸变点，合并普通点，保留长边
        merged_points = []
        if len(sorted_xs) > 0:
            current_start = sorted_xs[0]
            for i in range(1, len(sorted_xs)):
                distance = abs(sorted_xs[i] - current_start)
                if distance > dynamic_threshold * 2:  # 距离较大，是凸变点，保留
                    merged_points.append(current_start)
                    current_start = sorted_xs[i]
            merged_points.append(current_start)
        
        unique_xs = merged_points
        
        # 保留所有去重后的点
        for x in unique_xs:
            key = (x, y)
            final_point_dict[key] = processed_point_dict[key]
    
    # 更新point_dict为去重后的结果
    point_dict = final_point_dict
    
    # 更新x_coords和y_coords
    x_coords = sorted(set([x for (x, y) in point_dict.keys()]))
    y_coords = sorted(set([y for (x, y) in point_dict.keys()]))
    
    valid_edges = []
    
    for direction in ['top', 'bottom', 'left', 'right']:   # 识别有效边长，每一个方向
        if direction == 'top':
            for x in x_coords:
                y_list = sorted([y for (px, y) in point_dict.keys() if px == x])
                for i in range(len(y_list) - 1):
                    y1, y2 = y_list[i], y_list[i+1]
                    if abs(y1 - y2) < min_edge_threshold:
                        continue
                    
                    p1_is_endpoint = (x - int(endpoint_threshold), y1) not in point_dict
                    p2_is_endpoint = (x + int(endpoint_threshold), y2) not in point_dict
                    
                    is_edge = False
                    if p1_is_endpoint and p2_is_endpoint:
                        is_edge = True
                    elif p1_is_endpoint:  # p1是端点，p2不是端点，看p2的另一边（右侧）
                        if (x + int(trend_threshold), y2) not in point_dict or point_dict[(x + int(trend_threshold), y2)][0][1] > y2:
                            is_edge = True
                    elif p2_is_endpoint:  # p2是端点，p1不是端点，看p1的另一边（左侧）
                        if (x - int(trend_threshold), y1) not in point_dict or point_dict[(x - int(trend_threshold), y1)][0][1] > y1:
                            is_edge = True
                    else:  # 两个都不是端点，看两侧
                        if ((x - int(trend_threshold), y1) not in point_dict or point_dict[(x - int(trend_threshold), y1)][0][1] > y1) and \
                           ((x + int(trend_threshold), y2) not in point_dict or point_dict[(x + int(trend_threshold), y2)][0][1] > y2):
                            is_edge = True
                    
                    if is_edge:
                        edge_length = abs(y2 - y1)
                        edge_length_cm = (edge_length / warped_width) * A4W
                        if 5.0 <= edge_length_cm <= 13.0:
                            valid_edges.append(edge_length_cm)
        
        elif direction == 'bottom':
            for x in x_coords:
                y_list = sorted([y for (px, y) in point_dict.keys() if px == x])
                for i in range(len(y_list) - 1):
                    y1, y2 = y_list[i], y_list[i+1]
                    if abs(y1 - y2) < min_edge_threshold:
                        continue
                    
                    p1_is_endpoint = (x - int(endpoint_threshold), y1) not in point_dict
                    p2_is_endpoint = (x + int(endpoint_threshold), y2) not in point_dict
                    
                    is_edge = False
                    if p1_is_endpoint and p2_is_endpoint:
                        is_edge = True
                    elif p1_is_endpoint:  # p1是端点，p2不是端点，看p2的另一边（右侧）
                        if (x + int(trend_threshold), y2) not in point_dict or point_dict[(x + int(trend_threshold), y2)][0][1] < y2:
                            is_edge = True
                    elif p2_is_endpoint:  # p2是端点，p1不是端点，看p1的另一边（左侧）
                        if (x - int(trend_threshold), y1) not in point_dict or point_dict[(x - int(trend_threshold), y1)][0][1] < y1:
                            is_edge = True
                    else:  # 两个都不是端点，看两侧
                        if ((x - int(trend_threshold), y1) not in point_dict or point_dict[(x - int(trend_threshold), y1)][0][1] < y1) and \
                           ((x + int(trend_threshold), y2) not in point_dict or point_dict[(x + int(trend_threshold), y2)][0][1] < y2):
                            is_edge = True
                    
                    if is_edge:
                        edge_length = abs(y2 - y1)
                        edge_length_cm = (edge_length / warped_width) * A4W
                        if 5.0 <= edge_length_cm <= 13.0:
                            valid_edges.append(edge_length_cm)
        
        elif direction == 'left':
            for y in y_coords:
                x_list = sorted([x for (x, py) in point_dict.keys() if py == y])
                for i in range(len(x_list) - 1):
                    x1, x2 = x_list[i], x_list[i+1]
                    if abs(x1 - x2) < min_edge_threshold:
                        continue
                    
                    p1_is_endpoint = (x1, y - int(endpoint_threshold)) not in point_dict
                    p2_is_endpoint = (x2, y + int(endpoint_threshold)) not in point_dict
                    
                    is_edge = False
                    if p1_is_endpoint and p2_is_endpoint:
                        is_edge = True
                    elif p1_is_endpoint:  # p1是端点，p2不是端点，看p2的另一边（下方）
                        if (x2, y + int(trend_threshold)) not in point_dict or point_dict[(x2, y + int(trend_threshold))][0][0] > x2:
                            is_edge = True
                    elif p2_is_endpoint:  # p2是端点，p1不是端点，看p1的另一边（上方）
                        if (x1, y - int(trend_threshold)) not in point_dict or point_dict[(x1, y - int(trend_threshold))][0][0] > x1:
                            is_edge = True
                    else:  # 两个都不是端点，看两侧
                        if ((x1, y - int(trend_threshold)) not in point_dict or point_dict[(x1, y - int(trend_threshold))][0][0] > x1) and \
                           ((x2, y + int(trend_threshold)) not in point_dict or point_dict[(x2, y + int(trend_threshold))][0][0] > x2):
                            is_edge = True
                    
                    if is_edge:
                        edge_length = abs(x2 - x1)
                        edge_length_cm = (edge_length / warped_width) * A4W
                        if 5.0 <= edge_length_cm <= 13.0:
                            valid_edges.append(edge_length_cm)
        
        elif direction == 'right':
            for y in y_coords:
                x_list = sorted([x for (x, py) in point_dict.keys() if py == y])
                for i in range(len(x_list) - 1):
                    x1, x2 = x_list[i], x_list[i+1]
                    if abs(x1 - x2) < min_edge_threshold:
                        continue
                    
                    p1_is_endpoint = (x1, y - int(endpoint_threshold)) not in point_dict
                    p2_is_endpoint = (x2, y + int(endpoint_threshold)) not in point_dict
                    
                    is_edge = False
                    if p1_is_endpoint and p2_is_endpoint:
                        is_edge = True
                    elif p1_is_endpoint:  # p1是端点，p2不是端点，看p2的另一边（下方）
                        if (x2, y + int(trend_threshold)) not in point_dict or point_dict[(x2, y + int(trend_threshold))][0][0] < x2:
                            is_edge = True
                    elif p2_is_endpoint:  # p2是端点，p1不是端点，看p1的另一边（上方）
                        if (x1, y - int(trend_threshold)) not in point_dict or point_dict[(x1, y - int(trend_threshold))][0][0] < x1:
                            is_edge = True
                    else:  # 两个都不是端点，看两侧
                        if ((x1, y - int(trend_threshold)) not in point_dict or point_dict[(x1, y - int(trend_threshold))][0][0] < x1) and \
                           ((x2, y + int(trend_threshold)) not in point_dict or point_dict[(x2, y + int(trend_threshold))][0][0] < x2):
                            is_edge = True
                    
                    if is_edge:
                        edge_length = abs(x2 - x1)
                        edge_length_cm = (edge_length / warped_width) * A4W
                        if 5.0 <= edge_length_cm <= 13.0:
                            valid_edges.append(edge_length_cm)
    
    if len(valid_edges) == 0:
        return None
    
    return min(valid_edges)

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("相机坏了哦")
        return
    
    print("按 'Esc' 退出")
    
    stable_value = None
    new_value_count = 0
    new_value_sum = 0.0
    threshold_frames = 8
    similarity_threshold = 0.05
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取帧")
            break
        
        a4_contour = contour_det(frame)
        
        if a4_contour is not None:
            cv2.drawContours(frame, [a4_contour], -1, (0, 255, 0), 2)
            
            warped, M_inv = perspective_transform(frame, a4_contour)
            warped_width = warped.shape[1]
            
            inner_contours = inner_shapes_det(warped)
            
            min_side = None
            for cnt in inner_contours:
                cnt_back = transform_contour_back(cnt, M_inv)
                cv2.drawContours(frame, [cnt_back], -1, (255, 0, 0), 2)
                
                side = min_square_side_det(cnt, warped_width)
                if side is not None:
                    if min_side is None or side < min_side:
                        min_side = side
            
            if min_side is not None:
                if stable_value is None:
                    stable_value = min_side
                else:
                    if abs(min_side - stable_value) / stable_value < similarity_threshold:
                        new_value_count = 0
                        new_value_sum = 0.0
                    else:
                        new_value_count += 1
                        new_value_sum += min_side
                        
                        if new_value_count >= threshold_frames:
                            stable_value = new_value_sum / new_value_count
                            new_value_count = 0
                            new_value_sum = 0.0
                
                if new_value_count > 0:
                    cv2.putText(frame, f"Min Side: {stable_value + 0.7:.1f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(frame, f".", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Min Side: {stable_value + 0.7:.1f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imshow('Result', frame)
        
        key = cv2.waitKey(10)
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()