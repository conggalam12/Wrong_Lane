import cv2
import numpy as np
import json
def draw_polygon(frame, polygon, color=(0, 255, 0), thickness=2):
    pts = np.array(polygon[0], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness)
def calculate_centroid(polygon):
    """Tính toán tâm của polygon."""
    M = cv2.moments(np.array(polygon))
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def determine_side(point, polygon):
    result = cv2.pointPolygonTest(np.array(polygon[0], np.int32), point, False)
    if result >= 0:
        return "inside"
    else:
        result_2 = cv2.pointPolygonTest(np.array(polygon[1], np.int32), point, False)
        min_y = min([i[1] for i in polygon[1]])
        if result_2 >= 0:
            return "straight"
        else:
            centroid = calculate_centroid(polygon[0]) 
            if centroid[0] > point[0] and point[1]>min_y:
                return "left"
            elif centroid[0] < point[0] and point[1]>min_y:
                return "right"
            else:
                return "straight"

def object_side(bbox, polygon):
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return determine_side(center, polygon)
def read_file_config(path):
    with open(path, 'r') as file:
        data = json.load(file)
    coordinates_1 = data['1']['area']
    coordinates_2 = data['2']['area']
    coordinates_3 = data['2']['area']

    point_1 = [(point['x'], point['y']) for point in coordinates_1]
    point_2 = [(point['x'], point['y']) for point in coordinates_2]
    point_3 = [(point['x'], point['y']) for point in coordinates_3]
    points = [point_1,point_2,point_3]
    return points
if __name__ == "__main__":
    polygon = [(100, 100), (400, 100), (400, 400), (100, 400)]

