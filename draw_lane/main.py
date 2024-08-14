import cv2
import numpy as np
import json

points = []
current_polygon = []
area_polygon = []
data = {}

def draw_polygon(img, points, is_closed=False):
    for i in range(len(points)):
        cv2.circle(img, points[i], 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(img, points[i-1], points[i], (255, 0, 0), 2)
    if is_closed and len(points) == 4:
        cv2.line(img, points[3], points[0], (255, 0, 0), 2)

def mark_point(event, x, y, flags, param):
    global current_polygon, area_polygon, image, temp_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_polygon) < 4:
            current_polygon.append((x, y))
            area_polygon.append({'x': x, 'y': y})
            temp_img = image.copy()
            draw_polygon(temp_img, current_polygon, len(current_polygon) == 4)
            cv2.imshow("Image", temp_img)

image_path = "/home/congnt/congnt/ATIN/test/img_test/lane.jpg" 
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

original = image.copy()
temp_img = image.copy()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mark_point)

count = 0

while True:
    cv2.imshow("Image", temp_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("z"):
        if current_polygon:
            current_polygon.pop()
            area_polygon.pop()
        temp_img = image.copy()
        draw_polygon(temp_img, current_polygon)

    elif key == ord("s"):
        if len(current_polygon) == 4:
            rule_base = input("Enter rule for this polygon: ").split()
            count += 1
            
            # Save to data dictionary
            data[str(count)] = {
                'area': area_polygon.copy(),
                'rule': rule_base
            }
   
            print(f"Data saved for polygon {count}")
            
            current_polygon.clear()
            area_polygon.clear()
            temp_img = image.copy()
        else:
            print("Need exactly 4 points to save.")

    elif key == ord("q"):
        if data:
            with open("../config/data.json", "w") as json_file:
                json.dump(data, json_file, indent=2)
            print("\nAll saved polygons have been written to data.json")
        else:
            print("\nNo data to save.")
        break

cv2.destroyAllWindows()