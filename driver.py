import os
import cv2

from yolov5_solar_panel.yolov5_model import inference, load_model, inference

path = './yolov5_solar_panel/best.pt'
model, device = load_model(path)

# print(model)
# print(device)

img = cv2.imread('./yolov5_solar_panel/test.jpg')
# cv2.imshow("result", img)
# cv2.waitKey(0)

# image = os.getcwd() + '/yolov5_solar_panel/test.jpg'
output_dict = inference(img, model, device, conf=0.25)
print(output_dict)