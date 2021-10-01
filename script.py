import os
import cv2
import random

# Function to get current working directory
def get_cwd():
    cwd = os.getcwd()
    return cwd

def get_crops():
    cwd = get_cwd()
    print(cwd)
    # folder_path = cwd + '/data/images/'
    # annotation_path = cwd + '/data/annotations/'

    folder_path = cwd + '/data/images_annotations/'
    annotation_path = cwd + '/data/images_annotations/'

    write_path = cwd + '/data/saved_crops/'

    print(folder_path)

    for filename in sorted(os.listdir(folder_path)):
        name = filename.split('.')
        print(filename)
        print(name)
        if '.jpg' or '.png' in filename:
            print("------------------------------------------")
            print("Opening file: ", name[0])
            print("------------------------------------------")
            img = cv2.imread(os.path.join(folder_path, filename))
            # imL = cv2.resize(img, (960, 540))
            # cv2.imshow("image", imL)
            # cv2.waitKey(0)

            dh, dw, _ = img.shape

            for annotation_file in sorted(os.listdir(annotation_path)):
                print("Annotation file: ", annotation_file)
                annotation_file_sub = annotation_file.split('.')
                print("Annotation file sub, ", annotation_file_sub)
                if name[0] in annotation_file_sub[0]:
                    print("Filename substring found")
                    file_path = os.path.join(annotation_path, annotation_file)
                    print(file_path)
                    file = open(file_path, "r+")
                    data = file.readlines()
                    file.close()

                    for line in data:
                        panel_id, x, y, w, h = map(float, line.split(' '))
                        x1, y1, x2, y2 = yolo_bbox_to_bbox(x, y, w, h, dw, dh)
                        cropped_img = img[y1:y2, x1:x2]

                        # cv2.imshow("crops", cropped_img)
                        # cv2.waitKey(0)

                        color_0 = (255, 0, 0) # blue
                        color_1 = (0,255,0) # green
                        color_2 = (0, 0, 255) # red
                        thickness = 1

                        # Panel
                        if panel_id == 0:
                            cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)), color_0 , thickness)
                        # Pallet_full
                        elif panel_id == 1:
                            cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)), color_1 , thickness)
                        # Pallet_empty
                        elif panel_id == 2:
                            cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)), color_2 , thickness)

                        file_to_write = write_path + filename
                        cv2.imwrite(file_to_write, img)
                        
                        '''
                        tl = 3 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
                        color = [random.randint(0, 255) for _ in range(3)]
                        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
                        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        '''

                        # imS = cv2.resize(img, (960, 540))

                  # cv2.imshow("bboxes", img)
                  # cv2.waitKey(0) 

                  # cv2.destroyAllWindows()

                   # cv2.imwrite("tmp.jpg", img)

# Method to draw bounding boxes on detections for single image and then save result
def draw_specific_file_bbox():
    cwd = get_cwd()
    print(cwd)
    # folder_path = cwd + '/data/images_annotations/'
    # filename = 'DJI_0233_Crop1.jpg'
    # text_file = 'DJI_0233_Crop1.txt'

    folder_path = cwd + '/data/split_test/'
    filename = 'DJI_0200_crop1_2x1.png'
    text_file = 'DJI_0200_crop1_2x1.txt'


    img = cv2.imread(os.path.join(folder_path, filename))

    dh, dw, _ = img.shape
    
    text_file_path = folder_path + '/' + text_file
    print(text_file_path)
    file = open(text_file_path, "r+")
    data = file.readlines()
    file.close()

    write_path = folder_path = cwd + '/data/saved_crops/test/'
    file_to_write = write_path + filename

    for line in data:
        panel_id, x, y, w, h = map(float, line.split(' '))
        x1, y1, x2, y2 = yolo_bbox_to_bbox(x, y, w, h, dw, dh)
        cropped_img = img[y1:y2, x1:x2]

        # cv2.imshow("crops", cropped_img)
        # cv2.waitKey(0)

        # color = (255, 0, 0) # blue
        # color = (0,255,0) # green
        color = (0, 0, 255) # red
        thickness = 1
        cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)), color , thickness)

        cv2.imwrite(file_to_write, img)
        
        '''
        tl = 3 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        '''

        # imS = cv2.resize(img, (960, 540))

    # cv2.imshow("bboxes", img)
    # cv2.waitKey(0) 

    # cv2.destroyAllWindows()

# Method to draw bounding boxes on detections for all images in the directory
# and then save results
def draw_bboxes_save_result():
    cwd = get_cwd()
    print(cwd)

    folder_path = cwd + '/data/images_annotations/'

    write_path = cwd + '/data/saved_crops/'

    print(folder_path)

    for filename in sorted(os.listdir(folder_path)):
        name = filename.split('.')
        print(filename)
        print(name)
        if '.txt' not in filename:
            print("------------------------------------------")
            print("Opening file: ", name[0])
            print("------------------------------------------")
            img = cv2.imread(os.path.join(folder_path, filename))

            dh, dw, _ = img.shape

            file_path = folder_path + name[0] + '.txt'
            print(file_path)
            file = open(file_path, "r+")
            data = file.readlines()
            file.close()

            for line in data:
                panel_id, x, y, w, h = map(float, line.split(' '))
                x1, y1, x2, y2 = yolo_bbox_to_bbox(x, y, w, h, dw, dh)

                
                color_0 = (0, 0, 255) # red for panel
                color_1 = (0,255,0) # green for pallet_full
                color_2 = (255, 0, 0) # blue for pallet_empty
                thickness = 1

                # Panel
                if panel_id == 0:
                    cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)), color_0 , thickness)
                # Pallet_full
                elif panel_id == 1:
                    cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)), color_1 , thickness)
                # Pallet_empty
                elif panel_id == 2:
                    cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)), color_2 , thickness)

            print("----Writing file-----")        

            file_to_write = write_path + filename
            cv2.imwrite(file_to_write, img)

            print("SUCCESS")
        
        elif '.txt' in filename:
            pass
                    


# Function to convert yolo bbox coordinates into opencv format
def yolo_bbox_to_bbox(x, y, w, h, dw, dh):
    x1 = int((x-w/2) * dw)
    y1 = int((y-h/2) * dh)
    x2 = int((x+w/2) * dw)
    y2 = int((y+h/2) * dh)

    if x1 < 0:
        x1 = 0
    if y1 > dw - 1:
        y1 = dw - 1
    if x2 < 0:
        x2 = 0
    if y2 > dh - 1:
        y2 = dh - 1

    return x1, y1, x2, y2

if __name__ == "__main__":
    draw_bboxes_save_result()
    # get_specific_file_crop()