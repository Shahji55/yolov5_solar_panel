import cv2 
import numpy as np
import math
import glob
import os

# files = ['DJI_0220', 'DJI_0258', 'DJI_0259', 'DJI_0265', 'DJI_0271', 'DJI_0272', 'DJI_0273', 'DJI_0276', 'DJI_0279', 'DJI_0280', 'DJI_0281', 'DJI_0303', 'DJI_0324', 'DJI_0325', 'DJI_0327', 'DJI_0329', 'DJI_0330', 
        # 'DJI_0332', 'DJI_0333', 'DJI_0334', 'DJI_0404', 'DJI_0337']

# files_grabbed = [glob.glob(e) for e in ['./data/images_annotations/*.png', './data/images_annotations/*.jpg']]

cwd = os.getcwd()
new_anno_dir = cwd + '/data/new_data_annotations/'

dir = cwd + '/data/images_annotations/'
os.chdir(dir)
# types = ('./data/images_annotations/*.png', './data/images_annotations/*.jpg') # the tuple of file types
types = ('*.png', '*.jpg')
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(files))

print(len(files_grabbed))
# print(files_grabbed)


for file in files_grabbed:

    # img= cv2.imread("./data/data_annot/" + file + '.png')
    print(file)
    img = cv2.imread(file)
    # print(img.shape)

    height,width= img.shape[0], img.shape[1]


    heightUpdated=int(height/3)
    widthUpdated=int(width/3)

    # print(heightUpdated, widthUpdated)



    img1=img[:heightUpdated,:widthUpdated]
    img2=img[heightUpdated:heightUpdated*2, :widthUpdated]
    img3=img[heightUpdated*2:heightUpdated*3, :widthUpdated]
    img4=img[:heightUpdated,widthUpdated:widthUpdated*2]
    img5=img[heightUpdated:heightUpdated*2, widthUpdated:widthUpdated*2]
    img6=img[heightUpdated*2:heightUpdated*3, widthUpdated:widthUpdated*2]
    img7=img[:heightUpdated,widthUpdated*2:]
    img8=img[heightUpdated:heightUpdated*2,widthUpdated*2:]
    img9=img[heightUpdated*2:heightUpdated*3,widthUpdated*2:]

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img3 = cv2.resize(img3, (img1.shape[1], img1.shape[0]))
    img4 = cv2.resize(img4, (img1.shape[1], img1.shape[0]))
    img5 = cv2.resize(img5, (img1.shape[1], img1.shape[0]))
    img6 = cv2.resize(img6, (img1.shape[1], img1.shape[0]))
    img7 = cv2.resize(img7, (img1.shape[1], img1.shape[0]))
    img8 = cv2.resize(img8, (img1.shape[1], img1.shape[0]))
    img9 = cv2.resize(img9, (img1.shape[1], img1.shape[0]))

    img_dixt= {'0x0': img1, '1x0': img2, '2x0': img3, '0x1': img4 ,'1x1': img5 ,'2x1': img6, '0x2': img7, '1x2': img8, '2x2': img9}
    img_points = {'0x0':[], '1x0':[], '2x0':[], '0x1':[], '1x1':[], '2x1':[], '0x2':[], '1x2':[], '2x2':[]}

    file_split = file.split(".")
    print(file_split[0])
    text_file = file_split[0] + '.txt'
    
    # text_file = '.' + file_split[1] + '.txt'
    print("Reading: ", text_file)
    print('\n')
    text_file = open(text_file, 'r')
    lines = text_file.readlines()

    for line in lines:
        line = line.split(' ')

        cl = int(line[0])
        x = math.floor(float(line[1]) * width)
        y = math.floor(float(line[2]) * height)
        w = math.floor(float(line[3]) * width)
        h = math.floor(float(line[4].strip('\n')) * height)

        x1_orig = int(x - (w/2))
        y1_orig = int(y - (h/2))

        

        # import pdb; pdb.set_trace()
        # import matplotlib.pyplot as plt
        # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)


        # cv2.imshow("Hello", img)
        # cv2.waitKey(11111110)
        # break
        # print("original annotations")
        # print(x1, y1, w, h)

        x2_orig = int(x1_orig + w)
        y2_orig = int(y1_orig + h)




        # print(x1, x2, y1, y2)
        # print(img1.shape)
        sliceidx1 = int(x1_orig/img1.shape[1])
        sliceidy1 = int(y1_orig/img1.shape[0])
        
        img_grid1 = str(sliceidy1) + 'x' + str(sliceidx1)

        sliceidx2 = int(x2_orig/img1.shape[1])
        sliceidy2 = int(y2_orig/img1.shape[0])

        img_grid2 = str(sliceidy2) + 'x' + str(sliceidx2)

        # print(img_grid)

        x1 = x1_orig - (sliceidx1)*img1.shape[1]
        x2 = x2_orig - (sliceidx1)*img1.shape[1]
        y1 = y1_orig - (sliceidy1)*img1.shape[0]
        y2 = y2_orig - (sliceidy1)*img1.shape[0]

        w = x2 - x1
        h = y2 - y1

        t1 = x1 + w
        t2 = y1 + h
        
        if t1 > img1.shape[1]:
            w = (w - (t1 - img1.shape[1])) - 1
            #w = w - 10
        if t2 > img1.shape[0]:
            h = (h - (t2 - img1.shape[0])) - 1
            # h = h - 10
        
        if x1 < 0:
            x1 = 0
            
        if y1 < 0:
            y1 = 0

        # cv2.rectangle(img_dixt[img_grid1], (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        x1 = (x1 + (w/2)) / img1.shape[1]
        y1 = (y1 + (h/2)) / img1.shape[0]
        w = w / img1.shape[1]
        h = h / img1.shape[0]

        temp = [str(cl), str(x1), str(y1), str(w), str(h)]

        if temp in img_points[img_grid1]:
            pass
        else:
            img_points[img_grid1].append(temp)
        
        

        if img_grid1 != img_grid2 and '3' not in img_grid2:
            x1 = x1_orig - (sliceidx2)*img1.shape[1]
            x2 = x2_orig - (sliceidx2)*img1.shape[1]
            y1 = y1_orig - (sliceidy2)*img1.shape[0]
            y2 = y2_orig - (sliceidy2)*img1.shape[0]

            w = x2 - x1
            h = y2 - y1

            t1 = x1 + w
            t2 = y1 + h
            
            if t1 > img1.shape[1]:
                w = w - 10

            if t2 > img1.shape[0]:
                h = h - 10

            if x1 < 0:
                # print(x2,x1)
                x1 = 0
            
            if y1 < 0:
                y1 = 0

            # cv2.rectangle(img_dixt[img_grid2], (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

            x1 = (x1 + (w/2)) / img1.shape[1]
            y1 = (y1 + (h/2)) / img1.shape[0]
            w = w / img1.shape[1]
            h = h / img1.shape[0]
            
            temp = [str(cl), str(x1), str(y1), str(w), str(h)]
            
            if temp in img_points[img_grid2]:
                pass
            else:
                img_points[img_grid2].append(temp)
            


        # print(sliceidy, sliceidx)
        # # break
        
        # print("before")
        # print(x1, y1, w, h)
        
        # import pdb; pdb.set_trace()
        # x1 = x1 + (img1.shape[1] / 2)
        
        # x1 = x1 / img1.shape[1]
        # y1 = y1 + (img1.shape[0] / 2)
        # y1 = y1 / img1.shape[0]
        # w = w / img1.shape[1]
        # h = h / img1.shape[0]
        # import pdb; pdb.set_trace()
        
        # break
        # import pdb; pdb.set_trace()
        # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        # cv2.putText
        
        # print("after")
        # print(x1, y1, w, h)
        # print(x1, x2, y1, y2)
        # print(h)
        # break

# plt.imshow(img1)
# plt.imshow(img9)
# plt.show()


    for k, v in img_dixt.items():
        if len(img_points[k]) == 0:
            pass
        else:
            cv2.imwrite(new_anno_dir + file_split[0] + '_' + k + '.png', v)
            
            with open(new_anno_dir + file_split[0] + '_' + k + '.txt', 'w') as anno:
                for point in img_points[k]:
                    anno.write(point[0] + ' ' + point[1] + ' ' + point[2] + ' ' + point[3] + ' ' + point[4] + '\n')

