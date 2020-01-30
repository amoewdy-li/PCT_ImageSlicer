import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import csv
import math


def transform_coordinates(rectangles):
    new_rec = []
    for rect in rectangles:
        dot_x = []
        dot_y = []
        for dot in rect:
            dot_x.append(dot[0])
            dot_y.append(dot[1])
        new_rec.append([min(dot_x), min(dot_y), max(dot_x) - min(dot_x), max(dot_y) - min(dot_y)])
    return new_rec


def get_bounding_rect(rect1, rect2):
    dot_x = []
    dot_y = []
    for rect in rect1:
        dot_x.append(rect[0])
        dot_y.append(rect[1])
    for rect in rect2:
        dot_x.append(rect[0])
        dot_y.append(rect[1])
    rect3 = [
        [min(dot_x), min(dot_y)],
        [max(dot_x), min(dot_y)],
        [min(dot_x), max(dot_y)],
        [max(dot_x), max(dot_y)]
    ]
    return rect3


class ImageSlicer:
    def __init__(self, img, min_rect, min_distance, stroke):
        self.min_rect = min_rect
        self.min_distance = min_distance
        self.stroke = stroke
        self.img = img

    def get_binary_img(self, crop_img):
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        th, threshed = cv2.threshold(gaussian_blur, 160, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blurred_binary = cv2.GaussianBlur(threshed, (3, 3), 0)
        return blurred_binary

    def get_distance(self, rect1, rect2):
        flag = False
        for dot1 in rect1:
            for dot2 in rect2:
                dist = math.sqrt((dot2[0] - dot1[0]) ** 2 + (dot2[1] - dot1[1]) ** 2)
                if dist <= self.min_distance:
                    flag = True
        return flag

    def rect_filter(self, rect_mask):
        rect_all = []
        for rect in rect_mask:
            rect_all.append([
                [rect[0], rect[1]],
                [rect[0] + rect[2], rect[1]],
                [rect[0], rect[1] + rect[3]],
                [rect[0] + rect[2], rect[1] + rect[3]]
            ])

        for i in range(len(rect_all) - 1):
            for j in range(i + 1, len(rect_all)):
                dist_flag = self.get_distance(rect_all[i], rect_all[j])
                if dist_flag:
                    bounding_rec = get_bounding_rect(rect_all[i], rect_all[j])
                    rect_all.remove(rect_all[j])
                    rect_all.remove(rect_all[i])
                    rect_all.append(bounding_rec)
                    i = i - 1
                    break
        return rect_all

    def color_detect(self):
        rect_mask = []
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        upper_rec = {'blue': (130, 230, 255), 'pale': (135, 100, 255)}
        lower_rec = {'blue': (70, 160, 200), 'pale': (65, 5, 200)}
        print('image loaded')

        for key, value in lower_rec.items():
            kernel = np.ones((16, 16), np.uint8)
            mask = cv2.inRange(hsv_img, lower_rec[key], upper_rec[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            for cnt in contours:
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                if key == 'blue':
                    cv2.drawContours(self.img, [cnt], 0, (0, 150, 000), 2)
                    approx_rec_blue = cv2.approxPolyDP(cnt, epsilon, True)
                    plt.imshow(self.img)
                    a, b, c, d = cv2.boundingRect(approx_rec_blue)
                    cv2.rectangle(self.img, (a, b), (a + c, b + d), (0, 255, 0), 2)
                    print(key, a, b, c, d)
                if key == 'pale':
                    # cv2.drawContours(img, [cnt], 0, (150, 150, 000), 2)
                    approx_rec_blue = cv2.approxPolyDP(cnt, epsilon, True)
                    plt.imshow(self.img)
                    a, b, c, d = cv2.boundingRect(approx_rec_blue)
                    print(key, a, b, c, d)
                    if c >= self.min_rect and d >= self.min_rect:
                        # cv2.rectangle(self.img, (a, b), (a + c, b + d), (255, 0, 0), 4)
                        rect_mask.append([a, b, c, d])

            # cv2.imwrite("Jan29.png", self.img)
        return rect_mask

    def crop_image(self, rect_mask):
        crop_imgs = []
        i = 0
        rect_filtered = self.rect_filter(rect_mask)
        rect_filtered = transform_coordinates(rect_filtered)
        for mask in rect_filtered:
            crop_img = self.img[mask[1]+self.stroke:mask[1]+mask[3]-self.stroke, mask[0]+self.stroke:mask[0]+mask[2]-self.stroke]
            file_name_relative = "crop_binary" + str(i) + ".png"
            binary_img = self.get_binary_img(crop_img)
            cv2.imwrite(file_name_relative, binary_img)
            print(file_name_relative)
            crop_imgs.append(binary_img)
            i += 1
        return crop_imgs


if __name__ == "__main__":
    image_path = '/Users/sixuan/Desktop/Logo_Sketch/template_test_final.png'
    img = cv2.imread(image_path)
    image_slicer_test = ImageSlicer(img, 100, 80, 25)
    rect_masks = image_slicer_test.color_detect()
    crop_imgs = image_slicer_test.crop_image(rect_masks)

















