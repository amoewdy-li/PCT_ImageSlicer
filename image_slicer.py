import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import csv
import math


def get_upsample_img(img):
    width = 5*img.shape[1]
    height = 5*img.shape[0]
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    return resized

def get_dots(rect):
    rect_dot = [
        [rect[0], rect[1]],
        [rect[0] + rect[2], rect[1]],
        [rect[0], rect[1] + rect[3]],
        [rect[0] + rect[2], rect[1] + rect[3]]
    ]
    return rect_dot


class ImageSlicer:
    def __init__(self, img, min_rect, min_distance, stroke):
        self.min_rect = min_rect
        self.min_distance = min_distance
        self.stroke = stroke
        self.img = img

    def get_binary_img(self, crop_img):
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # TODO change kernel size when binarize images for higher img quality.
        gaussian_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        th, threshed = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blurred_binary = cv2.GaussianBlur(threshed, (3, 3), 0)
        return blurred_binary

    def get_rec_from_stack(self, stack):
        new_rect = []
        point_set = []
        for rect in stack:
            points = get_dots(rect)
            point_set.append(points)
        array = np.array(point_set)
        new_array_flatten = array.flatten()
        dot_x = []
        dot_y = []
        for i in range(len(new_array_flatten)):
            if i % 2 == 0:
                dot_x.append(new_array_flatten[i])
            else:
                dot_y.append(new_array_flatten[i])
        new_rect = [min(dot_x), min(dot_y), max(dot_x)-min(dot_x), max(dot_y)-min(dot_y)]
        return new_rect

    def rect_filter_contig(self, rect1, rect2):
        flag = False
        rect1_dot = get_dots(rect1)
        rect2_dot = get_dots(rect2)
        for dot1 in rect1_dot:
            for dot2 in rect2_dot:
                dist = math.sqrt((dot2[0] - dot1[0]) ** 2 + (dot2[1] - dot1[1]) ** 2)
                if dist <= self.min_distance:
                    flag = True
        return flag

    def rect_filter_overlap(self, rect1, rect2):
        flag = False
        rect1_dot = get_dots(rect1)
        for dot in rect1_dot:
            if(
              (rect2[0] <= dot[0] <= rect2[0]+rect2[2]) and (rect2[1] <= dot[1] <= rect2[1]+rect2[3])
            ):
                flag = True
        return flag

    def rect_filter_merge(self, rect_mask):
        array_length = len(rect_mask)
        i = 0
        print('array_length', array_length)
        while i < array_length:
            stack = []
            j = i+1
            while j < array_length:
                if self.rect_filter_contig(rect_mask[i], rect_mask[j]) \
                        or self.rect_filter_overlap(rect_mask[i], rect_mask[j]):
                    stack.append(rect_mask[j])
                j += 1
            if not stack:
                i += 1
            else:
                stack.append(rect_mask[i])
                bounding_rec = self.get_rec_from_stack(stack)
                rect_mask.append(bounding_rec)
                array_length += 1
                array_length -= len(stack)
                for rect in stack:
                    rect_mask.remove(rect)
                print("merged")
        return rect_mask

    def color_detect(self):
        rect_mask = []
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        upper_rec = {'blue': (130, 230, 255), 'pale': (135, 100, 255), 'gray': (180, 240, 252)}
        lower_rec = {'blue': (70, 160, 200), 'pale': (65, 5, 200), 'gray': (1, 1, 80)}
        print('image loaded')

        for key, value in lower_rec.items():
            kernel = np.ones((8, 8), np.uint8)
            mask = cv2.inRange(hsv_img, lower_rec[key], upper_rec[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            for cnt in contours:
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                # if key == 'blue':
                #     cv2.drawContours(self.img, [cnt], 0, (0, 150, 000), 2)
                #     approx_rec_blue = cv2.approxPolyDP(cnt, epsilon, True)
                #     plt.imshow(self.img)
                #     a, b, c, d = cv2.boundingRect(approx_rec_blue)
                #     cv2.rectangle(self.img, (a, b), (a + c, b + d), (0, 255, 0), 2)
                #     print(key, a, b, c, d)
                if key == 'gray':
                    cv2.drawContours(img, [cnt], 0, (150, 150, 000), 2)
                    approx_rec_blue = cv2.approxPolyDP(cnt, epsilon, True)
                    # plt.imshow(self.img)
                    a, b, c, d = cv2.boundingRect(approx_rec_blue)
                    print(key, a, b, c, d)
                    if c >= self.min_rect and d >= self.min_rect:
                        cv2.rectangle(self.img, (a, b), (a + c, b + d), (255, 0, 0), 4)
                        rect_mask.append([a, b, c, d])

            cv2.imwrite("test_gray_border_8.png", self.img)
        return rect_mask

    def crop_image(self, rect_mask):
        crop_imgs = []
        i = 0
        for mask in rect_mask:
            cv2.rectangle(self.img, (mask[0], mask[1]), (mask[0] + mask[2], mask[1] + mask[3]), (0, 0, 255), 4)
            cropped_img = self.img[mask[1] + self.stroke:mask[1] + mask[3] - self.stroke, mask[0] + self.stroke:mask[0] + mask[2] - self.stroke]
            file_name_relative = "crop_binary" + str(i) + ".png"
            upsample_img = get_upsample_img(cropped_img)
            binary_img = self.get_binary_img(upsample_img)
            cv2.imwrite(file_name_relative, binary_img)
            print(file_name_relative)
            crop_imgs.append(binary_img)
            i += 1
        cv2.imwrite("test_gray_border_filtered_8.png", self.img)
        return crop_imgs


if __name__ == "__main__":
    image_path = '/Users/sixuan/Desktop/Logo_Sketch/template_test3.png'
    img = cv2.imread(image_path)
    image_slicer_test = ImageSlicer(img, 50, 80, 15)
    rect_masks = image_slicer_test.color_detect()
    filtered_rect_masks = image_slicer_test.rect_filter_merge(rect_masks)
    crop_img = image_slicer_test.crop_image(filtered_rect_masks)


















