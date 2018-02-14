# -*- coding: utf-8 -*-
import sys
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2hed
#from slide_related import convert_2_level_l_int
import cv2, math
from inspect import currentframe, getframeinfo

def generate_color_list_bgr(n_color):

    li_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    if len(li_color) < n_color:
        li_color += [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
    else:
        return li_color

    if len(li_color) < n_color:
        li_color += [(255, 128, 0), (0, 255, 128), (128, 0, 255)]
    else:
        return li_color

    if len(li_color) < n_color:
        li_color += [(128, 255, 0), (0, 128, 255), (255, 0, 128)]
    else:
        return li_color

    if len(li_color) < n_color:
        li_color += [(128, 128, 0), (0, 128, 128), (128, 0, 128)]
    else:
        return li_color

    if len(li_color) < n_color:
        print('You requested too many colors. Use generate_color_list_random_bgr')
        sys.exit(1)



def gen_random_color(r_min, g_min, b_min):
    rr = np.random.randint(r_min, 256)
    gg = np.random.randint(g_min, 256)
    bb = np.random.randint(b_min, 256)
    return (rr, gg, bb)


def otsu_thresholding(im_float, scale_otsu = 1.0):
    print ('threshold_otsu')
    threshold_global_Otsu = threshold_otsu(im_float)
    threshold_global_Otsu *= scale_otsu
    #   thresholding을 한다.
    im_bool = im_float >= threshold_global_Otsu
    return im_bool


def get_hem_eos_from_rgb(im_rgb):
    print ('rgb2hed')
    im_hed = rgb2hed(im_rgb)
    im_h = im_hed[:, :, 0]
    im_e = im_hed[:, :, 1]
    return im_h, im_e


def otsu_hematoxylin(im_rgb, scale_otsu = 1.0):
    im_h, _ = get_hem_eos_from_rgb(im_rgb)
    im_h = np.float32(im_h)
    im_bool = otsu_thresholding(im_h, scale_otsu)
    return im_bool


def compute_optical_density(intensity, intensity_0):
    t1 = np.maximum(1., intensity.astype(float))
    #t6 = np.max(t1)
    #t7 = np.min(t1)
    t2 = t1 / intensity_0
    #t8 = np.max(t2)
    #t9 = np.min(t2)
    t3 = -np.log(t2)
    #t10 = np.max(t3)
    #t11 = np.min(t3)
    return t3


def compute_average_od(red, green, blue, intensity_0):
    d_r = compute_optical_density(red, intensity_0)
    d_g = compute_optical_density(green, intensity_0)
    d_b = compute_optical_density(blue, intensity_0)
    return (d_r + d_g + d_b) / 3.

def compute_average_od_image(pxl, intensity_0):
    #print getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno
    d_rgb = compute_optical_density(pxl, intensity_0)
    #print getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno
    #print(d_rgb.shape)
    d_avg = np.average(d_rgb, axis=2)
    #print getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno
    #d_avg = math.mean(d_rgb)
    return d_avg


def compute_Otsu_mask_of_HE(im_rgba, im_edge_gray, non_zero, b_ccl, scale_otsu = 1.0, level = 0):
    from slide_related import convert_2_level_l_int
    if im_rgba.shape[2] > 3:
        im_rgb = np.array(im_rgba)[:, :, 0:3]
    else:
        im_rgb = im_rgba
    im_thres = otsu_hematoxylin(im_rgb, scale_otsu)
    print ('np.uint8')
    if b_ccl:
        h_l, w_l = im_thres.shape
        #th_r_fill = 0.17
        #th_r_fill = 0.11
        th_r_fill = 0.07
        th_r_w_h = 7
        thick_0 = 600
        #th_area_0 = 240 * 240
        #th_area_0 = 800 * 800
        th_area_0 = 1200 * 1200
        thick_l = convert_2_level_l_int(thick_0, level)
        th_area_l = th_area_0 / (skale * skale)
        im_thres = np.uint8(im_thres)

        #_, contours, _ = cv2.findContours(im_thres, 1, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(im_contour_gray, contours, -1, 255, 1)#, maxLevel=1)
        #cv2.imshow('im_contour_gray_b4', im_contour_gray); cv2.waitKey(1)
        im_line_bgr = np.zeros((im_thres.shape[0], im_thres.shape[1], 3), np.uint8)
        im_edge_gray_after = im_edge_gray.copy()
        minLineLength = min(w_l, h_l) * 0.5
        maxLineGap = min(w_l, h_l) * 0.0001
        th_Hough = int(100 * 2 * 2 * 2 * 1.8)
        lines = \
            cv2.HoughLines(im_edge_gray, 1., np.pi / (180.) , th_Hough)
            # cv2.HoughLinesP(im_edge_gray, 1., np.pi / 180., 100, minLineLength, maxLineGap)
        if lines is not None:
            n_line = len(lines)
            for line in lines:
                #x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]

                rho, theta = line[0][0], line[0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1 = int(x0 + 100000 * (-b))
                y1 = int(y0 + 100000 * (a))
                x2 = int(x0 - 100000 * (-b))
                y2 = int(y0 - 100000 * (a))

                cv2.line(im_thres, (x1, y1), (x2, y2), 0, thick_l)
                cv2.line(im_line_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.line(im_edge_gray_after, (x1, y1), (x2, y2), 0, thick_l)
            cv2.imwrite('im_line_bgr.jpg', im_line_bgr)
            cv2.imshow('im_edge_gray_after', im_edge_gray_after); cv2.waitKey(1)
            cv2.imshow('im_line_bgr', im_line_bgr); cv2.waitKey(1)


        print ('Opening')
        #half_side = 1
        half_side = 1
        side = 2 * half_side + 1
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (side, side))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        im_thres = cv2.morphologyEx(im_thres, cv2.MORPH_OPEN, kernel)
        print ('Finished to open')


        _, contours, _ = cv2.findContours(im_thres, 1, cv2.CHAIN_APPROX_NONE)


        """
        im_thres.fill(0)
        cv2.drawContours(im_thres, contours, -1, 255, -1)#, maxLevel=1)
        cv2.imshow('im_thres', im_thres)
        cv2.waitKey()
        """
        li_cnt_final = []
        x_min = w_l
        y_min = h_l
        x_max = -1
        y_max = -1
        im_blob_bgr = np.zeros(im_rgb.shape, np.uint8)
        for contour in contours:
            #   compute_area
            area = cv2.contourArea(contour)
            if area > th_area_l:
                """
                print ('area : ' + str(area) + ' / ' + str(th_area_l))
                if 73959 == area:
                    a = 0
                im_thres.fill(0)
                cv2.drawContours(im_thres, [contour], -1, 255, -1)  # , maxLevel=1)
                cv2.imshow('im_thres', im_thres)
                cv2.waitKey()
                """
                color_rand = gen_random_color(100, 100, 100)
                cv2.drawContours(im_blob_bgr, [contour], -1, color_rand, -1)  # , maxLevel=1)
                x, y, w, h = cv2.boundingRect(contour)
                r_w_h = float(max(w, h)) / float(min(w, h))
                if r_w_h < th_r_w_h:
                    area_rect = w * h
                    r_fill = area / area_rect
                    if r_fill > th_r_fill:
                        li_cnt_final.append(contour)
                        x_min = min(x, x_min)
                        y_min = min(y, y_min)
                        x_max = max(x + w, x_max)
                        y_max = max(y + h, y_max)
        #cv2.imshow('im_blob_bgr', im_blob_bgr)
        #cv2.waitKey()

        h_margin = h_l / 10.
        w_margin = w_l * 0.2
        p_txt = (int(w_margin), int(h_margin))
        color_font = (255, 255, 255)
        face_font = cv2.FONT_HERSHEY_SIMPLEX
        scale_font = int(round(float(w_l) / 1000.))
        thick_font = int(round(scale_font * 1.5))
        txt_t = 'blobs of area bigger than : ' + str(th_area_l)
        cv2.putText(im_blob_bgr, txt_t, p_txt, face_font, scale_font, color_font, thick_font)

        im_thres.fill(0)
        cv2.drawContours(im_thres, li_cnt_final, -1, non_zero, -1)#, maxLevel=1)
    else:
        im_thres = non_zero * np.uint8(im_thres)
        x_min = 0
        y_min = 0
        x_max = w_l - 1
        y_max = h_l - 1
        im_blob_bgr = None

    return im_thres, x_min, y_min, x_max, y_max, im_blob_bgr

def compute_CxCyD_from_rgb_and_DrDgDb(im_rgb, im_DrDgDb):
    im_Dr = im_DrDgDb[:, :, 0]
    im_Dg = im_DrDgDb[:, :, 1]
    im_Db = im_DrDgDb[:, :, 2]
    im_D = np.average(im_DrDgDb, axis = 2)
    #im_Cx = im_Dr / im_D - 1
    im_Cx = np.divide(im_Dr, im_D) - 1.
    #im_Cy = im_Dg - im_Db / (math.sqrt(3) * im_D)
    im_Cy = np.divide(im_Dg - im_Db, math.sqrt(3) * im_D)
    im_CxCyD = np.dstack((im_Cx, im_Cy, im_D))
    return im_CxCyD


def convert_rgb_2_CxCyD(im_rgb, intensity_0):
    im_DrDgDb = compute_optical_density(im_rgb, intensity_0)
    im_CxCyD = compute_CxCyD_from_rgb_and_DrDgDb(im_rgb, im_DrDgDb)
    return im_CxCyD, im_DrDgDb