#-*- coding: utf-8 -*

import numpy as np
import os, cv2, collections

from time import time
from PIL import Image
from os.path import join, splitext
from skimage.transform.integral import integrate, integral_image
#from slide_related import make_tile_mask_edge_threshold, check_if_tumor_slide_2, get_scale
from .system_related import get_exact_file_name_from_path, flatten
from .color_related import  otsu_thresholding, get_hem_eos_from_rgb




def get_sum_of_rectangle(im_int, x, y, w, h):
    x_f = int(round(x))
    y_f = int(round(y))
    x_t = int(round(x + w)) - 1
    y_t = int(round(y + h)) - 1
    area =  integrate(im_int, y_f, x_f, y_t, x_t)
    if isinstance(area, (collections.Sequence, np.ndarray)):
        return area[0]
    else:
        return area






def get_image_size_from_list_of_contour(li_li_point, margin):

    x_margin, y_margin = margin
    li_xy = flatten(li_li_point)
    li_x, li_y = li_xy[::2], li_xy[1::2]
    w_img = int(round(max(li_x) + x_margin))
    h_img = int(round(max(li_y) + y_margin))
    return (w_img, h_img)

def convert_list_of_contour_2_opencv_contours(li_li_point):
    contours = []
    for li_point in li_li_point:
        li_point_int = [[int(round(point[0])), int(round(point[1]))] for point in li_point]
        contour = np.array(li_point_int, dtype=np.int32)
        contours.append(contour)
#        contours = [numpy.array([[1, 1], [10, 50], [50, 50]], dtype=numpy.int32),
#                numpy.array([[99, 99], [99, 60], [60, 99]], dtype=numpy.int32)]
    return contours

#def get_opencv_contours_from_xml(fn_xml_0, level):
def get_opencv_contours_from_xml(fn_xml_0, skale):
    from slide_related import make_list_of_contour_l_from_xml_0
    #li_li_point_l = make_list_of_contour_l_from_xml_0(fn_xml_0, level)
    li_li_point_l = make_list_of_contour_l_from_xml_0(fn_xml_0, skale)
    contours_l = convert_list_of_contour_2_opencv_contours(li_li_point_l)
    return contours_l, li_li_point_l


def overlay_mask(im_rgb_masked, im_01_mask, kolor, thick_blob, shall_fill_hole = False):

    #   find contour
    li_blob_tumor = get_tumor_blobs(im_01_mask)
    if shall_fill_hole:
        im_th = np.zeros(im_rgb_masked.shape[:2], np.uint8)
        cv2.drawContours(im_th, li_blob_tumor, -1, 255, thick_blob)  # , maxLevel=1)
        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        li_blob_tumor = get_tumor_blobs(im_out)
        cv2.drawContours(im_rgb_masked, li_blob_tumor, -1, kolor, thick_blob)  # , maxLevel=1)
    else:
        cv2.drawContours(im_rgb_masked, li_blob_tumor, -1, kolor, thick_blob)  # , maxLevel=1)

    return im_rgb_masked


def get_masked_WSI(im_rgb_l, fn_slide, dir_mask, level, kolor,
                   thick_blob_l, postfix_ext_mask):
    if not check_if_tumor_slide_2(fn_slide, dir_mask, postfix_ext_mask):
        print('This is normal slide. Tumor mask can not be obtained from this slide : '
              + fn_slide)
        return im_rgb_l
    im_rgb_masked_l = im_rgb_l.copy()
    fn_only = get_exact_file_name_from_path(fn_slide)
    im_01_mask_l, w_m_l, h_m_l = load_mask(fn_only, dir_mask, postfix_ext_mask, level)
    '''
    thick_blob_0 = 50
    thick_blob_l = convert_2_level_l_int(thick_blob_0, level)
    '''
    im_rgb_masked_l = overlay_mask(im_rgb_masked_l, im_01_mask_l, kolor, thick_blob_l)

    return im_rgb_masked_l


def get_masked_WSI_xml(im_rgb_l, fn_slide, dir_xml_0, level, margin_l,
                       kolor, thick_blob_l, postfix_ext_mask):
    if not check_if_tumor_slide_2(fn_slide, dir_xml_0, postfix_ext_mask):
        print('This is normal slide. Tumor mask can not be obtained from this slide : '
              + fn_slide)
        return im_rgb_l
    fn_only = get_exact_file_name_from_path(fn_slide)
    fn_xml_0 = join(dir_xml_0, fn_only + '.xml')
    return get_masked_WSI_xml_2(im_rgb_l, fn_xml_0, level, margin_l, kolor, thick_blob_l)


def get_masked_WSI_xml_2(im_rgb_l, fn_xml_0, level, margin_l, kolor, thick_blob_l):
    im_rgb_masked_l = im_rgb_l.copy()
    im_01_mask_l, w_m_l, h_m_l = load_mask_xml(fn_xml_0, margin_l, level)
    im_rgb_masked_l = overlay_mask(im_rgb_masked_l, im_01_mask_l, kolor,
                                   thick_blob_l, False)
    return im_rgb_masked_l


def hue_sat_otsu_01(im_rgb, scale_otsu):
    im_hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV_FULL)
    im_h, im_s, _ = cv2.split(im_hsv)
    im_01_h = otsu_thresholding(im_h, scale_otsu)
    im_01_s = otsu_thresholding(im_s, scale_otsu)
    '''
    im_0255_h = 255 * im_01_h
    im_0255_s = 255 * im_01_s
    cv2.imshow('im_0255_h', im_0255_h.astype(np.uint8))
    cv2.imshow('im_0255_s', im_0255_s.astype(np.uint8))
    cv2.waitKey()
    '''
    return im_01_h, im_01_s


def h_and_s_otsu_01(im_rgb, scale_otsu):
    im_01_h, im_01_s = hue_sat_otsu_01(im_rgb, scale_otsu)
    im_01_h_and_s = im_01_h & im_01_s
    return im_01_h_and_s

def h_or_s_otsu_01(im_rgb, scale_otsu):
    im_01_h, im_01_s = hue_sat_otsu_01(im_rgb, scale_otsu)
    im_01_h_or_s = im_01_h | im_01_s
    return im_01_h_or_s

def h_and_s_otsu_01_integral(im_rgb, scale_otsu):

    im_01_h_and_s = h_and_s_otsu_01(im_rgb, scale_otsu)
    '''
    im_0255_h_and_s = 255 * im_01_h_and_s
    cv2.imshow('im_0255_h_and_e', im_0255_h_and_s.astype(np.uint8))
    cv2.waitKey()
    '''
    im_uint64_int_h_and_s = integral_image(im_01_h_and_s)
    return im_uint64_int_h_and_s, im_01_h_and_s

def h_or_s_otsu_01_integral(im_rgb, scale_otsu):
    im_01_h_or_s = h_or_s_otsu_01(im_rgb, scale_otsu)
    im_uint64_int_h_or_s = integral_image(im_01_h_or_s)
    return im_uint64_int_h_or_s, im_01_h_or_s


def h_and_e_otsu_01(im_rgb, scale_otsu):
    im_h, im_e = get_hem_eos_from_rgb(im_rgb)
    im_01_h = otsu_thresholding(im_h, scale_otsu)
    im_01_e = otsu_thresholding(im_e, scale_otsu)
    im_01_h_and_e = im_01_h & im_01_e
    '''
    im_0255_h = 255 * im_01_h
    im_0255_e = 255 * im_01_e
    cv2.imshow('im_0255_h', im_0255_h.astype(np.uint8))
    cv2.imshow('im_0255_e', im_0255_e.astype(np.uint8))
    cv2.waitKey()

    #im_01_h_or_e = im_01_h | im_01_e
    im_0255_h_and_e = 255 * im_01_h_and_e
    cv2.imshow('im_0255_h_and_e', im_0255_h_and_e.astype(np.uint8))
    cv2.waitKey()
    '''

    return im_01_h_and_e

def h_and_e_otsu_01_integral(im_rgb, scale_otsu):
    im_01_h_and_e = h_and_e_otsu_01(im_rgb, scale_otsu)
    im_uint64_int_h_and_e = integral_image(im_01_h_and_e)
    return im_uint64_int_h_and_e, im_01_h_and_e


def h_or_e_otsu_01(im_rgb, scale_otsu):
    im_h, im_e = get_hem_eos_from_rgb(im_rgb)
    im_01_h = otsu_thresholding(im_h, scale_otsu)
    im_01_e = otsu_thresholding(im_e, scale_otsu)
    im_01_h_or_e = im_01_h | im_01_e
    return im_01_h_or_e

def h_or_e_otsu_01_integral(im_rgb, scale_otsu):
    im_01_h_or_e = h_or_e_otsu_01(im_rgb, scale_otsu)
    im_uint64_int_h_or_e = integral_image(im_01_h_or_e)
    return im_uint64_int_h_or_e, im_01_h_or_e


def binarize_h_and_s_otsu_integral(im_rgb, th, size_tile, scale_otsu):
    start = time.time()
    im_uint64_int_h_and_s, im_01_h_and_s = h_and_s_otsu_01_integral(im_rgb, scale_otsu)
    im_0255_mask_h_and_s_int = make_tile_mask_edge_threshold(im_uint64_int_h_and_s, th, size_tile)
    end = time.time()
    lap_sec = end - start
    #cv2.imwrite('im_0255_mask_h_and_s_int.jpg', im_0255_mask_h_and_s_int)
    return im_0255_mask_h_and_s_int, lap_sec



def is_this_belong_2_blob(im_int, x, y, wid, hei, th_occ_rate):

    h_int, w_int = im_int.shape
    if x + wid < w_int and y + hei < h_int:
        #   점유도를 구한다.
        #im_integral, y_from, x_from, y_to, x_to
        area = get_sum_of_rectangle(im_int, x, y, wid, hei)
        area_tile = wid * hei
        r_occu = float(area) / float(area_tile)
    else:
        r_occu = 0
    #   점유도가 충분히 크면
    return r_occu >= th_occ_rate, r_occu










def is_this_patch_below_threshold(im_e_i, th_n_pxl, x_f, y_f, wid, hei):
    n_pixel = get_sum_of_rectangle(im_e_i, x_f, y_f, wid, hei)
    return n_pixel < th_n_pxl

def is_this_patch_below_or_equal_to_threshold(im_e_i, th_n_pxl, x_f, y_f, wid, hei):
    n_pixel = get_sum_of_rectangle(im_e_i, x_f, y_f, wid, hei)
    return n_pixel <= th_n_pxl

def is_this_patch_above_threshold(im_e_i, th_n_pxl, x_f, y_f, wid, hei):
    return (not is_this_patch_below_or_equal_to_threshold(im_e_i, th_n_pxl, x_f, y_f, wid, hei))

def is_this_patch_above_or_equal_to_threshold(im_e_i, th_n_pxl, x_f, y_f, wid, hei):
    return (not is_this_patch_below_threshold(im_e_i, th_n_pxl, x_f, y_f, wid, hei))




def get_tumor_blobs(im_tumor_bin_ori):
    """
    half_side = int(round(30. * skale))
    side = 2 * half_side + 1
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (side, side))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    im_closed_bin = cv2.morphologyEx(im_tumor_bin, cv2.MORPH_CLOSE, kernel)
    #   컨투어를 구한다.
    _, contours, _ = cv2.findContours(im_closed_bin, 1, cv2.CHAIN_APPROX_NONE);
    """
    im_tumor_bin = im_tumor_bin_ori.copy()
    #m1 = im_tumor_bin_ori.max()
    if np.uint8 != im_tumor_bin.dtype:
        im_tumor_bin = im_tumor_bin.astype(np.uint8)
    _, contours, _ = cv2.findContours(im_tumor_bin, 1, cv2.CHAIN_APPROX_NONE);
    #cv2.imwrite('im_tumor_bin.jpg', 255 * im_tumor_bin)
    #cv2.imwrite('im_closed_bin.jpg', 255 * im_closed_bin)
    #   컨투어를 리턴한다.
    #m2 = im_tumor_bin_ori.max()
    return contours


def compute_edge_integral(im_rgb, th_Canny_1 = 130, th_Canny_2 = 230):

    im_tissue_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    im_edge_255 = cv2.Canny(im_tissue_gray, th_Canny_1, th_Canny_2)
    _, im_edge_bin = cv2.threshold(im_edge_255, 1, 1, cv2.THRESH_BINARY)
    im_uint64_edge_integral = integral_image(im_edge_bin)
    return im_uint64_edge_integral, im_edge_bin


def resize_if_necessary(im, r_x, r_y, min_r):
    if r_x < min_r or r_y < min_r:
        ratio = min(r_x, r_y)
        im = cv2.resize(im, None, fx = ratio, fy = ratio)
    return im, im.shape[1], im.shape[0]


def compute_bounding_box_of_non_zero_pixels(im_bool):
    ar_y, ar_x = np.nonzero(im_bool)
    x_min = min(ar_x)
    x_max = max(ar_x)
    y_min = min(ar_y)
    y_max = max(ar_y)
    return (x_min, y_min), (x_max, y_max)