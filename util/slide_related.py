# -*- coding: utf-8 -*-

from .system_related import get_exact_file_name_from_path

import cv2
import numpy as np
from skimage.transform.integral import integral_image
from skimage.transform import resize
from os import listdir
from os.path import join, isfile, exists, splitext
from openslide import OpenSlide
from time import time
from PIL import Image
from .image_related import is_this_patch_below_threshold, get_opencv_contours_from_xml, \
    get_image_size_from_list_of_contour, get_sum_of_rectangle
from .system_related import round_i
from xml.etree.ElementTree import parse

#def compute_scale(level):
#    return 1.0 / pow(2, level)

def read_region_dummy(slaid, x_y_0, level, w_h_l):
    return slaid.read_region(x_y_0, level, w_h_l).convert('RGB')
    '''
    tile_0255_l = slaid.read_region(x_y_0, level, w_h_l)
    tile_0255_l = tile_0255_l.convert('RGB')
    w_h_inve = (int(t / 2) for t in w_h_l)
    tile_0255_inve = tile_0255_l.resize(w_h_inve, Image.BICUBIC)
    tile_0255_l.save('before.jpg')
    tile_0255_inve.save('after.jpg')
    #cv2.imwrite('before.jpg', cv2.cvtColor(tile_0255_l, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('after.jpg', cv2.cvtColor(tile_0255_inve, cv2.COLOR_BGR2RGB))
    '''


def get_scale(tu_downsample, i_level):
    return tu_downsample[i_level]

def convert_2_level_l(v_0, tu_downsample, i_level):
    #return v_0 / get_scale(tu_downsample, i_level)
    return convert_from_level_1st_to_level_2nd(v_0, tu_downsample, 0, i_level)

def convert_from_level_1st_to_level_2nd(v_1, tu_downsample, l_1, l_2):
    scale_1 = get_scale(tu_downsample, l_1)
    scale_2 = get_scale(tu_downsample, l_2)
    r_scale = scale_1 / scale_2
    return v_1 * r_scale

def get_ratio_of_resoultion_l_over_0(slide, level):
    return convert_2_level_l(1.0, slide.level_downsamples, level)


def convert_2_level_l_int(v_0, tu_downsample, i_level):
    v_l = convert_2_level_l(v_0, tu_downsample, i_level)
    return int(round(v_l))

def convert_from_level_1st_to_level_2nd_int(v_1, tu_downsample, l_1, l_2):
    return int(round(convert_from_level_1st_to_level_2nd(v_1, tu_downsample, l_1, l_2)))



def convert_x_y_2_level_l(x_y_0, tu_downsample, i_level):
    return convert_2_level_l(x_y_0[0], tu_downsample, i_level), convert_2_level_l(x_y_0[1], tu_downsample, i_level)
def convert_x_y_2_level_m_n(x_y_m, tu_downsample, l_m, l_n):
    return convert_from_level_1st_to_level_2nd(x_y_m[0], tu_downsample, l_m, l_n), convert_from_level_1st_to_level_2nd(x_y_m[1], tu_downsample, l_m, l_n)



def convert_x_y_2_level_l_int(x_y_0, tu_downsample, i_level):
    return convert_2_level_l_int(x_y_0[0], tu_downsample, i_level), convert_2_level_l_int(x_y_0[1], tu_downsample, i_level)
def convert_x_y_2_level_m_n_int(x_y_m, tu_downsample, l_m, l_n):
    return convert_from_level_1st_to_level_2nd_int(x_y_m[0], tu_downsample, l_m, l_n), convert_from_level_1st_to_level_2nd_int(x_y_m[1], tu_downsample, l_m, l_n)




def convert_x_y_w_h_2_level_l(x_y_w_h_0, tu_downsample, i_level):
    return convert_2_level_l(x_y_w_h_0[0], tu_downsample, i_level), convert_2_level_l(x_y_w_h_0[1], tu_downsample, i_level), \
           convert_2_level_l(x_y_w_h_0[2], tu_downsample, i_level), convert_2_level_l(x_y_w_h_0[3], tu_downsample, i_level)
def convert_x_y_w_h_2_level_m_n(x_y_w_h_m, tu_downsample, l_m, l_n):
    return convert_from_level_1st_to_level_2nd(x_y_w_h_m[0], tu_downsample, l_m, l_n),\
           convert_from_level_1st_to_level_2nd(x_y_w_h_m[1], tu_downsample, l_m, l_n),\
           convert_from_level_1st_to_level_2nd(x_y_w_h_m[2], tu_downsample, l_m, l_n), \
           convert_from_level_1st_to_level_2nd(x_y_w_h_m[3], tu_downsample, l_m, l_n)




def convert_x_y_w_h_2_level_l_int(x_y_w_h_0, tu_downsample, i_level):
    return convert_2_level_l_int(x_y_w_h_0[0], tu_downsample, i_level), convert_2_level_l_int(x_y_w_h_0[1], tu_downsample, i_level), \
           convert_2_level_l_int(x_y_w_h_0[2], tu_downsample, i_level), convert_2_level_l_int(x_y_w_h_0[3], tu_downsample, i_level)


def convert_x_y_w_h_2_x_y_x_y_level_l_int(x_y_w_h_0, tu_downsample, i_level):
    x_l, y_l, w_l, h_l = convert_x_y_w_h_2_level_l(x_y_w_h_0, tu_downsample, i_level)
    x_f = int(round(x_l))
    y_f = int(round(y_l))
    x_t = int(round(x_l + w_l))
    y_t = int(round(y_l + h_l))
    return x_f, y_f, x_t, y_t


def get_image_size_of_level_n_minimal(openslaid, level_to, size_from, level_from):
    #   get the image size from openslide
    w_to_os, h_to_os = openslaid.level_dimensions[level_to]#    h_0 = slide.level_dimensions[0][1]
    #   compute the image size by scaling the size of current level
    #   take the min of each size dimension and return it.
    level_dif = level_to - level_from
    w_to_scale, h_to_scale = convert_x_y_2_level_l_int(size_from, level_dif)
    w_to = min(w_to_os, w_to_scale)
    h_to = min(h_to_os, h_to_scale)
    return w_to, h_to


def get_subimage_mpp_0255(slaid, mpp_inve, x_y_inve, w_h_inve, is_debug):

    level, mpp_x_best, mpp_y_best, r_x_best, r_y_best = \
        get_nearest_mpp(slaid, mpp_inve, ratio_max=1.0)

    mpp_x_0, mpp_y_0, is_from_mpp_xy = get_mpp_slide(slaid)
    mpp_0 = 0.5 * (mpp_x_0 + mpp_y_0)

    #location = (x, y)
    #saiz = (w, h)
    mpp_l = 0.5 * (mpp_x_best + mpp_y_best)
    skale_0_over_inve = mpp_inve / mpp_0
    skale_l_over_inve = mpp_inve / mpp_l
    x_y_0 = tuple(round_i(t * skale_0_over_inve) for t in x_y_inve)
    w_h_l = tuple(round_i(t * skale_l_over_inve) for t in w_h_inve)
    #tile_0255_l = slaid.read_region(x_y_0, level, w_h_l)
    tile_0255_l = read_region_dummy(slaid, x_y_0, level, w_h_l)
    tile_0255_l.convert('RGB')
    tile_0255_inve = tile_0255_l.resize(w_h_inve, Image.BICUBIC)
    if is_debug:
        tile_0255_l.save('before.jpg')
        tile_0255_inve.save('after.jpg')
        #cv2.imwrite('before.jpg', cv2.cvtColor(tile_0255_l, cv2.COLOR_BGR2RGB))
        #cv2.imwrite('after.jpg', cv2.cvtColor(tile_0255_inve, cv2.COLOR_BGR2RGB))

        #t1 = tile_0255_l.thumbnail(w_h_inve)
    #tile_0255_l.thumbnail(w_h_inve)
    im_rgb = np.array(tile_0255_inve)
    if 3 < im_rgb.shape[2]:
        im_rgb = im_rgb[:, :, :3]
    return im_rgb


def get_subimage_level_0255(slaid, level, x_y_l, w_h_l):

    #location = (x, y)
    #saiz = (w, h)
    skale = slaid.level_downsamples[level]
    x_y_0 = (t * skale for t in x_y_l)
    tile_0255 = read_region_dummy(slaid, x_y_0, level, w_h_l) # slaid.read_region(x_y_0, level, w_h_l)
    im_rgb = np.array(tile_0255)
    if 3 < im_rgb.shape[2]:
        im_rgb = im_rgb[:, :, 0:3]
    return im_rgb


def get_subimage_0_0255(slide, x_y_0, w_h_0):
    return get_subimage_level_0255(slide, 0, x_y_0, w_h_0)



def get_subimage_01_for_cnn(slide, level, x_y_0, w_h_1):
    #location = (x, y)
    #saiz = (w, h)
    #tile = slide.read_region(x_y, level, w_h)
    tile = read_region_dummy(slide, x_y_0, level, w_h_1)
    #tile = tile.astype(np.float32)
    tile = np.divide(tile, 255.)
    tile = np.transpose(tile, [2, 0, 1])
    tile = tile[0:3]
    return tile


def get_subimage_0_01_for_cnn(slide, x_y, w_h):
    return get_subimage_01_for_cnn(slide, 0, x_y, w_h)


def draw_patches(im_rgb_l, li_pos_0, level, kolor, thick_0):
    #   for each patch
    thick_l = max(1, convert_2_level_l_int(thick_0, level))
    for pos_0 in li_pos_0:
        #   get x, y, w, h and scale them
        x_f, y_f, x_t, y_t = convert_x_y_w_h_2_x_y_x_y_level_l_int(pos_0, level)
        p_ul = (x_f, y_f)
        p_lr = (x_t, y_t)
        #   draw rectangle
        cv2.rectangle(im_rgb_l, p_ul, p_lr, kolor, thick_l)
    return im_rgb_l

def compute_integral_image_of_mask(fn_slide, dir_mask, level, postfix_ext_mask):
    im_01_mask_l, w_m_l, h_m_l = load_mask()
    if 1 != im_01_mask_l.max():
        print('Something wrong !! : 1 != im_01_mask_l.max()')
    #   integral image를 구한다.
    im_uint64_int_mask = integral_image(im_01_mask_l)
        #   blob의 area를 구한다.
    #area_blob = integrate(im_int, (0, 0), (h - 1, w - 1))
    return im_uint64_int_mask


def is_this_white_bg(im_int_gray_l, level, int_bg_per_pxl, x_from_0, y_from_0, wid_0, hei_0):
    #th_pxl_l = convert_2_level_l(th_edge_pxl_0, level)
    x_f_l, y_f_l, w_l, h_l = convert_x_y_w_h_2_level_l((x_from_0, y_from_0, wid_0, hei_0), level)
    area_tlle_l = w_l * h_l
    sum_int_tile = get_sum_of_rectangle(im_int_gray_l, x_f_l, y_f_l, w_l, h_l)
    int_per_pxl = sum_int_tile / float(area_tlle_l)
    return int_per_pxl > int_bg_per_pxl


def is_this_signpen(im_int_edge_l, level, th_edge_pxl_0, x_from_0, y_from_0, wid_0, hei_0):
    th_pxl_l = convert_2_level_l(th_edge_pxl_0, level)
    x_f_l, y_f_l, w_l, h_l = convert_x_y_w_h_2_level_l((x_from_0, y_from_0, wid_0, hei_0), level)
    return is_this_patch_below_threshold(im_int_edge_l, th_pxl_l, x_f_l, y_f_l, w_l, h_l)

def is_this_tissue(im_integral_edge_l, level, th_edge_pxl_0, x_from_0, y_from_0, wid_0, hei_0):
    return (not is_this_signpen(im_integral_edge_l, level, th_edge_pxl_0, x_from_0, y_from_0, wid_0, hei_0))

def is_this_patch_tissue(im_rgb, tissue_detection, im_edge_integral = None, integral_level = None):
    is_tissue = False
    if 'non-white' == tissue_detection:
        #is_tissue = im_rgb.mean() <= 240
        is_tissue = im_rgb.mean() <= 230
    elif 'near-magenta' == tissue_detection:
        # colors close to Magenta.
        # reference https://engineering.purdue.edu/~abe305/HTMLS/rgbspace.htm
        mean_red = im_rgb[..., 0].mean()
        mean_green = im_rgb[..., 1].mean()
        mean_blue = im_rgb[..., 2].mean()
        condition_red = mean_red > 180
        condition_green = mean_green < 230
        condition_blue = mean_blue > 180
        is_tissue = condition_red and condition_green and condition_blue
        '''
        #   degug mode
        print(condition_red, condition_green, condition_blue)
        im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
        txt_t = "%s %s %s"%(round_i(mean_red), round_i(mean_green), round_i(mean_blue))
        p_txt = (int(10), int(30))
        face_font = cv2.FONT_HERSHEY_SIMPLEX
        scale_font = 1
        color_font = (255, 255, 255)
        thick_font = round_i(scale_font * 1.5)
        cv2.putText(im_bgr, txt_t, p_txt, face_font, scale_font, color_font, thick_font)
        cv2.imwrite('im_bgr.jpg', im_bgr)
        if is_tissue:
            dummy = 0
        '''
    elif 'edge' == tissue_detection:
        dummy = 0
    return is_tissue



def get_slide_size(fn_slide, level):
    slide = get_slide(fn_slide)
    if slide is not None:
        w_level, h_level = slide.level_dimensions[level]
    else:
        w_level, h_level = 0, 0
    return w_level, h_level



def get_nearest_mpp(slide, mpp_requested, ratio_max = 1.0):

    mpp_x_0, mpp_y_0, is_from_mpp_xy = get_mpp_slide(slide)
    #   각 downsample마다
    #for downsample in slide.level_downsamples:
    max_r_xy = -1
    level_best = 0
    mpp_x_best = mpp_x_0
    mpp_y_best = mpp_y_0
    r_x_best = mpp_x_0 / mpp_requested
    r_y_best = mpp_y_0 / mpp_requested
    for level, downsample in enumerate(slide.level_downsamples):
        # 비율을 계산한다.
        mpp_x_l = mpp_x_0 * downsample
        mpp_y_l = mpp_y_0 * downsample
        # 비율을 리스트에 넣는다
        r_x = mpp_x_l / mpp_requested
        r_y = mpp_y_l / mpp_requested
        #   the default valude of ratio_max is 1.0
        if ratio_max < r_y or ratio_max < r_x:
        #if level and (th_ratio < r_y or th_ratio < r_x):
            break
        #r_xy.append(max(r_x, r_y))
        r_xy = max(r_x, r_y)
        if r_xy > max_r_xy:
            max_r_xy = r_xy
            level_best = level
            mpp_x_best = mpp_x_l
            mpp_y_best = mpp_y_l
            r_x_best = r_x
            r_y_best = r_y
    # 가장 직은 비율을 구한다.
    return level_best, mpp_x_best, mpp_y_best, r_x_best, r_y_best



#def read_slide(path_img, mpp_requested, max_width_height, th_ratio_resize, shall_save_as_jpg = False, fn_result = None):
def read_slide(path_img, level, max_width_height, shall_save_as_jpg = False, fn_result = None):

    slide = get_slide(path_img)
    if not slide:
        return None, None, None, None, None
    is_orig_jpg_saved = False
    fn_orig = None
    '''
    level, mpp_x_best, mpp_y_best, r_x_best, r_y_best = get_nearest_mpp(slide, mpp_requested)
    shall_resize = r_x_best < th_ratio_resize or r_y_best < th_ratio_resize
    '''
    w_level, h_level = slide.level_dimensions[level]
    t = time()
    print('Reading image of level {} from slide file {}'.format(level, path_img))
    print('width : ' + str(w_level) + '  height : ' + str(h_level))
    is_either_width_or_height_too_big = w_level >= max_width_height or h_level >= max_width_height
    if shall_save_as_jpg and is_either_width_or_height_too_big:
        print('Maximum image dimension supported by PIL is 65500 pixels')
        print('Either width or height is bigger than 65500 !!\n\n')
        return None, None, None, None, None
    try:
        #slideimg = slide.read_region((0, 0), level, (w_level, h_level))
        slideimg = read_region_dummy(slide, (0, 0), level, (w_level, h_level))
    except:
        print('Can NOT read the region from this slide with OpenSlide : ' + path_img)
        print('I do not know why. Just can not read !!')
        return None, None, None, None, None
    try:
        im_rgb = np.array(slideimg)
    except MemoryError as err:
        print(str(err))
        print('Can NOT create numpy array from slide : ' + path_img)
        print('The size is too big !!\n\n')
        return None, None, None, None, None
    print('im_rgb.shape :');    print(im_rgb.shape);
    #   이미지를 못받았으면
    if 0 == len(im_rgb.shape):
        if is_either_width_or_height_too_big:
            print('Maximum image dimension supported by PIL is 65500 pixels')
            print('Either width or height is bigger than 65500 !!\n\n')
            return None, None, None, None, None
        slideimg.save('temp.jpg')
        slideimg = Image.open('temp.jpg')
        im_rgb = np.array(slideimg)
        if 0 == len(im_rgb.shape):
            print('Can NOT create numpy array from slide : ' + path_img)
            print('The size is too big !!\n\n')
            #   이 슬라이드는 제낀다.
            return None, None, None, None, None
    if shall_save_as_jpg:
        fn_orig = fn_result + '.jpg'
        print('Saving image to : ' + fn_orig)
        slideimg.save(fn_orig)
        print('Saved image to : ' + fn_orig)
        is_orig_jpg_saved = True
    slide.close()
    if 3 < im_rgb.shape[2]:
        im_rgb = im_rgb[:, :, 0:3]
    '''
    if shall_resize:
        w_level = r_x_best * w_level
        h_level = r_y_best * h_level
        im_rgb = cv2.resize(im_rgb, (h_level, w_level, im_rgb.shape[-1]))
    '''
    print('Elapsed time: %.3f' % (time() - t))
    return im_rgb, w_level, h_level, is_orig_jpg_saved, fn_orig


#def read_image(path_img, mpp_requested, th_ratio_resize, shall_save_as_jpg = False, fn_result = None):
def read_image(path_img, level = 0, shall_save_as_jpg = False, fn_result = None):

    if not isfile(path_img):
        print('There is NO such a file : ' + path_img)
        return None, None, None, None, None, None
    #level = 2
    max_width_height = 65500
    is_orig_jpg_saved = False
    is_openslide_used = False
    extension = splitext(path_img)[-1].lower()
    #extension = os.path.splitext(SlideFileName)[-1][1:]
    if extension == ".svs" or extension == ".bif" or extension == ".tif":
        im_rgb, w_level, h_level, is_orig_jpg_saved, fn_orig = \
            read_slide(path_img, level, max_width_height, shall_save_as_jpg, fn_result)
        is_openslide_used = True
    # else
    else:
        # open with PIL
        print('Reading image file : ' + path_img)
        t = time()
        slideimg = Image.open(path_img)
        w_level, h_level = slideimg.size
        print('width : ' + str(w_level) + '  height : ' + str(h_level))
        try:
            im_rgb = np.array(slideimg)
        except MemoryError as err:
            print(str(err))
            print('Can NOT create numpy array from slide : ' + path_img)
            print('The size is too big !!\n\n')
            return None, None, None, None, None, None

        print('Elapsed time: %.3f' % (time() - t))
        fn_orig = path_img
        if 3 < im_rgb.shape[2]:
            im_rgb = im_rgb[:, :, 0:3]
    return im_rgb, w_level, h_level, is_openslide_used, is_orig_jpg_saved, fn_orig


def read_slide_by_id_and_level(li_slide_path, id, level):

    fn_slide = None
    im_rgb = None
    for slide_path in li_slide_path:
        fn_only = get_exact_file_name_from_path(slide_path)
        if id in fn_only:
            fn_slide = slide_path
            break
    if fn_slide:
        res = read_image(fn_slide, level)
        im_rgb = res[0]
    else:
        print('There is no such slide whose name includes : ' + id)
    return im_rgb


def load_mask(is_from_xml, fn_only, dir_mask, postfix_ext_mask, scale_or_level):
    #fn_only = get_exact_file_name_from_path(fn_slide)
    #if postfix_ext_mask.endswith('.xml'):
    if is_from_xml:
        skale = scale_or_level
        margin = (100, 100)
        fn_xml_0 = join(dir_mask, fn_only + '.xml')
        return load_mask_xml(fn_xml_0, margin, skale)
    else:
        level = scale_or_level
        return load_mask_image(fn_only, dir_mask, postfix_ext_mask, level)

def load_mask_image(fn_only, dir_mask, postfix_ext_mask, level = 0):
    fn_mask = join(dir_mask, fn_only.title() + postfix_ext_mask)
    im_rgb, w_level, h_level, _ , _ , _ = read_image(fn_mask, level)
    im_mask_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    res, im_mask_01_l = cv2.threshold(im_mask_gray, 1, 1, cv2.THRESH_BINARY)
    return im_mask_01_l, w_level, h_level


#def load_mask_xml(fn_xml_0, margin, level):
def load_mask_xml(fn_xml_0, margin, skale):
    #fn_xml_0 = join(dir_xml_0, fn_only + '.xml')
    '''
    li_li_point_l = make_list_of_contour_l_from_xml_0(fn_xml_0, level)
    contours_l = convert_list_of_contour_2_opencv_contours(li_li_point_l)
    '''
    #contours_l, li_li_point_l = get_opencv_contours_from_xml(fn_xml_0, level)
    contours_l, li_li_point_l = get_opencv_contours_from_xml(fn_xml_0, skale)
    w_mask_l, h_mask_l = get_image_size_from_list_of_contour(li_li_point_l, margin)
    im_01_mask_l = np.zeros((h_mask_l, w_mask_l), np.uint8)
    cv2.drawContours(im_01_mask_l, contours_l, -1, 1, -1)
    return im_01_mask_l, w_mask_l, h_mask_l


#def make_list_of_contour_l_from_xml_0(fn_xml, level):
def make_list_of_contour_l_from_xml_0(fn_xml, skale):
    #skale = get_ratio_of_resoultion_l_over_0(slide, level)
    #skale = compute_scale(level)
    li_li_point = []
    tree = parse(fn_xml)
    #note = tree.getroot()
    #i_0 = 0
    for parent in tree.getiterator():
        for i_1, child1 in enumerate(parent):
            #print(child1)
            for i_2, child2 in enumerate(child1):
                #print(child2)
                for i_3, child3 in enumerate(child2):
                    #print(child3)
                    li_point = []
                    for i_4, child4 in enumerate(child3):
                        #print(child4)
                        x_0 = float(child4.attrib['X'])
                        y_0 = float(child4.attrib['Y'])
                        x_l = x_0 * skale
                        y_l = y_0 * skale
                        li_point.append([x_l, y_l])
                    #print('i_4 : ' + str(i_4))
                    if len(li_point):
                        #print('len(li_point) : ' + str(len(li_point)))
                        li_li_point.append(li_point)
                #print('i_3 : ' + str(i_3))
                #if 0 == i_3:
                    #a = 0
            #print('i_2 : ' + str(i_2))
        #print('i_1 : ' + str(i_1))
        #i_0 += 1
    #print('i_0 : ' + str(i_0))

    return li_li_point




def is_tumor_in_slide_name(fn):
    if 'tumor' in fn.lower():
        return True
    return False

def is_this_in_tumor_mask_folder(fn_only, dir_mask, postfix_ext_mask):

    #fn_only = get_exact_file_name_from_path(fn_slide)
    fn_mask = fn_only + postfix_ext_mask
    filelist = [f for f in listdir(dir_mask) if isfile(join(dir_mask, f)) and f.endswith(postfix_ext_mask)]
    return fn_mask in filelist

def check_if_tumor_slide_2(fn_only, dir_mask, postfix_ext_mask):
    return is_tumor_in_slide_name(fn_only) or is_this_in_tumor_mask_folder(
        fn_only, dir_mask, postfix_ext_mask)


def make_rect_mask_from_mask_integral_above(im_uint64_int, th, size_tile, interval = 1):
    h, w = im_uint64_int.shape#    h_0 = slide.level_dimensions[0][1]
    im_0255 = np.zeros((h, w), np.uint8)
    w_tile, h_tile = size_tile
    #th_n_edge = th_r_edge * w_tile * h_tile
    for y in xrange(0, h - h_tile, interval * h_tile):
        print(str(y) + ' / ' + str(h))
        for x in xrange(0, w - w_tile, interval * w_tile):
            #is_tissue = is_this_tissue(im_uint64_int, 0, th, x, y, w_tile, h_tile)
            is_tissue = is_this_patch_above_threshold(im_uint64_int, th,
                                                      x, y, w_tile, h_tile)
            if is_tissue:
                p_ul = (x, y)
                p_lr = (x + w_tile, y + h_tile)
                cv2.rectangle(im_0255, p_ul, p_lr, 255, -1)
    return im_0255

def make_rect_mask_from_contour_mask_above(im_01, th, size_tile, interval = 1):
    im_uint64_int = integral_image(im_01)
    return make_rect_mask_from_mask_integral_above(im_uint64_int, th, size_tile, interval)


def make_tile_mask_edge_threshold(im_uint64_int, th_r_edge, size_tile, interval = 1):

    w_tile, h_tile = size_tile
    th_n_edge = th_r_edge * w_tile * h_tile
    return make_rect_mask_from_mask_integral_above(im_uint64_int, th_n_edge, size_tile, interval)

def get_slide(fn_slide):
    if not exists(fn_slide):
        print('This file does NOT exist. Check the path : ' + fn_slide)
        slide = None
        return slide
    try:
        slide = OpenSlide(fn_slide)
    except:
        print('Can NOT open this slide with OpenSlide : ' + fn_slide)
        print('I do not know why. Just can not read !!')
        slide = None
    return slide



def get_mpp_slide(slide):
    is_from_mpp_xy = False
    if 'openslide.mpp-x' in slide.properties:
        str_mpp_x = slide.properties['openslide.mpp-x']
        str_mpp_y = slide.properties['openslide.mpp-y']
        mpp_x = float(str_mpp_x)
        mpp_y = float(str_mpp_y)
        is_from_mpp_xy = True
    elif 'tiff.XResolution' in slide.properties:
        str_XResolution = slide.properties['tiff.XResolution']
        str_YResolution = slide.properties['tiff.YResolution']
        str_ResolutionUnit = slide.properties['tiff.ResolutionUnit']

        X_Resolution = float(str_XResolution)
        Y_Resolution = float(str_YResolution)
        if 'inch' == str_ResolutionUnit:
            bunja = 25400.0
        elif 'centimeter' == str_ResolutionUnit:
            bunja = 10000.0
        else:
            print('The unit of resolution is ' + str_ResolutionUnit)
            print('Please do something for this unit !!!')
        mpp_x = bunja / X_Resolution
        mpp_y = bunja / Y_Resolution
    return mpp_x, mpp_y, is_from_mpp_xy


def get_mpp(fn_slide):
    mpp_x = -1
    mpp_y = - 1
    is_from_mpp_xy = False
    slide = get_slide(fn_slide)
    if slide:
        mpp_x, mpp_y, is_from_mpp_xy = get_mpp_slide(slide)
    return mpp_x, mpp_y, is_from_mpp_xy