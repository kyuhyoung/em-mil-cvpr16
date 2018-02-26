#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from openslide import OpenSlide
#from util.slide_related import get_mpp_slide, get_slide
import util.slide_related as slide_rel
import util.image_related as image_rel
import util.system_related as sys_rel
import torch, os, csv, re, ast, pdb, time
import numpy as np
import patch_classifier
from config import get_config
from skimage.transform import resize


def compute_patch_pos_to_sample(im_discri_ij, thres_descri, w_h_inve):
    li_ijm_discri_ij_bool = im_discri_ij > thres_descri
    li_ij = np.where(li_ijm_discri_ij_bool)
    li_xy = li_ij * w_h_inve
    return li_xy







def set_heat_map(im_heatmap_ij, ij_inve, prob, num_ij):
    i, j = ij_inve
    if im_heatmap_ij is None:
        num_i, num_j = num_ij
        im_heatmap_ij = np.zeros((num_j, num_i), np.float)
    im_heatmap_ij[j, i] = prob
    return im_heatmap_ij


def check_tissue_map(im_tissue_edge_integral, x_t, y_t, w_t, h_t):

    return True


def get_label_of_slide(di_slide_cancer_or_not, id_slide):
    return di_slide_cancer_or_not[id_slide]


#def get_exact_file_name_from_path(path):
#    return os.path.splitext(os.path.basename(path))[0]


# The toy example of 'fn_slide_label' : slide_cancer_or_not.csv
def get_slide_labels(fn_slide_label):
    di_slide_is_cancer = {}
    with open(fn_slide_label, 'r') as csv_slide_label:
        dr = csv.DictReader(csv_slide_label)
        #   for each slide
        for idx, row in enumerate(dr):
            slide_id = row['slide_id']
            is_cancer = eval(row['is_cancer']) is not 0
            di_slide_is_cancer[slide_id] = is_cancer
    return di_slide_is_cancer


def prepare_from_arguments(config):
    #li_fn_slide = make_slide_list(config)
    li_mpp = config.mpp
    n_iter = config.iter
    di_slide_cancer_or_not = get_slide_labels(config.fn_slide_label)
    len_tile_pxl = config.len_tile_pxl
    stride_factor = 1.0
    tissue_detection_method = config.tissue_detection
    use_gpu = config.use_gpu
    thres_descri = config.thres_desciriminative
    #return li_fn_slide, li_mpp, n_iter
    return li_mpp, n_iter, di_slide_cancer_or_not, len_tile_pxl, stride_factor, \
           tissue_detection_method, use_gpu, thres_descri

# The smaller mpp, the better resolution
def make_tile_pos_of_heatmap(fn_slide, mpp_2_investigate, len_tile_pxl_inve, stride_factor,
                             mpp_0_slide, tissue_detection_method,
                             im_edge_integral, level_integral):
    scale_from_level_0_2_level_inve = mpp_0_slide / mpp_2_investigate
    li_xywh_inve, li_ij_inve = [], []
    #   슬라이드를 연다
    slaid = OpenSlide(fn_slide)
    #   주어진 mpp에 맞는 적절한 레벨을 찾는다.
    #ret = slide_rel.get_nearest_mpp(slaid, mpp_2_investigate)
    #level_near, mpp_x_near, mpp_y_near = ret[0], ret[1], ret[2]
    #mpp_near = 0.5 * (mpp_x_near + mpp_y_near)
    width_0, height_0 = slaid.level_dimensions[0]
    #   해당 레벨의 mpp를 구한다.
    #   해당 레벨에서의 tile의 사이즈를 구한다.
    w_inve, h_inve = len_tile_pxl_inve, len_tile_pxl_inve
    #   티슈 맵에서의 tile의 사이즈를 구한다.
    w_0 = sys_rel.round_i(w_inve / scale_from_level_0_2_level_inve)
    h_0 = sys_rel.round_i(h_inve / scale_from_level_0_2_level_inve)
    #   해당 레벨에서의 stride_pxl을 구한다.
    stride_pxl_inve = stride_factor * len_tile_pxl_inve
    #   해당 레벨 이미지의 사이즈를 구한다.
    height_inve, width_inve = \
        height_0 * scale_from_level_0_2_level_inve, width_0 * scale_from_level_0_2_level_inve
    y_inve, j_inve = 0, 0
    #   height 동안
    while y_inve + h_inve < height_inve:
        #   width 동안
        y_0 = sys_rel.round_i(y_inve / scale_from_level_0_2_level_inve)
        x_inve, i_inve = 0, 0
        print(y_0)
        while x_inve + w_inve < width_inve:
            x_0 =  sys_rel.round_i(x_inve / scale_from_level_0_2_level_inve)
            #   tissue map에서 tissue에 해당하는지 본다.
            #is_tissue = check_tissue_map(im_edge_integral, x_t, y_t, w_t, h_t)
            if tissue_detection_method is not 'edge':
                im_edge_integral_patch = None
            im_rgb_patch_0 = slaid.read_region((x_0, y_0), 0, (w_0, h_0))
            im_rgb_patch_0 = np.array(im_rgb_patch_0)
            if 3 < im_rgb_patch_0.shape[2]:
                im_rgb_patch_0 = im_rgb_patch_0[:, :, 0:3]
            #im_rgb_patch_inve = resize(im_rgb_patch_0, (w_inve, h_inve))
            is_tissue = slide_rel.is_this_patch_tissue(
                im_rgb_patch_0, tissue_detection_method, im_edge_integral_patch, level_integral)

            if is_tissue:
                #   x, y, w, h 를 적는다.
                xywh_inve = (x_inve, y_inve, w_inve, h_inve)
                li_xywh_inve.append(xywh_inve)
                ij_inve = (i_inve, j_inve)
                li_ij_inve.append(ij_inve)
                #   x를 설정한다.
            x_inve += stride_pxl_inve
            i_inve += 1
        #   y를 설정한다.
        y_inve += stride_pxl_inve
        j_inve += 1
    ij_max = (i_inve, j_inve)
    return li_xywh_inve, li_ij_inve, ij_max


def make_slide_list(config):
    li_fn_slide = []
    return li_fn_slide

############################################################################################
def make_label_list(li_fn_slide, di_slide_cancer_or_not):
    n_slide = len(li_fn_slide)
    li_label = n_slide * [None]
    for iS, fn_slide in enumerate(li_fn_slide):
        # 아이디를 구한다.
        id_slide = sys_rel.get_exact_file_name_from_path(fn_slide)
        # 라벨을 구한다.
        label = get_label_of_slide(di_slide_cancer_or_not, id_slide)
        li_label[iS] = label
    return li_label

############################################################################################
def initialize(li_mpp_2_investigate, li_fn_slide, len_tile_pxl_inve, stride_factor,
               tissue_detection_method, use_gpu):
    n_res = len(li_mpp_2_investigate)
    n_slide = len(li_fn_slide)
    li_im_edge_integral, li_level_edge = n_slide * [None], n_slide * [None]
    li_mpp_0_slide = n_slide * [None]    # The actual mpp of test slides
    li_model = n_res * [None] \
    #li_im_count = []
    li_li_num_ij, li_li_xywh_inve, li_li_im_discri_ij = \
        n_slide * [None], n_slide * [None], n_slide * [None]
    for iS, fn_slide in enumerate(li_fn_slide):
        slaid = slide_rel.get_slide(fn_slide)
        mpp_x, mpp_y, is_from_mpp_xy = slide_rel.get_mpp_slide(slaid)
        li_mpp_0_slide[iS] = 0.5 * (mpp_x + mpp_y)
        li_li_im_discri_ij[iS] = n_res * [None]
        li_li_num_ij[iS] = n_res * [None]
        li_li_xywh_inve[iS] = n_res * [None]
        if tissue_detection_method is 'edge':
            ret = slide_rel.get_nearest_mpp(slaid, 1.0)
            li_level_edge[iS] = ret[0]
            im_rgb_4_edge = slide_rel.read_image(fn_slide, li_level_edge[iS])
            li_im_edge_integral[iS] = image_rel.compute_edge_integral(im_rgb_4_edge)
    # 각 레졸루션에 대해
    for iR, mpp_2_investigate in enumerate(li_mpp_2_investigate):
        w_h_inve = (len_tile_pxl_inve, len_tile_pxl_inve)
        li_model[iR] = patch_classifier.create_patch_classifier(w_h_inve, True, use_gpu)
        # 각 슬라이드에 대해서
        for iS, fn_slide in enumerate(li_fn_slide):
            # 아이디를 구한다.
            id_slide = sys_rel.get_exact_file_name_from_path(fn_slide)
            # 티슈의 패치 세트를 만든다.
            li_xywh_inve, li_ij_inve, num_ij = make_tile_pos_of_heatmap(
                fn_slide, mpp_2_investigate, len_tile_pxl_inve, stride_factor,
                li_mpp_0_slide[iS], tissue_detection_method,
                li_im_edge_integral[iS], li_level_edge[iS])
            if li_li_num_ij[iS][iR] is None:
                li_li_num_ij[iS][iR] = num_ij
            #if iR:
                #di_slide_res_tiles[iS].append(li_tile_pos)
            #else:
                #di_slide_res_tiles[iS] = [li_tile_pos]
            #im_discriminative = np.zeros()
            #im_count = np.zeros()
            # 티슈 패치 세트의 각 사각형에 대해서
            #for xywh_inve in li_xywh_inve:
            for ij_inve in li_ij_inve:
                # discriminive flag를 1로 둔다.
                li_li_im_discri_ij[iS][iR] = \
                    set_heat_map(li_li_im_discri_ij[iS][iR], ij_inve, 1.0, li_li_num_ij[iS][iR])
            li_li_xywh_inve[iR][iS] = li_xywh_inve
        # 학습기를 초기화한다.
    return li_model, li_li_im_discri_ij, li_xywh_inve#, li_im_count

############################################################################################
def normalize_heatmaps(li_fn_slide):
    li_im_discriminative = []
    # 각 슬라이드에 대해서
    for iS, fn_slide in enumerate(li_fn_slide):
        # discriminive flag map을 normalize한다.
        li_im_discriminative[iS] = normalized_heatmap(li_li_im_discriminative[iS], li_im_count[iS])
    return li_im_discriminative

############################################################################################
def update_model(li_model, n_iter, li_mpp_inve, li_fn_slide, li_li_im_discri_ij, w_h_inve,
                 li_label, thres_descri):
    # 각 iteration에 대해
    for iI in range(n_iter):
        # 각 resolution에 대해
        for iR, mpp_inve in enumerate(li_mpp_inve):
            # 각 slide에 대해
            for iS, fn_slide in enumerate(li_fn_slide):
                label = li_label[iS]
                # 뽑을 패치 수를 정한다.
                #n_patch = compute_num_patch_to_sample()
                # 뽑을 패치 위치를 정한다.
                li_patch_pos_l = compute_patch_pos_to_sample(
                    li_li_im_discri_ij[iS][iR], thres_descri, w_h_inve)
                slaid = OpenSlide(fn_slide)
                # 각 패치에 대해
                for pos_l in li_patch_pos_l:
                    # 패치 이미지를 뜯는다.
                    im_patch = get_sub_image(slide, pos_l, mpp, mpp_standard)
                    # 패치-라벨을 뽑는다.
                    li_im_patch.append(im_patch)
                    li_label.append(label)
            # 학습기를 학습한다.
            li_model[iR] = train_model(model[iR], li_im_patch, li_label)
            # 각 슬라이드에 대해
            for iS, fn_slide in enumerate( li_fn_slide):
                slide = openslide(fn_slide)
                # 각 티슈 패치에 대해 평가를 하여 discriminive flag를 매긴다.
                for tile_pos_l in li_li_tile_pos[iS][iR]:
                    im_tile = get_sub_image(slide, tile_pos_l, mpp, mpp_standard)
                    prob = test_model(model[iR], im_tile)
                    li_li_im_discriminative[iS][iR], li_im_count[iS] = set_heat_map(im_discriminative, im_count, tile_pos_l,
                                                                                    prob)
        # 각 슬라이드에 대해
        for iS, fn_slide in enumerate(li_fn_slide):
            # discriminive flag map을 normalize한다.
            li_im_discriminative[iS] = normalized_heatmap(li_li_im_discriminative[iS], li_im_count[iS])
    return li_model
############################################################################################


def main():

    # em-mil cvpr 16
    mpp_standard = 100

    #config, unparsed = get_config()
    config, li_fn_slide = get_config()
    li_mpp_inve, n_iter, di_slide_cancer_or_not, len_tile_pxl_inve, stride_factor, \
    tissue_detection_method, use_gpu, thres_descri = \
        prepare_from_arguments(config)
    ############################################################################################
    li_label = \
        make_label_list(li_fn_slide, di_slide_cancer_or_not)
    ############################################################################################
    li_model, li_li_im_discri_ij, li_xywh_inve = \
        initialize(li_mpp_inve, li_fn_slide, len_tile_pxl_inve,
                   stride_factor, tissue_detection_method, use_gpu)


    ############################################################################################
    #li_li_im_discriminative = normalize_heatmaps(li_fn_slide)
    ############################################################################################
    li_model = \
        update_model(li_model, n_iter, li_mpp_inve, li_fn_slide, li_li_im_discri_ij,
                     li_xywh_inve, li_label, thres_descri)
    ############################################################################################


# -T 0.5 -g 0 -t near-magenta -L 500 -l slide_cancer_or_not.csv /mnt/nfs/users/yscha/eval_68/06_S15-11514-11.svs /mnt/nfs/users/yscha/eval_68/06_S15-11514-2.svs
if __name__=='__main__':

    main()