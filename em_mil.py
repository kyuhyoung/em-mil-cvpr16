#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from openslide import OpenSlide

import os, csv, re, ast, pdb, time

from config import get_config

def check_tissue_map(im_tissue_int, x_t, y_t, w_t, h_t):

    return True


def get_label_of_slide(di_slide_cancer_or_not, id_slide):
    return di_slide_cancer_or_not[id_slide]


def get_exact_file_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def create_patch_classifier():
    model = None
    return model

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
    len_tile_pxl, stride_factor = 0, 0
    #return li_fn_slide, li_mpp, n_iter
    return li_mpp, n_iter, di_slide_cancer_or_not, len_tile_pxl, stride_factor


def make_tile_pos_of_heatmap(fn_slide, mpp, len_tile_pxl, stride_factor,
                             im_tissue_int, tissue_mpp):
    f_tissue_over_current = tissue_mpp / mpp
    li_xywh_l = []
    #   슬라이드를 연다
    slide = OpenSlide(fn_slide)
    #   주어진 mpp에 맞는 적절한 레벨을 찾는다.
    lebel = 0
    #   해당 레벨의 mpp를 구한다.
    #   해당 레벨에서의 tile의 사이즈를 구한다.
    w_l, h_l = len_tile_pxl, len_tile_pxl
    #   티슈 맵에서의 tile의 사이즈를 구한다.
    w_t, h_t = 0, 0
    #   해당 레벨에서의 stride_pxl을 구한다.
    stride_pxl_l = stride_factor * len_tile_pxl
    #   해당 레벨 이미지의 사이즈를 구한다.
    height_l, width_l = 0, 0
    y_l, y_t = 0
    #   height 동안
    while y_l + h_l < height_l:
        #   width 동안
        x_l, y_t = 0
        while x_l + w_l < width_l:
            #   tissue map에서 tissue에 해당하는지 본다.
            is_tissue = check_tissue_map(im_tissue_int, x_t, y_t, w_t, h_t)
            if not is_tissue:
                continue
            #   x, y, w, h 를 적는다.
            xywh_l = (x_l, y_l, w_l, h_l)
            li_xywh_l.append(xywh_l)
            #   x를 설정한다.
            x_l += stride_pxl_l
            x_t = x_l * f_tissue_over_current
        #   y를 설정한다.
        y_l += stride_pxl_l
        y_t = y_l * f_tissue_over_current
    return lebel, li_xywh_l


def make_slide_list(config):
    li_fn_slide = []
    return li_fn_slide

############################################################################################
def make_label_list(li_fn_slide, di_slide_cancer_or_not):
    n_slide = len(li_fn_slide)
    li_label = n_slide * [None]
    for iS, fn_slide in enumerate(li_fn_slide):
        # 아이디를 구한다.
        id_slide = get_exact_file_name_from_path(fn_slide)
        # 라벨을 구한다.
        label = get_label_of_slide(di_slide_cancer_or_not, id_slide)
        li_label[iS] = label
    return li_label

############################################################################################
def initialize(li_mpp, li_fn_slide, len_tile_pxl, stride_factor):
    n_res = len(li_mpp)
    n_slide = len(li_fn_slide)
    li_im_tissue_int, li_tissue_mpp = n_slide * [None], n_slide * [None]
    li_model, li_li_im_discriminative, li_im_count, li_li_tile_pos = \
        n_res * [None], [], [], []
    # 각 레졸루션에 대해
    for iR, mpp in enumerate(li_mpp):
        # 각 슬라이드에 대해서
        for iS, fn_slide in enumerate(li_fn_slide):
            # 아이디를 구한다.
            id_slide = get_exact_file_name_from_path(fn_slide)
            # 티슈의 패치 세트를 만든다.
            li_tile_pos = make_tile_pos_of_heatmap(
                fn_slide, mpp, len_tile_pxl, stride_factor,
                li_im_tissue_int[iS], li_tissue_mpp[iS])

            if iR:
                di_slide_res_tiles[iS].append(li_tile_pos)
            else:
                di_slide_res_tiles[iS] = [li_tile_pos]
            im_discriminative = np.zeros()
            im_count = np.zeros()
            # 티슈 패치 세트의 각 사각형에 대해서
            for tile_pos in li_tile_pos:
                # discriminive flag를 1로 둔다.
                li_li_im_discriminative[iS][iR], li_im_count[iS] = set_heat_map(im_discriminative, im_count, tile_pos, prob)
            li_li_tile_pos[iS][iR] = li_tile_pos
        # 학습기를 초기화한다.
        li_model[iR] = create_patch_classifier()
    return li_model, li_li_im_discriminative, li_im_count, li_li_tile_pos

############################################################################################
def normalize_heatmaps(li_fn_slide):
    li_im_discriminative = []
    # 각 슬라이드에 대해서
    for iS, fn_slide in enumerate(li_fn_slide):
        # discriminive flag map을 normalize한다.
        li_im_discriminative[iS] = normalized_heatmap(li_li_im_discriminative[iS], li_im_count[iS])
    return li_im_discriminative

############################################################################################
def update_model(li_model, n_iter, li_mpp, li_fn_slide):
    # 각 iteration에 대해
    for iI in range(n_iter):
        # 각 resolution에 대해
        for iR, mpp in enumerate(li_mpp):
            # 각 slide에 대해
            for iS, fn_slide in enumerate(li_fn_slide):
                label = li_label[iS]
                # 뽑을 패치 수를 정한다.
                n_patch = compute_num_patch_to_sample()
                # 뽑을 패치 위치를 정한다.
                li_patch_pos_l = compute_patch_pos_to_sample(n_patch, fn_slide, mpp, mpp_standard)
                slide = openslide(fn_slide)
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
    li_mpp, n_iter, di_slide_cancer_or_not, len_tile_pxl, stride_factor = \
        prepare_from_arguments(config)
    ############################################################################################
    li_label = \
        make_label_list(li_fn_slide, di_slide_cancer_or_not)
    ############################################################################################
    li_model, li_li_im_discriminative, li_im_count, li_li_tile_pos = \
        initialize(li_mpp, li_fn_slide, len_tile_pxl, stride_factor)
    ############################################################################################
    li_li_im_discriminative = \
        normalize_heatmaps(li_fn_slide)
    ############################################################################################
    li_model = \
        update_model(li_model, n_iter, li_mpp, li_fn_slide)
    ############################################################################################



if __name__=='__main__':

    main()

=======
# -*- coding: utf-8 -*-
em - mil
cvpr
16
mpp_standard = ?
############################################################################################
for iS, fn_slide enumerate( in li_fn_slide):
    아이디를
    구한다.
    id_slide = get_exact_name(fn_slide)
    라벨을
    구한다.
    label = get_label_of_slide(id_slide)
    li_label[iS] = label
############################################################################################
각
레졸루션에
대해
for iR, mpp in enumerate(li_mpp):
    각
    슬라이드에
    대해서
    for iS, fn_slide enumerate( in li_fn_slide):
        아이디를
        구한다.
        id_slide = get_exact_name(fn_slide)
        티슈의
        패치
        세트를
        만든다.
        li_tile_pos = make_tiles_of_heatmap(fn_slide, mpp, mpp_standard)
        if iR:
            di_slide_res_tiles[iS].append(li_tile_pos)
        else:
            di_slide_res_tiles[iS] = [li_tile_pos]
        im_discriminative = np.zeros()
        im_count = np.zeros()
        티슈
        패치
        세트의
        각
        사각형에
        대해서
        tile_pos in li_tile_pos:
        discriminive
        flag를
        1
        로
        둔다.
        li_li_im_discriminative[iS][iR], li_im_count[iS] = set_heat_map(im_discriminative, im_count, tile_pos, prob)
    li_li_tile_pos[iS][iR] = li_tile_pos
학습기를
초기화한다.
model[iR] = create_model()
############################################################################################
각
슬라이드에
대해서
for iS, fn_slide enumerate( in li_fn_slide):
    discriminive
    flag
    map을
    normalize한다.
    li_im_discriminative[iS] = normalized_heatmap(li_li_im_discriminative[iS], li_im_count[iS])
############################################################################################
각
iteration에
대해
for iI in range(n_iter):
    각
    resolution에
    대해
    for iR, mpp in enumerate(li_mpp):
        각
        slide에
        대해
        for iS, fn_slide enumerate( in li_fn_slide):
            label = li_label[iS]
            뽑을
            패치
            수를
            정한다.
            n_patch = compute_num_patch_to_sample()
            뽑을
            패치
            위치를
            정한다.
            li_patch_pos_l = compute_patch_pos_to_sample(n_patch, fn_slide, mpp, mpp_standard)
            slide = openslide(fn_slide)
            각
            패치에
            대해
            for pos_l in li_patch_pos_l:
                패치
                이미지를
                뜯는다.
                im_patch = get_sub_image(slide, pos_l, mpp, mpp_standard)
                패치 - 라벨을
                뽑는다.
                li_im_patch.append(im_patch)
                li_label.append(label)
        학습기를
        학습한다.
        model[iR] = train_model(model[iR], li_im_patch, li_label)
        각
        슬라이드에
        대해
        for iS, fn_slide enumerate( in li_fn_slide):
            slide = openslide(fn_slide)
            각
            티슈
            패치에
            대해
            평가를
            하여
            discriminive
            flag를
            매긴다.
            for tile_pos_l in li_li_tile_pos[iS][iR]:
                im_tile = get_sub_image(slide, tile_pos_l, mpp, mpp_standard)
                prob = test_model(model[iR], im_tile)
                li_li_im_discriminative[iS][iR], li_im_count[iS] = set_heat_map(im_discriminative, im_count, tile_pos_l,
                                                                                prob)
    각
    슬라이드에
    대해
    for iS, fn_slide enumerate( in li_fn_slide):
        discriminive
        flag
        map을
        normalize한다.
        li_im_discriminative[iS] = normalized_heatmap(li_li_im_discriminative[iS], li_im_count[iS])
############################################################################################
