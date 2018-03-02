#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from openslide import OpenSlide
#from util.slide_related import get_mpp_slide, get_slide
import util.slide_related as slide_rel
import util.image_related as image_rel
import util.system_related as sys_rel
import torch, cv2, os, csv, re, ast, pdb, time
import numpy as np
import patch_classifier
from config import get_config
from torch.autograd import Variable
import torch.utils.data as utils_data
import torchvision.transforms as vis_trans
import torch.optim as optim
import torch.nn as nn
import logging
from multiprocessing import Pool, Manager
#from skimage.transform import resize


def read_region_tissue(params):
    """Read patches from given set of points in slide.
    Args:
        queue (queue Object): patch, x, y will be pushed
        patch_size (int) : patch size
        points (int tuple) : set of x, y points to be calculated.
        sub (str) : subject slide name to be classified.
    """
    queue, slaid, mpp_inve, w_h_inve, li_xy_inve, li_ij = params
    #queue, patch_size, points, sub = params
    #slide = OpenSlide(os.path.join(args.sub_path, sub))
    #slide_size = patch_size + (patch_size//2)*2
    #margin_size = patch_size//2

    try:
        for ii, xy_inve in enumerate(li_xy_inve):
            #patch = slaid.read_region((x-margin_size, y-margin_size),
            #                          0,
            #                          (slide_size, slide_size))
            patch = slide_rel.get_subimage_mpp_0255(slaid, mpp_inve, xy_inve, False)
            #patch = np.array(patch)[..., :3]

            patch = patch.astype(np.float32)
            patch = np.ascontiguousarray(patch, dtype=np.float32)
            if patch.shape != (w_h_inve[1], w_h_inve[0], 3):
                raise Exception('[!] Error : shape is somehow wrong')
            while True:
                if queue.full():
                    time.sleep(0.1)
                else:
                    #queue.put((patch, xy_inve[0], xy_inve[1]))
                    queue.put((patch,  xy_inve[0], xy_inve[1]))
                    break
        return True
    except Exception as e:
        return e






def make_zero_heatmap(w_h):
    wid, hei = w_h
    return np.zeros((hei, wid), np.float)


def update_heatmap(slaid, model, mpp_inve, li_ij_tissue, w_h_inve, num_ij, n_proc,
                   batch_size, use_gpu):
    # make zero heatmap
    im_heatmap_ij = make_zero_heatmap(num_ij)
    # set the model as test mode
    model.eval()
    # for each tissue position
    li_xy_inve_tissue = [tuple(map(lambda a, b: a * b, ij_tissue, w_h_inve)) for ij_tissue in li_ij_tissue]

    queue_size = batch_size * n_proc
    queue = Manager().Queue(queue_size)
    pool = Pool(n_proc)

    split_points = []
    for i in range(n_proc):
        split_points.append(li_xy_inve_tissue[i::n_proc])
    result = pool.map_async(read_region_tissue,
                            [(queue, slaid, mpp_inve, w_h_inve, li_xy_inve, transform)
                             for li_xy_inve in split_points])
    li_ij, li_patch_inve = [], []
    while True:
        if queue.empty():
            if not result.ready():
                time.sleep(0.5)
            elif result.ready() and 0 == len(li_patch_inve):
                break
        else:
            patch_inve, i, j = queue.get()
            li_ij.append((i, j))
            li_patch_inve.append(patch_inve)

        if len(li_patch_inve) == batch_size or \
                (result.ready() and queue.empty() and len(li_patch_inve) > 0):
            batch = Variable(torch.FloatTensor(np.stack(li_patch_inve)),
                             volatile=True)
            if use_gpu:
                batch = batch.cuda()
            start_time = time.time()
            output = model(batch)
            elapsed_batch = time.time() - start_time
            output = output.cpu().data.numpy()[:, 0]
            logging.debug(f'elapsed time for computing one batch is '
                          + f'{elapsed_batch:.3f}')
            n_img_in_this_batch = batch.size(0)
            for ii in range(n_img_in_this_batch):
                i, j = li_ij[ii]
                im_heatmap_ij[j, i] = output[ii]
            logging.debug(queue.qsize())
            li_ij, li_patch_inve = [], []

    if not result.successful():
        logging.debug('[!] Error: something wrong in result.')
    pool.close()
    pool.join()
    return im_heatmap_ij




def train_model(model, train_loader, optimizer, criterion, running_loss, n_img_total, n_eopch, use_gpu):
    for e in range(n_eopch):
        for i, data in enumerate(train_loader, 0):
    #def train_model(model, li_tu_im_patch_label, use_gpu):
        #n_patch = len(li_tu_im_patch_label)
        #for iP in range(n_patch, batch_size):
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            n_img_4_batch = labels.size()[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            # labels += 10
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # n_image_total += labels.size()[0]
            # print statistics
            loss_current = loss.data[0]
            running_loss += loss_current
            #n_image_total += n_img_per_batch
            n_img_total += n_img_4_batch
            print ('{} | {} : {}'.format(e, i, loss_current))
        dummy = 0

    return model, optimizer, running_loss, n_img_total








def compute_patch_pos_to_sample_inve(im_discri_ij, thres_descri, w_h_inve):
    li_ijm_discri_ij_bool = im_discri_ij > thres_descri
    li_ji = np.where(li_ijm_discri_ij_bool)

    #t1 = [t0 * w_h_inve for t0 in li_ij]
    t1 = [t0 * w_h_inve[(i + 1) % 2] for i, t0 in enumerate(li_ji)]
    #t1_ij = t1[::-1]
    t2 = zip(*t1[::-1])
    li_xy = list(t2)

    #li_xy = list(zip(*[t2 * w_h_inve for t2 in li_ij]))
    return li_xy

def set_heat_map(im_heatmap_ij, ij_inve, prob, num_ij):
    i, j = ij_inve
    if im_heatmap_ij is None:
        im_heatmap_ij = make_zero_heatmap(num_ij)
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


def make_dict_label_li_fn(di_fn_slide_cancer_or_not):
    di_label_li_fn = {}
    for fn_slide, cancer_or_not in di_fn_slide_cancer_or_not.items():
        if cancer_or_not in di_label_li_fn:
            di_label_li_fn[cancer_or_not].append(fn_slide)
        else:
            di_label_li_fn[cancer_or_not] = [fn_slide]
    di_label_s_label_i = {}
    i = 0
    for label_s, li_fn in di_label_li_fn.items():
        di_label_s_label_i[label_s] = i
        i += 1
    return di_label_li_fn, di_label_s_label_i



def prepare_from_arguments(config):
    #li_fn_slide = make_slide_list(config)
    li_mpp = config.mpp
    li_n_iter = config.iter
    li_n_epoch = config.epoch
    di_slide_cancer_or_not = get_slide_labels(config.fn_slide_label)
    di_cancer_or_not_slide, di_label_s_label_i = make_dict_label_li_fn(di_slide_cancer_or_not)
    len_tile_pxl = config.len_tile_pxl
    stride_factor = 1.0
    tissue_detection_method = config.tissue_detection
    use_gpu = config.use_gpu
    thres_descri = config.thres_desciriminative
    n_img_per_batch = config.batch_size
    n_worker = config.n_worker
    is_debug = config.is_debug is not 0
    trans = vis_trans.Compose([
        vis_trans.ToTensor()
        , vis_trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    li_transform = [trans, trans]
    #return li_fn_slide, li_mpp, n_iter
    return li_mpp, li_n_iter, li_n_epoch, di_slide_cancer_or_not, di_cancer_or_not_slide, di_label_s_label_i,\
           len_tile_pxl, stride_factor, tissue_detection_method, use_gpu, thres_descri, li_transform, \
           n_img_per_batch, n_worker, is_debug


# The smaller mpp, the better resolution
def make_tile_pos_of_heatmap(fn_slide, mpp_2_investigate, len_tile_pxl_inve, stride_factor,
                             mpp_0_slide, tissue_detection_method,
                             im_edge_integral, level_integral, is_debug):
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
            #im_rgb_patch_0 = slaid.read_region((x_0, y_0), 0, (w_0, h_0))
            im_rgb_patch_0 = slide_rel.read_region_dummy(slaid, (x_0, y_0), 0, (w_0, h_0))
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
                if is_debug:
                    id_slide = sys_rel.get_exact_file_name_from_path(fn_slide)
                    cv2.imwrite(
                        'im_patch_{}_{}_{}_{}_inve.jpg'.format(id_slide, mpp_2_investigate, i_inve, j_inve),
                        cv2.cvtColor(im_rgb_patch_0, cv2.COLOR_BGR2RGB)
                    )
                dummy = 0
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
               tissue_detection_method, use_gpu, is_debug):
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
                li_im_edge_integral[iS], li_level_edge[iS], is_debug)
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
def update_model(li_model, li_n_iter, li_mpp_inve, li_fn_slide, li_li_im_discri_ij, li_wh_inve,
                 li_label_s, li_n_epoch, thres_descri, di_label_s_label_i, li_transform,
                 n_img_per_batch, n_worker, use_gpu, is_debug):
    # 각 resolution에 대해
    n_resol = len(li_mpp_inve)

    li_optim, li_criterion, li_n_img_total, li_running_loss = \
        n_resol * [None], n_resol * [None], n_resol * [None], n_resol * [None]

    for iR, mpp_inve in enumerate(li_mpp_inve):
        li_optim[iR] = optim.SGD(li_model[iR].parameters(), lr=0.01, momentum=0.9)
        li_criterion[iR] = nn.CrossEntropyLoss()
        li_running_loss[iR] = 0
        li_n_img_total[iR] = 0
        if use_gpu:
            li_model[iR].cuda()
            li_criterion[iR].cuda()
        # 각 iteration에 대해
        for iI in range(li_n_iter[iR]):
            #li_tu_im_patch_label = []
            li_im_patch = []
            li_label = []
            # 각 slide에 대해
            for iS, fn_slide in enumerate(li_fn_slide):
                label_s = li_label_s[iS]
                label_i = di_label_s_label_i[label_s]
                # 뽑을 패치 수를 정한다.
                #n_patch = compute_num_patch_to_sample()
                # 뽑을 패치 위치를 정한다.
                li_patch_xy_inve = compute_patch_pos_to_sample_inve(
                    li_li_im_discri_ij[iS][iR], thres_descri, li_wh_inve[iR])
                n_patch = len(li_patch_xy_inve)
                slaid = OpenSlide(fn_slide)
                # 각 패치에 대해
                #li_tu_im_patch_label += \
                    #[(slide_rel.get_subimage_mpp_0255(slaid, mpp_inve, xy_inve, w_h_inve), label) for xy_inve in li_patch_xy_inve]
                li_im_patch += [slide_rel.get_subimage_mpp_0255(
                    slaid, mpp_inve, xy_inve, li_wh_inve[iR], is_debug)
                    for xy_inve in li_patch_xy_inve]
                li_label += n_patch * [label_i]
                if is_debug:
                    id_slide = sys_rel.get_exact_file_name_from_path(fn_slide)
                    i_from = len(li_im_patch) - n_patch
                    if mpp_inve > 1:
                        dummy = 0
                    [cv2.imwrite(
                        'im_patch_{}_{}_{}_{}_torch.jpg'.format(id_slide, mpp_inve,
                                                       int(li_patch_xy_inve[i][0] / li_wh_inve[iR]),
                                                       int(li_patch_xy_inve[i][1] / li_wh_inve[iR])),
                        cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
                    ) for i, im_rgb in enumerate(li_im_patch[i_from::])]
                    dummy = 0

            li_ts_img = [li_transform[iR](im_patch) for im_patch in li_im_patch]
            start = time.time()
            features = torch.stack(li_ts_img)
            end = time.time()
            lap_sec = end - start
            print ('sec. for torch.stack', lap_sec)
            start = time.time()
            targets = torch.LongTensor(li_label)
            end = time.time()
            lap_sec = end - start
            print ('sec. for torch.LongTensor', lap_sec)
            start = time.time()
            dset = utils_data.TensorDataset(features, targets)
            end = time.time()
            lap_sec = end - start
            print ('sec. for TensorDataset', lap_sec)
            dset_loader = utils_data.DataLoader(dset, batch_size=n_img_per_batch,
                                                shuffle=True, num_workers=n_worker)
            # 학습기를 학습한다.
            li_model[iR], li_optim[iR], li_running_loss[iR], li_n_img_total[iR] = \
                train_model(li_model[iR], dset_loader, li_optim[iR], li_criterion[iR],
                            li_running_loss[iR], li_n_img_total[iR], li_n_epoch[iR], use_gpu)


            for iS, slaid in enumerate(li_slide):
                li_li_im_discri_ij[iS][iR] = update_heatmap(
                    slaid, li_model[iR], mpp_inve, li_li_ij_tissue[iS][iR], li_wh_inve[iR],
                    num_ij, n_proc, batch_size, li_transform[iR], use_gpu)

            update_heatmap()
            # 각 슬라이드에 대해
            for iS, fn_slide in enumerate(li_fn_slide):
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
    #mpp_standard = 100

    #config, unparsed = get_config()
    config, li_fn_slide = get_config()
    li_mpp_inve, li_n_iter, li_n_epoch, di_slide_cancer_or_not, di_cancer_or_not_li_fn_slide, di_label_s_label_i, \
    len_tile_pxl_inve, stride_factor, tissue_detection_method, use_gpu, thres_descri, li_transform, \
    n_img_per_batch, n_worker, is_debug = \
        prepare_from_arguments(config)
    ############################################################################################
    li_label_s = \
        make_label_list(li_fn_slide, di_slide_cancer_or_not)
    ############################################################################################

    w_h_inve = (len_tile_pxl_inve, len_tile_pxl_inve)
    li_model, li_li_im_discri_ij, li_xywh_inve = \
        initialize(li_mpp_inve, li_fn_slide, len_tile_pxl_inve,
                   stride_factor, tissue_detection_method, use_gpu, is_debug)


    ############################################################################################
    #li_li_im_discriminative = normalize_heatmaps(li_fn_slide)
    ############################################################################################
    li_model = update_model(li_model, li_n_iter, li_mpp_inve, li_fn_slide, li_li_im_discri_ij, w_h_inve,
                            li_label_s, li_n_epoch, thres_descri, di_label_s_label_i, li_transform,
                            n_img_per_batch, n_worker, use_gpu, is_debug)


############################################################################################

##-d 1 -b 10 -w 2 -e 5 5 -m 0.5 4.0 -i 10 10 -T 0.5 -g 0 -t near-magenta -L 500 -l slide_cancer_or_not.csv /mnt/nfs/users/yscha/eval_68/06_S15-11514-11.svs /mnt/nfs/users/yscha/eval_68/06_S15-11514-2.svs
#-d 1 -b 10 -w 2 -e 5 5 -m 0.5 2.5 -i 10 10 -T 0.5 -g 0 -t near-magenta -L 500 -l slide_cancer_or_not.csv /mnt/nfs/users/yscha/eval_68/06_S15-11514-11.svs /mnt/nfs/users/yscha/eval_68/06_S15-11514-2.svs

if __name__=='__main__':

    '''
    aa, bb, cc = (1, 3), (9, 31), (20, 60)
    d = tuple(map(lambda a, b, c: a * b + c, aa, bb, cc))
    print(d)
    e = tuple(map(lambda a, b, c: a * 10 + b > c, aa, bb, cc))
    print(e)
    dummy = 0
    
    t1 = (np.array([2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7,
            7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10,
            10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12,
            12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
            14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,
            16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18,
            18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20,
            20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22,
            22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24,
            24, 24, 25, 25, 25]), np.array([3, 1, 2, 3, 4, 1, 2, 3, 5, 2, 3, 6, 9, 2, 3, 4, 3,
            4, 5, 10, 32, 3, 4, 5, 10, 31, 32, 3, 4, 5, 30, 31, 32, 3,
            4, 5, 7, 29, 30, 31, 32, 4, 5, 6, 28, 29, 30, 31, 4, 5, 6,
            28, 29, 30, 31, 4, 5, 6, 11, 27, 28, 29, 30, 4, 5, 6, 7, 26,
            27, 28, 29, 5, 6, 7, 8, 10, 26, 27, 28, 5, 6, 7, 8, 25, 26,
            27, 28, 7, 8, 9, 10, 24, 25, 26, 27, 8, 9, 10, 23, 24, 25, 26,
            27, 9, 10, 11, 22, 23, 24, 25, 26, 9, 10, 11, 12, 20, 21, 22, 23,
            24, 10, 11, 12, 13, 19, 20, 21, 22, 23, 11, 12, 13, 14, 15, 17, 18,
            19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 14, 15, 16, 17,
            18, 19, 15, 16, 17]))

    w_h_inve = (10, 100)
    t0 = [t2 * w_h_inve[i] for i,  t2 in enumerate(t1)]
    t444 = list(zip(*[t2 * 100 for t2 in t1]))
    print(t444)
    '''
    dummy = 0



    main()