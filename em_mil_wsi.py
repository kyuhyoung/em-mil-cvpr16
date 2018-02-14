#-*-coding: cp949-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

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













