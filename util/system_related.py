# -*- coding: utf-8 -*-
#import PyTorchHelpers
import slackweb
import socket
import subprocess
import os.path
import os
import sys
import math
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists, basename
from subprocess import call
from PIL import Image
import collections




def round_i(x):
    return int(round(x))

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def slack_notify(message, channel="#botlog", username="ellie"):
    '''
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
    slack.notify(text="[%s] %s" % (socket.gethostname(), message), channel="#botlog", username="ellie",
                 icon_emoji=":ellie:")
    '''
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
    slack.notify(text="[%s] %s" % (socket.gethostname(), message), channel=channel, username=username,
                 icon_emoji=":ellie:")


def get_num_gpu():
    cmd = ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader']
    try:
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
        li_gpu_temperature = output.split()
        try:
            n_gpu = len(li_gpu_temperature)
        except ValueError:
            n_gpu = 0
    except:
        n_gpu = 0
    return n_gpu


def generate_tiled_tif(fn_img, shall_remove_non_tif, postfix):

    #fn_tif = marked_image_path.replace('.untiled.tif', '.mark.tif')
    if None == postfix:
        fn_tif = os.path.splitext(fn_img)[0] + '.tif'
    else:
        fn_tif = os.path.splitext(fn_img)[0] + '_' + postfix + '.tif'
    cmd = 'vips tiffsave "%s" "%s" --compression=jpeg --vips-progress --tile --pyramid --tile-width=240 --tile-height=240' % (fn_img, fn_tif)
    #print(cmd)
    call(cmd, shell=True)
    if shall_remove_non_tif:
        os.remove(fn_img)


def generate_tiled_tif_2(fn_img, fn_result, postfix):

    #file_name_only = get_exact_file_name_from_path(fn_img)
    #fn_tif = marked_image_path.replace('.untiled.tif', '.mark.tif')
    fn_tif = fn_result + '.' + postfix + '.tif'
    cmd = 'vips tiffsave "%s" "%s" --compression=jpeg --vips-progress --tile --pyramid --tile-width=240 --tile-height=240' % (fn_img, fn_tif)
    #print(cmd)
    call(cmd, shell=True)


def probe_model(model, batch_factor_per_gpu_given):

    # IMPORTANT: test model and fail back
    print('\nStart testing model and fail back')
    tile = np.array(Image.open('input.png'))
    tile = tile.astype(np.float32)
    tile = np.divide(tile, 255)
    tile = np.transpose(tile, [2, 0, 1])
    tile = tile[0:3]
    output = model.predict(tile)
    print(output.values())
    print('Finished testing model and fail back\n')
    batch_per_gpu_model = len(output)
    if 1 == batch_per_gpu_model:
        if batch_factor_per_gpu_given:
            batch_per_gpu = int(batch_factor_per_gpu_given * 20)
        else:
            print('You must provide the batch size per GPU\n')
            sys.exit(1)
    else:
        if batch_factor_per_gpu_given:
            batch_per_gpu = round_i(batch_factor_per_gpu_given * batch_per_gpu_model)
        else:
            batch_per_gpu = batch_per_gpu_model
    return batch_per_gpu, batch_per_gpu_model


def get_exact_file_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_file_path_if_string_exists_in_the_path_list(li_fn, strng):
    li_path = [fn for fn in li_fn if strng in get_exact_file_name_from_path(fn)]
    n_file = len(li_path)
    if len(li_path) > 1:
        print('There is more than one file whose name includes : ' + strng)
        sys.exit(1)
    elif 0 == n_file:
        print('There is NO such a file whose name includes : ' + strng)
        return None
    return li_path[0]

def is_this_file_name_in_the_list(li_fn, fn_query):
    li_path = [fn for fn in li_fn if fn_query == basename(fn)]
    n_same = len(li_path)
    return n_same > 0


def get_list_of_file_names_with_string_in_multi_directories(txt_li_dir):
    li_fn = []
    with open(txt_li_dir) as f:
        for line in f:
            line = line.strip()
            li_str = line.split()
            n_str = len(li_str)
            if 3 <= n_str:
                dir = li_str[0]
                ext = li_str[1]
                substring = li_str[2]
                li_fn_str = \
                    [join(dir, f) for f in listdir(dir) if (isfile(join(dir, f)) and f.endswith(ext) and substring in f and (not is_this_file_name_in_the_list(li_fn, f)))]
            elif 2 == n_str:
                dir = li_str[0]
                ext = li_str[1]
                li_fn_str = \
                    [join(dir, f) for f in listdir(dir) if (isfile(join(dir, f)) and f.endswith(ext) and (not is_this_file_name_in_the_list(li_fn, f)))]
            elif 1 == n_str:
                dir = li_str[0]
                li_fn_str = \
                    [join(dir, f) for f in listdir(dir) if (isfile(join(dir, f)) and (not is_this_file_name_in_the_list(li_fn, f)))]
            li_fn += li_fn_str
    return li_fn











def get_batch_size_as_multiple_of_num_gpu(len_batch, n_gpu):
    #batch_size = 0
    batch_per_gpu_actuall = 0
    if len_batch >= n_gpu:
        batch_per_gpu_actuall = int(math.floor(float(len_batch) / float(n_gpu)))
        #batch_size = batch_per_gpu_actuall * n_gpu
    return batch_per_gpu_actuall
