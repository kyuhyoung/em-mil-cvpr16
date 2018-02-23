# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import pdb

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Data

# default=0.493
parser.add_argument('-g', '--use_gpu', type=int, #default='directory',
                    choices=[0, 1], help='whether use GPU or not')
parser.add_argument('-i', '--iter', type=int,
                    default=100, help='# of iterations')
parser.add_argument('-l', '--fn_slide_label', type=str, help='name of csv file where [slide id] - [slide level annotation] pair is listed.')
parser.add_argument('-L', '--len_tile_pxl', type=int, help='side length of tile')
parser.add_argument('-m', '--mpp', nargs=2, type=float,
                    default=[0.5, 4.0], help='micron per pixel of model')
parser.add_argument('-t', '--tissue_detection', type=str, #default='directory',
                    choices=['edge', 'near-magenta', 'non-white'], help='method of tissue detection')


'''
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_mode', type=str, default='directory',
                      choices=['directory', 'txt'],
                      help='Data mode such as [ directory, txt ]')
data_arg.add_argument('--data_path', type=str,
                      help='Data path depending on data mode.')
#data_arg.add_argument('--slide_dir',
#                      help='Whole Slide Image directory')
#data_arg.add_argument('--slide_list',
#                      help='Txt file : List of Whole Slide Image directory')

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--model_library', type=str, default='luatorch',
                       choices=['luatorch', 'pytorch'],
                       help='deep learning libaray')
model_arg.add_argument('--model_frame', type=str, 
                       default='lua-torch/pytorch_model.lua',
                       help='load path of model frame')
model_arg.add_argument('--model_state', type=str, 
                       default='/mnt/nfs/users/khkim/snapshot/final-preprocess/model.t7',
                       help='load path of model wieght state')
model_arg.add_argument('-S', '--patch_size', type=int,  # default=240,
                    help='patch size')
model_arg.add_argument('-N', '--neighbor_margin', type=int,  # default=239,
                    help='patch neighbor hood margin')

preproc_arg = add_argument_group('Prepocessing')
#preproc_arg.add_argument('--pre_process_mode', type=str, default='default',
#                      choices=['default'],
#                      help='pre process mode')
preproc_arg.add_argument('--color_norm_unit', type=str,
                      choices=['directory', 'slide', 'batch', 'patch'],
                      help='The unit of color normalization.')
preproc_arg.add_argument('--color_norm_method', type=str,
                      choices=['std_dev', 'whitening'],
                      help='The method of color normalization using mean and cov.')
preproc_arg.add_argument('-f', '--fn_sum_sum_sq', type=str,
                    help='file name of list of (file_id, n_pixel, sum, sum_squared).')
preproc_arg.add_argument('--tissue_detection', type=str,
                      choices=['non-white', 'near-magenta', 'edge', 'gray-thres'],
                      help='The method for tissue region extraction.')

# Process
proc_arg = add_argument_group('Processing')
proc_arg.add_argument('--nProc', type=int, default=8,
                       help='# of processes of multiprocessing Pool')
proc_arg.add_argument('--sliding', type=int, default=1,
                       help='sliding step : e.g. 1/4 sliding -> sliding = 4')
proc_arg.add_argument('--num_gpu', type=int, default=4,
                       help='# of GPU')
proc_arg.add_argument('--one_gpu_batch_size', type=int, #default=14,
                        help='one_gpu_batch_size')
proc_arg.add_argument('--fcn', type=int, default=1,
                        help='output_size of fcn')

# Result
result_arg = add_argument_group('Result')
result_arg.add_argument('--save', type=str, default='./result',
                        help='Directory path of result file for saving')
result_arg.add_argument('--threshold', type=float, default=0.01367,
                        help='f1 threshold for prediction')

# Debug image
debug_img_arg = add_argument_group('Debug_Image')
debug_img_arg.add_argument('--debug_image', default=False, action='store_true',
                            help='whether make debug image or not')
debug_img_arg.add_argument('--dest', default='./debug_image',
                            help='path to destination')
debug_img_arg.add_argument('--thresh', default='0.1')
debug_img_arg.add_argument('--annotation', default=None, dest='anno',
                            help='path to annotation directory')
debug_img_arg.add_argument('--all', default=False, action='store_true',
                            help='shows all patch')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--debugging', dest='debugging', action='store_true',
                      default=False)
'''

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

