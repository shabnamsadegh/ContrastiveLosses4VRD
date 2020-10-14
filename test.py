"""By Shabnam
To get scene graph of one image
I ran
python test.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_X-101-64x4d-FPN_8_epochs_vg_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5.yaml --load_ckpt trained_models/vg_X-101-64x4d-FPN/model_step62722.pth --output_dir Outputs/vg_X-101-64x4d-FPN
"""



import sys
sys.path.append("./lib")
sys.path.append("./tools")
import pickle

import cv2
from core.test_rel import im_detect_rels
from core.test_engine_rel import initialize_model_from_cfg
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
import argparse
import torch
import os

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--do_val', dest='do_val', help='do evaluation', action='store_true')
    parser.add_argument(
        '--do_vis', dest='do_vis', help='visualize the last layer of conv_body', action='store_true')
    parser.add_argument(
        '--do_special', dest='do_special', help='visualize the last layer of conv_body', action='store_true')
    parser.add_argument(
        '--use_gt_boxes', dest='use_gt_boxes', help='use gt boxes for sgcls/prdcls', action='store_true')
    parser.add_argument(
        '--use_gt_labels', dest='use_gt_labels', help='use gt boxes for sgcls/prdcls', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    if args.dataset == "oi_rel":
        cfg.TEST.DATASETS = ('oi_rel_val',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_rel_mini":
        cfg.TEST.DATASETS = ('oi_rel_val_mini',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_all_rel_train":
        cfg.TEST.DATASETS = ('oi_all_rel_train',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_all_rel":
        cfg.TEST.DATASETS = ('oi_all_rel_val',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_kaggle":
        cfg.TEST.DATASETS = ('oi_kaggle_rel_test',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "vg_mini":
        cfg.TEST.DATASETS = ('vg_val_mini',)
        cfg.MODEL.NUM_CLASSES = 151
        cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background
    elif args.dataset == "vg":
        cfg.TEST.DATASETS = ('vg_val',)
        cfg.MODEL.NUM_CLASSES = 151
        cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background
    elif args.dataset == "vrd_train":
        cfg.TEST.DATASETS = ('vrd_train',)
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70  # exclude background
    elif args.dataset == "vrd":
        cfg.TEST.DATASETS = ('vrd_val',)
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70  # exclude background
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'

    assert_and_infer_cfg()

    if not cfg.MODEL.RUN_BASELINE:
        assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
            'Exactly one of --load_ckpt and --load_detectron should be specified.'

    # manually set args.cuda
    args.cuda = True

    if args.use_gt_boxes:
        if args.use_gt_labels:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.pkl')
        else:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.pkl')
    else:
        det_file = os.path.join(args.output_dir, 'rel_detections.pkl')

    box_proposals = None
    img_path = "Sakura.jpeg"

    model = initialize_model_from_cfg(args)

    im = cv2.imread(img_path)
    im_results = im_detect_rels(model, im,box_proposals=box_proposals, dataset_name = None)

    with open(img_path+".pkl", 'wb') as handle:
        pickle.dump(im_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
