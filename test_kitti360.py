import os
import argparse
import torch
from test_MT_util import test_all_case
# from networks.EGNet import build_model
from networks.MTMT import build_model
# from networks.EGNet_onlyDSS import build_model
# from networks.EGNet_task3 import build_model

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/home/ext/chenzhihao/Datasets/ISTD_USR/test', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='/home/ext/chenzhihao/Datasets/UCF', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='/home/ext/chenzhihao/Datasets/SBU-shadow/SBU-Test_rename', help='Name of Experiment')
parser.add_argument('--root_path', type=str, default='/hdd/datasets/KITTI-360', help='Name of Experiment')
parser.add_argument('--seq', type=int, default=0)
parser.add_argument('--model', type=str,  default='EGNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--base_lr', type=float,  default=0.005, help='base learning rate')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
parser.add_argument('--epoch_name', type=str,  default='iter_7000.pth', help='choose one epoch/iter as pretained')
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--scale', type=int,  default=416, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--subitizing', type=float,  default=5.0, help='subitizing loss weight')
parser.add_argument('--repeat', type=int,  default=6, help='repeat')
FLAGS = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
# snapshot_path = os.path.join('/home/chenzhihao/shadow_detection/shadow-MT/model_SBU_EGNet', 'baseline_edgeX5_C64', FLAGS.epoch_name)
# test_save_path = os.path.join('/home/chenzhihao/shadow_detection/shadow-MT/model_SBU_EGNet', 'baseline_edgeX5_C64', 'prediction')
# snapshot_path = "../model_SBU_EGNet/baselineC64_DSS/"+str(FLAGS.edge)+str(FLAGS.base_lr)+'/'+FLAGS.epoch_name

# snapshot_path = "../model_SBU_EGNet/baselineC64_DSS_unlabelcons/repeat"+str(FLAGS.repeat)+'_edge'+str(FLAGS.edge)+'lr'+str(FLAGS.base_lr)+'consistency'+str(FLAGS.consistency)+'subitizing'+str(FLAGS.subitizing)+'/'+FLAGS.epoch_name
# snapshot_path = "../model_SBU_EGNet/baselineC64_DSS_unlabelcons/"+'edge'+str(FLAGS.edge)+'lr'+str(FLAGS.base_lr)+'consistency'+str(FLAGS.consistency)+'/'+FLAGS.epoch_name
# test_save_path = "../model_SBU_EGNet/baselineC64_DSS_unlabelcons/prediction_SBU_sub"
# test_save_path = "/home/chenzhihao/shadow_detection/EGNet/prediction"+FLAGS.model+"_post/"
# snapshot_path = "../model_SBU_EGNet_ablation/onlyDSS/"+FLAGS.epoch_name
# snapshot_path = "../model_SBU_EGNet_ablation/meanteacher/consistency"+str(FLAGS.consistency)+"/"+FLAGS.epoch_name # meanteacher
# snapshot_path = '../model_SBU_EGNet/baselineC64_DSS/10.00.005/iter_7000.pth' # multi-tasks
# test_save_path = '../model_SBU_EGNet_ablation/multi-task/prediction'
snapshot_path = 'iter_10000.pth'
# snapshot_path = "../model_ISTD_EGNet/salience/iter_3000.pth"

num_classes = 1

def test_calculate_metric(dir_image):
    dir_src = os.path.join(dir_image, 'data_rect')
    dir_tgt = os.path.join(dir_image, 'shadow')
    os.makedirs(dir_tgt, exist_ok=True)
    img_list = sorted([img for img in os.listdir(dir_src)])
    data_path = [(os.path.join(dir_src, img), os.path.join(dir_src, img)) for img in img_list]

    net = build_model('resnext101').cuda()
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=True, test_save_path=dir_tgt, trans_scale=FLAGS.scale)

    return avg_metric


if __name__ == '__main__':
    dir_image_00 = os.path.join(FLAGS.root_path, 'data_2d_raw/2013_05_28_drive_00{:0>2d}_sync'.format(FLAGS.seq), 'image_00')
    dir_image_01 = os.path.join(FLAGS.root_path, 'data_2d_raw/2013_05_28_drive_00{:0>2d}_sync'.format(FLAGS.seq), 'image_01')
    metric = test_calculate_metric(dir_image_00)
    print('[image_00] Test ber results: {}'.format(metric))
    metric = test_calculate_metric(dir_image_01)
    print('[image_01] Test ber results: {}'.format(metric))
