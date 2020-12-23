import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import torch.nn.functional as F
import torch

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', theta=1.0, printlog=False, alpha=0, beta=1, crf=False, sal_path=None):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, input_type, theta, alpha, beta, crf, sal_path):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder, '%s.npy' % name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21, h, w), np.float32)
                for key in predict_dict.keys():
                    if crf:
                        tensor[key] = predict_dict[key]
                    else:
                        tensor[key + 1] = predict_dict[key]
                if sal_path is not None:
                    sal_file = os.path.join(sal_path, name + '.png')
                    sal_map = np.array(Image.open(sal_file))
                    sal_map = np.where(sal_map <= theta*100, 1, 0)
                    tensor = np.where(tensor < alpha, -2, tensor)
                    tensor = np.where(tensor > beta, 2, tensor)
                    tensor[0, :, :] = sal_map
                elif not crf:
                    tensor[0, :, :] = theta
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, input_type, theta, alpha, beta, crf, sal_path))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        if TP[i].value > 0:
            T_TP.append(T[i].value / (TP[i].value + 1e-10))
            P_TP.append(P[i].value / (TP[i].value + 1e-10))
        else:
            T_TP.append(0)
            P_TP.append(0)
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if T_TP[i] == 0:
                T_TP[i] = 1000000
            if P_TP[i] == 0:
                P_TP[i] = 1000000
            if i % 2 != 1:
                print('%11s:%7.3f%% %7.3f%% %7.3f%%' % (categories[i], IoU[i] * 100, 100/T_TP[i], 100/P_TP[i]), end='\t')
            else:
                print('%11s:%7.3f%% %7.3f%% %7.3f%%' % (categories[i], IoU[i] * 100, 100/T_TP[i], 100/P_TP[i]))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
        print('recall:', np.mean(np.array([100/T_TP[i] for i in range(1,20)])))
        print('predict:', np.mean(np.array([100/P_TP[i] for i in range(1,20)])))
    return loglist


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt', type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--theta', default=None, type=float)
    parser.add_argument('--alpha', default=None, type=float)
    parser.add_argument('--beta', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--crf', default=False, type=bool)
    parser.add_argument('--sal_path', default=None, type=str)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.theta is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, args.type, args.theta, True, args.alpha, args.beta, args.crf, args.sal_path)
        writelog(args.logfile, loglist, args.comment)
    else:
        l = []
        for i in range(40, 60, 2):
            theta = i / 100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, args.type, theta, False, args.alpha, args.beta, args.crf, args.sal_path)
            l.append(loglist['mIoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f%%' % (i, theta, loglist['mIoU']))
        writelog(args.logfile, {'mIoU': l}, args.comment)
