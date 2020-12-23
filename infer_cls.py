
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
import imageio
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet_EDAM_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--voc12_root", required=True, type=str)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_cam_pred", default=None, type=str)
    parser.add_argument("--out_crf",default=None,type=str)
    parser.add_argument("--out_crf_pred", default=None, type=str)
    parser.add_argument("--theta", default=0.5, type=float)
    parser.add_argument("--alpha", default=0, type=float)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--sal_path", default=None, type=str)


    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()
    print(model.load_state_dict(torch.load(args.weights)))

    model.eval()
    model.cuda()


    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=(1, 0.5, 1.5, 2.0),
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    tbar = tqdm(infer_data_loader)
    for iter, (img_name, img_list, label) in enumerate(tbar):
        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, input):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    img, label = input
                    cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam
        input = [(img_list[i], label) for i in range(8)]
        thread_pool = pyutils.BatchThreader(_work, list(enumerate(input)),
                                            batch_size=8, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()
        sum_cam = np.sum(cam_list, axis=0)

        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True))

        cam_dict = {}

        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            if not os.path.exists(args.out_cam):
                os.mkdir(args.out_cam)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            if not os.path.exists(args.out_cam_pred):
                os.mkdir(args.out_cam_pred)
            h, w = list(cam_dict.values())[0].shape
            tensor = np.zeros((21, h, w), np.float32)
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]
            if args.sal_path is not None:
                sal_file = os.path.join(args.sal_path, img_name + '.png')
                sal_map = np.array(Image.open(sal_file))
                sal_map = np.where(sal_map <= args.theta * 100, 1, 0)
                tensor = np.where(tensor < args.alpha, -2, tensor)
                tensor = np.where(tensor > args.beta, 2, tensor)
                tensor[0, :, :] = sal_map
            else:
                tensor[0, :, :] = args.theta
            pred = np.argmax(tensor, axis=0)
            imageio.imwrite(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, t=3, labels=bgcam_score.shape[0])
            n_crf_al = dict()
            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key + 1] = crf_score[i + 1]
            return n_crf_al

        if args.out_crf is not None:
            if not os.path.exists(args.out_crf):
                os.mkdir(args.out_crf)
            alpha = 1
            crf = _crf_with_alpha(cam_dict, alpha)
            np.save(os.path.join(args.out_crf, img_name + '.npy'), crf)

        if args.out_crf_pred is not None:
            if not os.path.exists(args.out_crf_pred):
                os.mkdir(args.out_crf_pred)
            alpha = 1
            crf_dict = _crf_with_alpha(cam_dict, alpha)
            crf = np.zeros((21, orig_img_size[0], orig_img_size[1]), np.float32)
            for key in crf_dict.keys():
                crf[key] = crf_dict[key]
            if args.sal_path is not None:
                sal_file = os.path.join(args.sal_path, img_name + '.png')
                sal_map = np.array(Image.open(sal_file))
                sal_map = np.where(sal_map <= args.theta * 100, 1, 0)
                tensor = np.where(tensor < args.alpha, -2, tensor)
                tensor = np.where(tensor > args.beta, 2, tensor)
                tensor[0, :, :] = sal_map
            pred = np.argmax(crf, 0)
            imageio.imwrite(os.path.join(args.out_crf_pred, img_name + '.png'), pred.astype(np.uint8))


