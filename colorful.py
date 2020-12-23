import numpy as np
from PIL import Image
import os
import argparse


def colorful(img_name, img_path, out_path):
    print(img_name)
    name = os.path.join(img_path, img_name + '.png')
    im = Image.open(name)
    out_path = os.path.join(out_path, img_name + '.png')

    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [128, 128, 0],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128]
                                 ], dtype='uint8').flatten()

    im.putpalette(palette)
    im.save(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default=None, type=str)
    parser.add_argument("--out_path", default=None, type=str)
    args = parser.parse_args()
    img_name_list = os.listdir(args.img_path)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    for img_name in img_name_list:
        if img_name[-4:] != '.png':
            continue
        img_name = img_name[:-4]
        colorful(img_name, args.img_path, args.out_path)