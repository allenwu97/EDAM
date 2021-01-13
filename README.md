# EDAM-WSSS
The code and pretrain models of EDAM(Embedded Discriminative Attention Mechanism for Weakly Supervised Semantic Segmentation)

## Env
We train our model with Python 3.5, PyTorch 1.1.0 and 4 Tesla V100 GPUs with 16 GB memory. Other Python modules can be installed by running
 ```
conda install --yes --file requirements.txt
 ```
You can also download our env directly [[BaiduYun]](https://pan.baidu.com/s/18oMN2_1gAbmdFTaJBDEL5A) tlu7

After unzipping, running
 ```
source py3.5/bin/activate
 ```
 
## Dataset
  * [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/): You also need to specify the path ('voc12_root') of your downloaded dev kit.
  * [Saliency Maps](https://pan.baidu.com/s/1LfTwcm22Zup84yB635Ij1g) (fjst): Class-agnostic saliency maps generated by [PoolNet](https://github.com/backseason/PoolNet) as background cues.

## Pretrain Models
We provide three pre-trained models, including initialization parameters, EDAM parameters and Deeplabv2 parameters [[BaiduYun]](https://pan.baidu.com/s/1hbhBQKFAWtE69mucJXf6wQ) u7q7


## Train
Training EDAM from scratch.(Noting: We have pre-processed the dataset, and the amount of data in each epoch is equivalent to the original 25 epochs)
```
python3 train_EDAM_cls.py --lr 0.001 --batch_size 4 --max_epoches 1 --crop_size 368 --network network.resnet38_EDAM_cls --voc12_root [Root of VOC12] --weights [Root of initialization parameters] --wt_dec 5e-4 --session_name resnet38_EDAM_cls
```

To monitor loss and lr, run the following command in a separate terminal.
```
tensorboard --logdir runs
```


## Test
Generate pseudo labels.
 ```
python3 infer_cls.py --infer_list voc12/train_aug.txt --voc12_root /workdir/VOCdevkit/VOC2012 --network network.resnet38_EDAM_cls --weights [Path of EDAM Parameters]  --out_crf_pred [Output Path] --theta 0.2  --alpha 0.000001  --beta 0.99999 --sal_path [Path of Saliency Map]
 ```
 
Vis pseudo labels 
 ```
python3 colorful.py --img_path [Path of Pseudo Labels] —out_path [Path of Colorful Pseudo Labels]
 ```
 
## Segmentation Network
We use our pseudo labels fully-supervised train a [Deeplab-v2 Network](https://github.com/kazuto1011/deeplab-pytorch)  
We also provide the final pseudo labels for segmentation network training. [[BaiduYun]](https://pan.baidu.com/s/1ovEYet0JTiW9wj8UuP7-0g) 9aij

## Results
<table>
    <tr>
        <th>Model</th>
        <th>Train set</th>
        <th>Val set</th>
        <th>Crf?</th>
        <th>Saliency?</th>
        <th>Mean IoU</th>
    </tr>
    <tr>
        <td rowspan="3">EDAM</td>
        <td rowspan="3">
            <i>trainaug</i>
        </td>
        <td rowspan="3">
            <i>train</i>
        </td>
        <td>-</td>
        <td>-</td>
        <td>52.83</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>-</td>
        <td>58.61</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>&#10003;</td>
        <td><strong>68.11</td>
    </tr>
    <tr>
        <td rowspan="3">DeepLab-v2</td>
        <td rowspan="3">
            <i>trainaug</i>
        </td>
        <td rowspan="3">
            <i>val</i>
        </td>
        <td>-</td>
        <td>-</td>
        <td>69.66</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>-</td>
        <td><strong>70.96</td>
    </tr>
</table>
