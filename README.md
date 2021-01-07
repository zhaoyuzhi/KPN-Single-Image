# KPN-Single-Image

A PyTorch implementation of kernel prediction network for single image denoising.

## 1 Samples

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch50_gt.png)
![Represent](./img/train_epoch50_in.png)
![Represent](./img/train_epoch50_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch51_gt.png)
![Represent](./img/train_epoch51_in.png)
![Represent](./img/train_epoch51_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch52_gt.png)
![Represent](./img/train_epoch52_in.png)
![Represent](./img/train_epoch52_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch53_gt.png)
![Represent](./img/train_epoch53_in.png)
![Represent](./img/train_epoch53_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch54_gt.png)
![Represent](./img/train_epoch54_in.png)
![Represent](./img/train_epoch54_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch55_gt.png)
![Represent](./img/train_epoch55_in.png)
![Represent](./img/train_epoch55_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch56_gt.png)
![Represent](./img/train_epoch56_in.png)
![Represent](./img/train_epoch56_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch57_gt.png)
![Represent](./img/train_epoch57_in.png)
![Represent](./img/train_epoch57_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch58_gt.png)
![Represent](./img/train_epoch58_in.png)
![Represent](./img/train_epoch58_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch59_gt.png)
![Represent](./img/train_epoch59_in.png)
![Represent](./img/train_epoch59_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch60_gt.png)
![Represent](./img/train_epoch60_in.png)
![Represent](./img/train_epoch60_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch61_gt.png)
![Represent](./img/train_epoch61_in.png)
![Represent](./img/train_epoch61_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch62_gt.png)
![Represent](./img/train_epoch62_in.png)
![Represent](./img/train_epoch62_pred.png)

GT | Input | Denoised by KPN-Single-Image

![Represent](./img/train_epoch63_gt.png)
![Represent](./img/train_epoch63_in.png)
![Represent](./img/train_epoch63_pred.png)

## 2 Training

Trained models are available via this [OneDrive link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EuR2U0LqQyxDtgK06ObvK8gBsvKk3ez0miHSjWMIfkqcpQ?e=hR6sXc)

If you want to train your own data, change arg `baseroot` to your own data path, then run:
```bash
sh run.sh
```

## 3 Validation

We only provide one kind of model for specific noise level. If you want to test your own data, change the arg `baseroot` to the path to your validation set, `save_name` to saving path, and `load_name` to trained model path.
```bash
python validation.py
```

## 4 Acknowledgement

This KPN code is borrowed from the [project](https://github.com/z-bingo/kernel-prediction-networks-PyTorch).

```bash
@inproceedings{mildenhall2018burst,
  title={Burst denoising with kernel prediction networks},
  author={Mildenhall, Ben and Barron, Jonathan T and Chen, Jiawen and Sharlet, Dillon and Ng, Ren and Carroll, Robert},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2502--2510},
  year={2018}
}
```
