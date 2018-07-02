### MobileFaceNet_TF

Tensorflow implementation for MobileFaceNet.

## dependencies

- tensorflow >= r1.5
- opencv-python 3.x
- python 3.x
- scipy
- sklearn
- numpy
- mxnet
- pickle

## Prepare dataset

1. choose one of The following links to download dataset which is provide by insightface. (Special Recommend MS1M)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://drive.google.com/open?id=1KORwx_DWyIScAjD6vbo4CSRu048APoum)
2. move dataset to ${MobileFaceNet_TF_ROOT}/datasets.
3. run ${MobileFaceNet_TF_ROOT}/utils/data_process.py.

## training

1. refined super parameters by yourself special project.
2. run script
'''${MobileFaceNet_TF_ROOT}/train_nets.py'''
3. have a snapshot result at ${MobileFaceNet_TF_ROOT}/output.

## performance

|  size  | LFW(%) | Val@1e-3(%) | inference@MSM8976(ms) |
| ------ | ------ | ----------- | --------------------- |
|  5.7M  | 99.25+ |    96.8+    |          260-         |

## References

1. [facenet](https://github.com/davidsandberg/facenet)
2. [InsightFace mxnet](https://github.com/deepinsight/insightface)
3. [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
4. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)