# MobileFaceNet_TF

Tensorflow implementation for MobileFaceNet.  
This version is different from the original repo [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF):
- Checkpoint and re-train continuously is more easier. Suitable for model training progress on Google Colaboratory.
- Bug fixes
- Convert to Tensorflow Lite format
- Examples on ipython notebooks  

---

## Pre-trained model
Pretrain is available at [arch/pretrained_model/MobileFaceNet_TFLite/](./arch/pretrained_model/MobileFaceNet_TFLite/)

---

## Dependencies

- tensorflow >= r1.5
- opencv-python 3.x
- python 3.x
- scipy
- sklearn
- numpy
- mxnet
- pickle


---

## Prepare dataset
1. choose one of the following links to download dataset which is provide by insightface. (Special Recommend MS1M-refine-v2)
* [MS1M-refine-v2@BaiduDrive](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ), [MS1M-refine-v2@GoogleDrive](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://www.dropbox.com/s/m9pm1it7vsw3gj0/faces_vgg2_112x112.zip?dl=0)
* [Insightface Dataset Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
2. move dataset to `${MobileFaceNet_TF_ROOT}/datasets`.
3. run `${MobileFaceNet_TF_ROOT}/utils/data_process.py`.

> Take a look at [data_prepair.ipynb](data_prepair.ipynb) for more details.

---

## Training

1. refined super parameters by yourself special project.
2. run script
`${MobileFaceNet_TF_ROOT}/train_nets.py`
3. have a snapshot result at `${MobileFaceNet_TF_ROOT}/output`.

> Take a look at [train.ipynb](train.ipynb) for more details.

---

## Convert to Tensorflow Lite format
1. Choose your best model checkpoint.
2. Freeze graph and get the `.pb` model file using [freeze_graph.py](freeze_graph.py)
```bash
python3 freeze_graph.py \
--pretrained_model ./arch/pretrained_model/MobileFaceNet_TFLite/checkpoints/895000_MobileFaceNet.ckpt   \
--output_file ./arch/pretrained_model/MobileFaceNet_TFLite/MobileFaceNet.pb
```
3. Convert to TFLite format  
For example:

```bash
tflite_convert \
--output_file ./arch/pretrained_model/MobileFaceNet_TFLite/MobileFaceNet.tflite  \
--graph_def_file ./arch/pretrained_model/MobileFaceNet_TFLite/MobileFaceNet.pb  \
--input_arrays "input" \
--input_shapes "1,112,112,3"  \
--output_arrays embeddings \
--output_format TFLITE
```
> You may need to choose the suitable inference batch size. The previous example use batchsize = 1.

> Take a look at [test.ipynb](test.ipynb) for more details.

---

## Performance

|  size  | LFW(%) | Val@1e-3(%) | inference@GPU: Qualcomm Adreno TM 509(ms) |
| ------ | ------ | ----------- | --------------------- |
|  4.9MB  |  99.4+ |    98.3+    |          30-         |

---

## References

1. [facenet](https://github.com/davidsandberg/facenet)
2. [InsightFace mxnet](https://github.com/deepinsight/insightface)
3. [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
4. [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
5. [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
6. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
