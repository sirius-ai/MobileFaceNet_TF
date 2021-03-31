# Notification:

model of MobileFaceNet and L_Resnet_E_IR, all of them input shape is [batch_size, h, w, c] and data type is float32, input image must be RGB.
every pixel of input image should be subtracted 127.5, then divided 128.0.

## for MobileFaceNet

main API: inference

## for L_Resnet_E_IR

main API: get_resnet