import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add,Concatenate,Conv2D, Input, Lambda, LeakyReLU, MaxPool2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.models import Model

from batch_norm import BatchNormalization
from utils import broadcast_iou

IOU_TRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

ANCHORS = np.array([(10, 13), (16, 30), (32, 23), (30, 61), (62,45), (59, 119), (116,90),(156, 198),(373, 326)], dtype=np.float32)/416.0

#print('anchors = ',ANCHORS)

MASKS = np.array([[6,7,8], [3,4,5],[0,1,2]],dtype=np.int8)
CLASSES = 80

def darkConv(x, filters, kernel_size, strides = 1, batch_norm = True) :
    if strides == 1 :
        padding = 'same'
    else :
        x = ZeroPadding2D(((1,0),(1,0)))(x) ##좌측 상단에서부터 패딩
        padding = 'valid'

    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding=padding, use_bias= not batch_norm, kernel_regularizer = l2(0.0005))(x)

    if batch_norm :
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x

def darkResidual(x, filters) :
    short_cut = x
    x = darkConv(x, filters//2, 1)
    x = darkConv(x, filters, 3)
    x = Add()([short_cut, x])

    return x

def darkBlock(x, filters, blocks) :
    x = darkConv(x, filters, 3, strides=2)
    for i in range(blocks) :
        x = darkResidual(x,filters)

    return x

def Darknet_Model(name=None) :
    x = inputs = Input([None, None, 3]) ## (Height, Width, Channels) => None으로 설정한 이유는 다른 해상도 이미지로 훈련할때 이렇게하면 코드 수정 안해도됨.
    x = darkConv(x, 32, 3) 
    x = darkBlock(x, 64, 1)
    x = darkBlock(x, 128, 2)
    x = x_36 = darkBlock(x,256,8)
    x = x_61 = darkBlock(x,512,8)
    x = darkBlock(x,1024,4)

    return Model(inputs, (x_36, x_61, x), name = name)

def yoloConv(filters, name = None) :
    def yolo_conv(x_input) :
        if isinstance(x_input, tuple) : ##  입력값의 자료형이 튜플이라면
            inputs = Input(x_input[0].shape[1:]), Input(x_input[1].shape[1:]) ## input  형태가 (1, dim, dim , channels)  이런식이라 이렇게 받야야함
            x, short_cut = inputs

            x = darkConv(x,filters,1) ## 1 by 1 conv
            x = UpSampling2D(2)(x) ## 업샘플링 레이어
            x = Concatenate()([x, short_cut]) ##  architecture picture  상의 * 가 이거
        else :
            x = inputs = Input(x_input.shape[1:])

        x = darkConv(x,filters,1)
        x = darkConv(x,filters*2,3)
        x = darkConv(x,filters,1)
        x = darkConv(x,filters*2,3)
        x = darkConv(x,filters,1)
    
        return Model(inputs, x, name=name)(x_input)

    return yolo_conv


def yoloOutput(filters, anchors, classes, name=None) :
    def yolo_out(x_input) :
        x = inputs = Input(x_input.shape[1:])
        x = darkConv(x,filters*2,3)
        x = darkConv(x, anchors * (classes + 5), 1, batch_norm = False) ## 피쳐 추출을 하기 위해 하는 최종 컨볼루션
        x = Lambda(lambda x : tf.reshape(x,(-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes+5)))(x)

        ## 마지막 컨볼루션 부분이 최종 아웃풋 형태를 만들어내는 부분이다. 
        ## feature map 의 한 각 픽셀을 그리드로 하여, 각 그리드의 셀마다 앵커박스 숫자 x (x,y,w,h +(오브젝트 존재여부) +p classes)
        ## 가 나오게 되며 이게 피쳐맵의 각 그리드마다 쭉 늘어선 형태이므로 3차원 형태의 텐서가 된다.
        ## 마지막 람다 부분은 그러한 3차원 형태를 
        ## [[grid_y][grid_x][anchor_idx][anchor_idx][x1,y1,w1,h1,PC(오브젝트가 있는가?),pclasses]] 형태로 바꾼것 뿐임.

        return Model(inputs, x, name=name)(x_input)

    return yolo_out

def yoloBox(predict, anchors, classes) : 
    ## predict = {batch_size, grid_size, grid_size, anchors, (x,y,w,h,P(obj),P(classes))}


    grid_size = tf.shape(predict)[1] ## (1,height,width) 어차피 정방형이므로 하나만 읽어와도됨
    box_xy, box_wh, is_obj_in, class_probs = tf.split(predict,(2,2,1,classes),axis=-1) ## 그리드 셀마다 나온 아웃풋 형태를 2자리 2자리 1자리 num(classes)로 잘라낸것
    ##box_xy => 바운딩 박스의 중앙점 좌표 (x, y)
    ##box_wh => 바운딩 박스의 너비,높이 (w, h)
    ##is_obj_in => 현재 셀에 오브젝트가 있는가?
    ##class_probs => 오브젝트가 있다면 현재 오브젝트에 대한 사전 훈련 네트워크의 확률값들
    box_xy = tf.sigmoid(box_xy)
    is_obj_in = tf.sigmoid(is_obj_in)
    class_probs = tf.sigmoid(class_probs)

    predict_box = tf.concat((box_xy,box_wh),axis=-1)

    grid = tf.meshgrid(tf.range(grid_size),tf.range(grid_size)) ## 그리드 생성
    grid = tf.expand_dims(tf.stack(grid,axis=-1),axis=2) # [grid_x, grid_y , 1  ,2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size,tf.float32) 
    box_wh = tf.exp(box_wh) * anchors

    box_xy1 = box_xy - box_wh / 2 ## 바운딩 박스 좌측 상단
    box_xy2 = box_xy + box_wh / 2 ## 바운딩 박스 우측 하단
    bounding_box = tf.concat([box_xy1, box_xy2],axis=-1)

    return bounding_box, is_obj_in, class_probs, predict_box

def NMS(outputs, anchors, masks, classes) :
    ## box , confidence, type
    b = []
    c = []
    t = []

    for output in outputs :
        b.append(tf.reshape(output[0], (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1])))
        c.append(tf.reshape(output[1], (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1])))
        t.append(tf.reshape(output[2], (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1])))


    b_box = tf.concat(b,axis=1)
    conf = tf.concat(c,axis=1)
    class_probs = tf.concat(t,axis=1)
    scores = conf * class_probs

    boxes, scores, classes, valid_detect = \
    tf.image.combined_non_max_suppression(
        boxes = tf.reshape(b_box,   (tf.shape( b_box)[0], -1, 1, 4)),
        scores = tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class = 100,
        max_total_size = 100,
        iou_threshold = IOU_TRESHOLD,
        score_threshold = SCORE_THRESHOLD
    )
    

    return boxes, scores, classes, valid_detect

def Yolo(size=None, channels=3, anchors = ANCHORS, masks=MASKS, classes=CLASSES, training=False) :
    x = inputs = Input([size,size,channels])

    x36, x61, x = Darknet_Model(name='yolo_darknet')(x)

    x = yoloConv(512,name='yolo_conv_0')(x)
    output0 = yoloOutput(512,len(masks[0]),classes,name='yolo_output_0') (x)


    x=yoloConv(256, name='yolo_conv_1')((x,x61))
    output1 = yoloOutput(256,len(masks[1]),classes,name='yolo_output_1') (x)

    x=yoloConv(128, name='yolo_conv_2')((x,x36))
    output2 = yoloOutput(128,len(masks[2]),classes,name='yolo_output_2') (x)

    if training == True :
        return Model(inputs,(output0,output1,output2), name='yolov3')

    boxes0 = Lambda(lambda x: yoloBox(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output0)
    boxes1 = Lambda(lambda x: yoloBox(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output1)
    boxes2 = Lambda(lambda x: yoloBox(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output2)

    outputs = Lambda(lambda x: NMS(x, anchors, masks, classes), name='yolo_nms')((boxes0[:3],boxes1[:3],boxes2[:3]))

    return Model(inputs,outputs, name='yolov3')

def yoloLoss(anchors, classes=CLASSES, ignore_threshold = 0.5) :
    def yolo_loss(gt, predict) :
        ##predict = (batch_size, grid_size, grid_size, anchors, (x,y,w,h),P(C), P(classes))

        ## 예측 바운딩박스 변형
        predict_box, predict_obj, predict_class, predict_xywh = yoloBox(predict, anchors, classes)

        p_xy = predict_xywh[...,0:2]
        p_wh = predict_xywh[...,2:4]

        ## ground truth 박스 변형

        gt_box, gt_obj, gt_class = tf.split(gt,(4,1,1),axis=-1)

        gt_xy = gt_box[...,0:2] + gt_box[...,2:4]/2
        gt_wh = gt_box[...,2:4] - gt_box[...,0:2]


        ## 작은 상자들에게 가중치를 줌

        box_loss_scale = 2 - gt_wh[...,0] * gt_wh[..., 1]

        # 예측 박스 방정식 뒤집기

        grid_size = tf.shape(gt)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid,axis=-1),axis=2)

        gt_xy = gt_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)

        gt_wh = tf.math.log(gt_wh/anchors)
        gt_wh = tf.where(tf.math.is_inf(gt_wh), tf.zeros_like(gt_wh),gt_wh)


        # masks 계산

        object_mask = tf.squeeze(gt_obj, -1)
        
        ## IOU가 threshold 보다 높으면 False Positive 무시

        gt_box_flat = tf.boolean_mask(gt_box, tf.cast(object_mask,tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(gt_box,gt_box_flat),axis = -1)
        ignore_mask = tf.cast(best_iou < ignore_threshold, tf.float32)

        ## loss 계산

        xy_loss = object_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_xy-p_xy),axis=-1)
        wh_loss = object_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_wh-p_wh),axis=-1)

        object_loss = binary_crossentropy(gt_obj,predict_obj)
        object_loss = object_mask * object_loss + (1 - object_mask) * ignore_mask * object_loss

        class_loss = object_mask * sparse_categorical_crossentropy(gt_class, predict_class)

        return xy_loss + wh_loss + object_loss + class_loss

    return yolo_loss


#model = Yolo()


