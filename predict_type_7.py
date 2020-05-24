# coding:utf-8
import os
import sys
import json
import matplotlib as mpl
import numpy as np
import tensorflow as tf

mpl.use('Agg')

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops
from utils import label_map_util


type_name_list_cn = ['黄斑', '视盘', '激光治疗疤痕', '分界线', '膜嵴', '膜嵴伴血管扩张', '视网膜部分脱离', '视网膜完全脱离', '动脉纡曲', '静脉扩张', '出血']

MODEL_NAME = "./object_detection/rop/export_type_7"
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('data', 'rop_label_map.pbtxt_type_7')
PATH_TO_VAL_LIST = "./object_detection/dataset_tools/val_file_list_11_13.txt"
NUM_CLASSES = 7

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, ops, sess):
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def get_val_path_list():
    with open(PATH_TO_VAL_LIST, "r") as t:
        val_path_list = t.readlines()
        print(type(val_path_list))
        print(val_path_list[0])
        print(val_path_list[1])
    val_path_list = [x.replace("\n","") for x in val_path_list]
    return val_path_list


def pred():
    val_path_list = get_val_path_list()
    fovea_count = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    #optic_disc_count = [0,0,0,0]
    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            for index,json_path in enumerate(val_path_list):
                #print(json_path)
                image_path = json_path[:-4]+"jpeg"
                image_path = image_path.replace("png.jpeg","jpeg")
                print(image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, ops, sess)
                with open(json_path, "r", encoding="utf-8") as ex:
                    json_info = ex.readlines()
                    json_info = json_info[0]
                    json_info = json.loads(json_info, encoding="utf-8")
                    index_list = []
                    for key in json_info:
                        value_list = json_info.get(key)
                        if isinstance(value_list, list) and len(value_list) > 0:
                            for value in value_list:
                                jrect = value.get("jRect")
                                type_name_cn = value.get("type")
                                index_type = type_name_list_cn.index(type_name_cn)
                                if index_type < 10 and index_type > 1:
                                    if index_type == 9:
                                        index_type = 8
                                    index_list.append(index_type-1)
                #print(index_list)
                detection_scores = output_dict.get("detection_scores")
                detection_classes = output_dict.get("detection_classes")
                class_list = []
                for index_1,score in enumerate(detection_scores):
                    if score > 0.5:
                        class_list.append(detection_classes[index_1])
                #print(class_list)
                for tt_index in range(7):
                    tt = tt_index + 1
                    if tt in index_list and tt in class_list:
                        fovea_count[tt_index][0] += 1
                    if tt in index_list and tt not in class_list:
                        fovea_count[tt_index][1] += 1
                    if tt not in index_list and tt in class_list:
                        fovea_count[tt_index][2] += 1
                    if tt not in index_list and tt not in class_list:
                        fovea_count[tt_index][3] += 1
                #if 2 in index_list and 2 in class_list:
                #    optic_disc_count[0] += 1
                #if 2 in index_list and 2 not in class_list:
                #    optic_disc_count[1] += 1
                #if 2 not in index_list and 2 in class_list:
                #    optic_disc_count[2] += 1
                #if 2 not in index_list and 2 not in class_list:
                #    optic_disc_count[3] += 1
                #if index > 100 :
                #    break         
            print(fovea_count)
            #print(optic_disc_count)


if __name__ == '__main__':
    pred()
    pass
