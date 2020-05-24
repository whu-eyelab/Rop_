# coding:utf-8
import tensorflow as tf
import random
from object_detection.utils import dataset_util
from PIL import Image
import json
import io

flags = tf.app.flags
flags.DEFINE_string('output_path_train', './object_detection/rop_sample/train_type_7.tf',
                    'Path to output train TFRecord')
flags.DEFINE_string('output_path_val', './object_detection/rop_sample/val_type_7.tf',
                    'Path to output val TFRecord')
FLAGS = flags.FLAGS

key_list = ['stage', 'file', 'additionalPathological', 'pathologicalParts', 'manualAnnotations']
type_name_list_cn = ['激光治疗疤痕', '分界线', '膜嵴', '膜嵴伴血管扩张', '视网膜部分脱离', '视网膜完全脱离', '动脉纡曲', '静脉扩张' ]

type_name_list_en = [ "laser photocogulation scar", "demarcation line", "ridge",
                     "ridge with extra retinal fibrovascular",
                     "subtotal retinal detachment", "total retinal detachment", "arterial tortuosity",
                     "venous dilation"]
type_name_count = [0,0,0,0,0,0,0,0]

def create_tf_example(example):
    ''' example json 文件的路径 保证同名png在同一路径下 '''
    png_path = example.replace(".json", ".png")
    png_path = png_path.replace(".png.png", ".png")
    jpeg_path = example.replace(".json", ".jpeg")
    jpeg_path = jpeg_path.replace(".png.jpeg", ".jpeg")
    img = Image.open(png_path)
    r, g, b, a = img.split()
    img = Image.merge("RGB", (r, g, b))
    img.save(jpeg_path)
    with open(example, "r", encoding="utf-8") as ex:
        json_info = ex.readlines()
        json_info = json_info[0]
        json_info = json.loads(json_info, encoding="utf-8")
    with tf.gfile.GFile(jpeg_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    height = image.height  # Image height
    width = image.width  # Image width
    filename = jpeg_path  # Filename of the image. Empty if image is not from file
    filename = filename.encode("utf-8")
    print(filename)
    encoded_image_data = encoded_jpg
    image_format = b'jpeg'
    #
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for key in json_info:
        value_list = json_info.get(key)
        if isinstance(value_list, list) and len(value_list) > 0:
            for value in value_list:
                jrect = value.get("jRect")
                type_name_cn = value.get("type")
                try:
                    index = type_name_list_cn.index(type_name_cn)
                except:
                    continue
                if index < 0:
                    continue
                if index == 7:
                    index = 6
                type_name_en = type_name_list_en[index]
                type_name_count[index] +=1
                #print(type_name_en)
                #print(type_name_cn)
                xmin = jrect.get("x")
                ymin = jrect.get("y")
                xmax = xmin + jrect.get("width")
                ymax = ymin + jrect.get("height")
                xmins.append(xmin * 1.0 / width)
                xmaxs.append(xmax * 1.0 / width)
                ymins.append(ymin * 1.0 / height)
                ymaxs.append(ymax * 1.0 / height)
                classes_text.append(type_name_en.encode("utf-8"))
                classes.append(index + 1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
    pass


def divide_dataset(total_path):
    train_data_list = []
    val_data_list = []
    index_list = [5,4,3,2,0,6,7,1]
    for index__ in index_list:
        temple_list = []
        for sample_path in total_path:
            if sample_path in train_data_list:
                continue
            if sample_path in val_data_list:
                continue
            else:
                with open(sample_path, "r", encoding="utf-8") as ex:
                    json_info = ex.readlines()
                    json_info = json_info[0]
                    json_info = json.loads(json_info, encoding="utf-8")
                    for key in json_info:
                        value_list = json_info.get(key)
                        if isinstance(value_list, list) and len(value_list) > 0:
                            for value in value_list:
                                jrect = value.get("jRect")
                                type_name_cn = value.get("type")
                                try:
                                    index = type_name_list_cn.index(type_name_cn)
                                except:
                                    continue
                                if index == index__:
                                    temple_list.append(sample_path)
        temple_list = list(set(temple_list))
        temple_list = temple_list[:1000]
        length = len(temple_list)
        print(length,index__, type_name_list_cn[index__])
        train_path = random.sample(temple_list,int(length*0.8))
        val_path = []
        for sample_path in temple_list:
            if not sample_path in train_path:
                val_path.append(sample_path)
        print("train num: %s" % len(train_path))
        print("val num: %s" % len(val_path))
        train_data_list.extend(train_path)
        val_data_list.extend(val_path)

    print("total train num: %s" % len(train_data_list))
    print("total val num: %s" % len(val_data_list))
    return train_data_list,val_data_list



def main(_):
    writer_train  = tf.python_io.TFRecordWriter(FLAGS.output_path_train)
    writer_val  = tf.python_io.TFRecordWriter(FLAGS.output_path_val)

    # TODO(user): Write code to read in your dataset to examples variable
    import os
    sample_path = "/ob_models/data/rop"
    print(os.path.exists(sample_path))
    total_path = []
    train_path = []
    val_path = []
    for root, _, examples in os.walk(sample_path):
        for example in examples:
            if example.endswith(".json") and len(example) > 15:
                example = os.path.join(root, example)
                total_path.append(example)
    length = len(total_path)
    print(length)
    train_path,val_path = divide_dataset(total_path)
    with open("train_file_list_11_13.txt","w") as tt:
        for train in train_path:
            tt.write(train+"\n")
    with open("val_file_list_11_13.txt","w") as tt:
        for val in val_path:
            tt.write(val+"\n")
    for example in val_path:
        tf_example = create_tf_example(example)
        writer_val.write(tf_example.SerializeToString())
    writer_val.close()
    for example in train_path:
        tf_example = create_tf_example(example)
        writer_train.write(tf_example.SerializeToString())
    writer_train.close()
    print(type_name_count)


if __name__ == '__main__':
    tf.app.run()
