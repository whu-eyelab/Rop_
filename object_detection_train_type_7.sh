export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/model_main.py \
    --pipeline_config_path=./object_detection/faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config_type_7 \
    --model_dir=./object_detection/faster_rcnn_inception_v2_coco_2018_01_28/rop_type_7 \
    --num_train_steps=2000000 \
    --sample_1_of_n_eval_examples=2 \
