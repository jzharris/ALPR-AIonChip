import tensorflow as tf
import os

meta_path = '/media/qiujing/e5439522-63c4-4b7c-a968-fefee6a3d960/omead/aiOnChip/AIonChip_HOZ/main/LP_segmentation/pruned_models/converted_checkpoint/model__pruned_trained_quant.ckpt.meta' # Your .meta file
output_node_names = ['lambda_1_1/Identity']    # Output nodes
TRAIN_DIR = '/media/qiujing/e5439522-63c4-4b7c-a968-fefee6a3d960/omead/aiOnChip/AIonChip_HOZ/main/LP_segmentation/pruned_models/converted_checkpoint/'

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)
    tf.contrib.saved_model.save_tf
    # Load weights
    saver.restore(sess, '/media/qiujing/e5439522-63c4-4b7c-a968-fefee6a3d960/omead/aiOnChip/AIonChip_HOZ/main/LP_segmentation/pruned_models/converted_checkpoint/model__pruned_trained_quant.ckpt')

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
