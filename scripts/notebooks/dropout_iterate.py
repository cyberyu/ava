import numpy as np
import warnings
import tensorflow as tf
import os
import sys
sys.path.insert(0,'../bert')
import run_classifier_inmem_noexport
from tensorflow.saved_model import tag_constants 
import pickle 
import tensorflow as tf
from bert import tokenization
from io import StringIO
import io
import pandas as pd
import logging

warnings.simplefilter('ignore')

tf.get_logger().setLevel(logging.ERROR)
flags = tf.flags
FLAGS = flags.FLAGS

def getSess():
    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string(
        "export_dir", '../../export_models/percent80/',
        "The dir where the exported model has been written.")
    
    tokenizer = tokenization.FullTokenizer(vocab_file='../../embeddings/uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
    sess = tf.InteractiveSession()
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)
    return sess, tokenizer
    
def predict_inmem(sess, FLAGS, tokenizer, question):
    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)
        
    examples = run_classifier_inmem_noexport.getinput(question)    
    tmp_file = os.path.join('/tmp/', "eval.tf_record")
    label_list = [str(x) for x in range(5)]
    
    # write test sentence to tmp_file
    run_classifier_inmem_noexport.file_based_convert_examples_to_features(examples, label_list,
                                            128, tokenizer,
                                            tmp_file)    
    
    predict_input_fn = run_classifier_inmem_noexport.file_based_input_fn_builder(
        input_file=tmp_file,
        seq_length=128,
        is_training=False,
        drop_remainder=False)


    record_iterator = tf.python_io.tf_record_iterator(path=tmp_file)    
    
    graph = tf.get_default_graph()
    tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
    tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
    tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
    tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
    tensor_outputs = graph.get_tensor_by_name('loss/Softmax:0')
    
    for sentence in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(sentence)

        unique_ids = example.features.feature['unique_ids'].int64_list.value
        input_ids = example.features.feature['input_ids'].int64_list.value
        input_mask = example.features.feature['input_mask'].int64_list.value
        segment_ids = example.features.feature['segment_ids'].int64_list.value
        label_ids = example.features.feature['label_ids'].int64_list.value


        result = sess.run([tensor_outputs],feed_dict={
            tensor_input_ids: np.array(input_ids).reshape(-1, 128),
            tensor_input_mask: np.array(input_mask).reshape(-1, 128),
            tensor_segment_ids: np.array(segment_ids).reshape(-1, 128),
            tensor_label_ids: np.array(label_ids).reshape(-1, 1)
        })  

        return result[0][0]
        
        
tf.app.flags.DEFINE_string('f', '', 'kernel')
session, tokenizer = getSess()

train_df = pd.read_csv('../../alternative_data/5fold/split0/train.tsv', sep='\t', header=None)
train_df.columns=['Questions','Labels']

train_df = train_df[pd.notnull(train_df['Questions'])]
r_full = []

for q in train_df['Questions']:
    r_array = []
    print(q)
    for loop in range(10):
        #r = predict_inmem(session, FLAGS, tokenizer, 'How much can clients add via automatic investment plan')
        r = predict_inmem(session, FLAGS, tokenizer, q)
        r_array.append(r)
    r_full.append(r_array)

pickle.dump(r_full, open('../../dropout_results/train_80percent.pkl','wb'))

# all_mean = np.array(np.mean(r_full[0],0))
# all_var = np.array(np.var(r_full[0],0))

# # B = numpy.array([3])
# # A = numpy.array([1, 2, 2])
# # B = numpy.append( B , A )


# for loop in range(1,len(r_full)):
#     all_mean = np.vstack((all_mean, np.array(np.mean(r_full[loop],0))))
#     all_var  = np.vstack((all_var, np.array(np.var(r_full[loop],0))))

# # save it as pickle files
# import pickle 

# train_10 = [all_mean, all_var]

test_df = pd.read_csv('../../alternative_data/5fold/split0/test.tsv', sep='\t', header=None)
test_df.columns=['Questions','Labels']

r_full = []
    
for q in test_df['Questions']:
    r_array = []
    print(q)
    for loop in range(10):
        #r = predict_inmem(session, FLAGS, tokenizer, 'How much can clients add via automatic investment plan')
        r = predict_inmem(session, FLAGS, tokenizer, q)
        r_array.append(r)
    r_full.append(r_array)
pickle.dump(r_full, open('../../dropout_results/test_80percent.pkl','wb'))    


test_df = pd.read_csv('../../alternative_data/data_oos.txt', sep='\t', header=None)
test_df.columns=['Questions','Labels']

r_full = []
    
for q in test_df['Questions']:
    r_array = []
    print(q)
    for loop in range(10):
        #r = predict_inmem(session, FLAGS, tokenizer, 'How much can clients add via automatic investment plan')
        r = predict_inmem(session, FLAGS, tokenizer, q)
        r_array.append(r)
    r_full.append(r_array)
    
pickle.dump(r_full, open('../../dropout_results/negative_80percent.pkl','wb'))   
