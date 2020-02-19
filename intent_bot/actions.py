from rasa_sdk import Action
import requests
import json
from rasa_sdk.events import SlotSet
import numpy as np
import pickle
import numpy as np
import warnings
import tensorflow as tf
import collections
import os
import sys
import pickle

warnings.simplefilter('ignore')

sys.path.insert(0,'./bert')

import run_classifier_inmem_noexport
from tensorflow.saved_model import tag_constants
from bert import tokenization
from io import StringIO
import io

flags = tf.flags
FLAGS = flags.FLAGS


class ActionGetAnswers(Action):

    def __init__(self):
        self.session, self.tokenizer = self.getSess()
        self.label2_text = pickle.load(open('num2label381classes.p','rb'))
        tf.app.flags.DEFINE_string('f', '', 'kernel')

    def getSess(self):
        flags = tf.flags
        FLAGS = flags.FLAGS

        flags.DEFINE_string(
            "export_dir", '/Users/shiyu/Documents/Project/intent_bot_working/bert_small_10epoch/',
            "The dir where the exported model has been written.")

        tokenizer = tokenization.FullTokenizer(vocab_file='/Users/shiyu/Documents/Project/intent_bot_working/bert_small_10epoch/vocab.txt',
                                               do_lower_case=True)
        sess = tf.InteractiveSession()
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)
        return sess, tokenizer

    def predict_inmem(self, sess, FLAGS, tokenizer, question):
        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        examples = run_classifier_inmem_noexport.getinput(question)
        tmp_file = os.path.join('/tmp/', "eval.tf_record")
        label_list = [str(x) for x in range(107)]

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

            result = sess.run([tensor_outputs], feed_dict={
                tensor_input_ids: np.array(input_ids).reshape(-1, 128),
                tensor_input_mask: np.array(input_mask).reshape(-1, 128),
                tensor_segment_ids: np.array(segment_ids).reshape(-1, 128),
                tensor_label_ids: np.array(label_ids).reshape(-1, 1)
            })

            return result[0][0]

    def name(self):
        return 'action_get_answers'


    def run(self, dispatcher, tracker, domain):
        question = (tracker.latest_message)['text']

        result = self.predict_inmem(self.session, FLAGS, self.tokenizer, question)

        label = np.argmax(result)
        
        entropy = -np.sum(np.multiply(result, np.log(result)))

        if entropy <1.5:
            message = 'Your intent classification is '  + str(self.label2_text[label]) + ' num ' + str(label) +' and Entropy is ' + str(entropy)
        else:
            message = 'Sorry I do not know the answer. Entropy is '+ str(entropy)
        dispatcher.utter_message(message)


if __name__ == "__main__":
    session, tokenizer = getSess()

