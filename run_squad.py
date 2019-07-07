from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('xlnet') # walkaround due to submodule absolute import...

import os
import json

import tensorflow as tf
import numpy as np
import sentencepiece as sp

from xlnet import xlnet
import function_builder
import prepro_utils
import model_utils
import squad_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", default=None, help="The name of the task to train.")
flags.DEFINE_string("model_config_path", default=None, help="Config file of the pre-trained model.")
flags.DEFINE_string("init_checkpoint", default=None, help="Initial checkpoint of the pre-trained model.")
flags.DEFINE_string("spiece_model_file", default="", help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="", help="Directory for saving the finetuned model.")
flags.DEFINE_string("predict_dir", default="", help="Dir for predictions.")
flags.DEFINE_string("output_dir", default="", help="Output dir for TF records.")
flags.DEFINE_string("train_file", default="", help="Path of train file.")
flags.DEFINE_string("predict_file", default="", help="Path of prediction file.")
flags.DEFINE_bool("overwrite_data", default=False, help="If False, will use cached data if available.")

flags.DEFINE_bool("do_train", default=False, help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False, help="Whether to run evaluation.")
flags.DEFINE_bool("do_predict", default=False, help="Whether to run prediction.")
flags.DEFINE_bool("do_export", default=False, help="Whether to run exporting.")

flags.DEFINE_bool("do_prepro", default=False, help="Perform preprocessing only.")
flags.DEFINE_integer("num_proc", default=1, help="Number of preprocessing processes.")
flags.DEFINE_integer("proc_id", default=0, help="Process id for preprocessing.")

flags.DEFINE_bool("lower_case", default=False, help="Enable lower case nor not.")
flags.DEFINE_integer("doc_stride", default=128, help="Doc stride")
flags.DEFINE_integer("max_seq_length", default=512, help="Max sequence length")
flags.DEFINE_integer("max_query_length", default=64, help="Max query length")
flags.DEFINE_integer("max_answer_length", default=64, help="Max answer length")
flags.DEFINE_integer("train_batch_size", default=48, help="Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", default=32, help="Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", default=32, help="Total batch size for predict.")

flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1, help="Initialization std when init is uniform.")
flags.DEFINE_bool("init_global_vars", default=False, help="If true, init all global vars. If false, init trainable vars only.")

flags.DEFINE_integer("train_steps", default=8000, help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_integer("max_save", default=5, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=1000, help="Save the model for every save_steps. If None, not to save any model.")
flags.DEFINE_integer("shuffle_buffer", default=2048, help="Buffer size used for shuffle.")

flags.DEFINE_integer("n_best_size", default=5, help="n best size for predictions")
flags.DEFINE_integer("start_n_top", default=5, help="Beam size for span start.")
flags.DEFINE_integer("end_n_top", default=5, help="Beam size for span end.")
flags.DEFINE_string("target_eval_key", default="best_f1", help="Use has_ans_f1 for Model I.")

flags.DEFINE_bool("use_bfloat16", default=False, help="Whether to use bfloat16.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")
flags.DEFINE_string("summary_type", default="last", help="Method used to summarize a sequence into a vector.")

flags.DEFINE_float("learning_rate", default=3e-5, help="initial learning rate")
flags.DEFINE_float("min_lr_ratio", default=0.0, help="min lr ratio for cos decay.")
flags.DEFINE_float("lr_layer_decay_rate", default=0.75, help="lr[L] = learning_rate, lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("num_hosts", 1, "How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", 1, "Total number of TPU cores to use.")
flags.DEFINE_string("tpu_job_name", None, "TPU worker job name.")
flags.DEFINE_string("tpu", None, "The Cloud TPU name to use for training.")
flags.DEFINE_string("tpu_zone", None, "GCE zone where the Cloud TPU is located in.")
flags.DEFINE_string("gcp_project", None, "Project name for the Cloud TPU-enabled project.")
flags.DEFINE_string("master", None, "TensorFlow master URL")
flags.DEFINE_integer("iterations", 1000, "number of iterations per TPU training loop.")

class InputExample(object):
    """A single training/test example for simple sequence classification.
    
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.is_impossible = is_impossible
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = "qas_id: %s" % (prepro_utils.printable_text(self.qas_id))
        s += ", question_text: %s" % (prepro_utils.printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
            s += ", is_impossible: %r" % (self.is_impossible)
        return s

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tok_start_to_orig_index,
                 tok_end_to_orig_index,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 p_mask,
                 segment_ids,
                 paragraph_len,
                 cls_index,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tok_start_to_orig_index = tok_start_to_orig_index
        self.tok_end_to_orig_index = tok_end_to_orig_index
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.p_mask = p_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.cls_index = cls_index
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class SquadProcessor(object):
    """Processor for SQuAD dataset."""
    def __init__(self,
                 data_dir,
                 task_name):
        self.data_dir = data_dir
        self.task_name = task_name
    
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        data_path = os.path.join(self.data_dir, "train-{0}".format(self.task_name), "train-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "dev-{0}".format(self.task_name), "dev-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        data_path = os.path.join(self.data_dir, "test-{0}".format(self.task_name), "test-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def _read_json(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)["data"]
                return data_list
        else:
            raise FileNotFoundError("data path not found: {0}".format(data_path))
    
    def _get_example(self,
                     data_list,
                     is_training):
        examples = []
        for entry in data_list:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    orig_answer_text = None
                    is_impossible = False
                    
                    if is_training:
                        is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError("For training, each question should have exactly 1 answer.")
                        
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            start_position = answer["answer_start"]
                        else:
                            start_position = -1
                            orig_answer_text = ""
                    
                    example = InputExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        paragraph_text=paragraph_text,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        is_impossible=is_impossible)
                    
                    examples.append(example)
        
        return examples

class XLNetTokenizer(object):
    """Default text tokenizer for XLNet"""
    def __init__(self,
                 sp_model_file,
                 lower_case=False):
        """Construct XLNet tokenizer"""
        self.sp_processor = sp.SentencePieceProcessor()
        self.sp_processor.Load(sp_model_file)
        self.lower_case = lower_case
    
    def tokenize(self,
                 text):
        """Tokenize text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        tokenized_pieces = prepro_utils.encode_pieces(self.sp_processor, processed_text, return_unicode=False)
        return tokenized_pieces
    
    def encode(self,
               text):
        """Encode text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        encoded_ids = prepro_utils.encode_ids(self.sp_processor, processed_text)
        return encoded_ids
    
    def token_to_id(self,
                    token):
        """Convert token to id for XLNet"""
        return self.sp_processor.PieceToId(token)
    
    def id_to_token(self,
                    id):
        """Convert id to token for XLNet"""
        return self.sp_processor.IdToPiece(id)
    
    def tokens_to_ids(self,
                      tokens):
        """Convert tokens to ids for XLNet"""
        return [self.sp_processor.PieceToId(token) for token in tokens]
    
    def ids_to_tokens(self,
                      ids):
        """Convert ids to tokens for XLNet"""
        return [self.sp_processor.IdToPiece(id) for id in ids]

class XLNetExampleConverter(object):
    """Default example converter for XLNet"""
    def __init__(self,
                 max_seq_length,
                 max_query_length,
                 tokenizer):
        """Construct XLNet example converter"""
        self.special_vocab_list = ["<unk>", "<s>", "</s>", "<cls>", "<sep>", "<pad>", "<mask>", "<eod>", "<eop>"]
        self.special_vocab_map = {}
        for (i, special_vocab) in enumerate(self.special_vocab_list):
            self.special_vocab_map[special_vocab] = i
        
        self.segment_vocab_list = ["<a>", "<b>", "<cls>", "<sep>", "<pad>"]
        self.segment_vocab_map = {}
        for (i, segment_vocab) in enumerate(self.segment_vocab_list):
            self.segment_vocab_map[segment_vocab] = i
                
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.tokenizer = tokenizer
    
    def convert_single_example(self,
                               example,
                               logging=False):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        pass
    
    def convert_examples_to_features(self,
                                     examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        for (idx, example) in enumerate(examples):
            if idx % 1000 == 0:
                tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

            feature = self.convert_single_example(example, logging=(idx < 20))
            features.append(feature)

        return features
    
    def file_based_convert_examples_to_features(self,
                                                examples,
                                                output_file,
                                                output_type="train"):
        """Convert a set of `InputExample`s to a TFRecord file."""
        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        
        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for (idx, example) in enumerate(examples):
                if idx % 1000 == 0:
                    tf.logging.info("Writing example %d of %d" % (idx, len(examples)))
                
                feature = convert_single_example(example, logging=(idx < 20))
                
                features = collections.OrderedDict()
                features["unique_ids"] = create_int_feature([feature.unique_id])
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_float_feature(feature.input_mask)
                features["p_mask"] = create_float_feature(feature.p_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["cls_index"] = create_int_feature([feature.cls_index])
                
                if output_type == "train":
                    features["start_positions"] = create_int_feature([feature.start_position])
                    features["end_positions"] = create_int_feature([feature.end_position])
                    features["is_impossible"] = create_float_feature([1 if feature.is_impossible else 0])
                
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
