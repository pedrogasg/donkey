#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from PIL import Image
class TensorWriter():

    def __init__(self, save_dir='data'):
        self.start_time = time.time()
        filename = os.path.join(save_dir, 'train-' + str(self.start_time) + '.tfrecords')
        self.writer = tf.python_io.TFRecordWriter(filename)

    def __del__(self):
        self.writer.close()

    def run(self, *args):

        record_time = int(time.time() - self.start_time)

        record = dict(zip(self.inputs, args))

        self.put_record(record, record_time)


    def _byte_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



    def put_record(self, data, record_time):
        """
        Save values like images that can't be saved in the csv log and
        return a record with references to the saved values that can
        be saved in a csv.
        """
        

        logger.info('write tensor record at time {} '.format(str(record_time)))
        features = self.serialize_data(data)
        example = tf.train.Example(features=tf.train.Features(features))
        self.writer(example.SerializeToString())
        return self.current_ix

    def serialize_data(self, data):

        feature = {}
        
        for key, val in data.items():
            typ = self.get_input_type(key)

            if typ is 'str':
                json_data[key] = _byte_feature(val)
            
            elif typ is 'float':
                json_data[key] = _float_feature(val)

            elif typ is 'int':
                json_data[key] = _int64_feature(val)

            elif typ is 'boolean':
                json_data[key] = _int64_feature(val)

            elif typ == 'numpy.float32':
                if (val is not None):
                    json_data[key] = _float_feature(val.item())
                else:
                    json_data[key] = _float_feature(0.0)

            elif typ is 'image':
                json_data[key]= _int64_feature(np.uint8(Image.toarray(val)))

            elif typ == 'image_array':
                json_data[key]= _int64_feature(np.uint8(val))

            elif typ == None:
                # Do nothing
                continue

            else:
                msg = 'TensorRecord does not know what to do with this type {}'.format(typ)
                raise TypeError(msg)

        return feature