import os.path
import os
import time

import numpy as np
import tensorflow as tf
import cv2
import scipy.ndimage

import layer_def as ld
import BasicConvLSTMCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_integer('label_size', 5,
                            """size of label vector""")
DATA_PATH = "../sequence"

def load_data():
  image_files = os.listdir(DATA_PATH)
  sorted_files = sorted(image_files)
  np_images = []
  for file_name in sorted_files:
    file_path = os.path.join(DATA_PATH, file_name)
    np_image = scipy.ndimage.imread(file_path)
    np_images.append(np_image)
  stacked = np.stack(np_images)
  return np.expand_dims(stacked,axis=0), np.asarray([[1,0,0,0,0]])

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

def network(inputs, hidden):
  conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1")
  # conv2
  conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
  # conv3
  conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
  # conv4
  conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4")
  y_0 = conv4
  with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([8,8], [3,3], 4)
    if hidden is None:
      hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
    y_1, hidden = cell(y_0, hidden)

  return y_1, hidden

# make a template for reuse
network_template = tf.make_template('network', network)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 216, 384, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.label_size])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []

    # conv network
    hidden = None
    for i in range(FLAGS.seq_length):
      x_1, hidden = network_template(x_dropout[:,i,:,:,:], hidden)
      x_unwrap.append(x_1)

    # pack them all together 
    x_unwrap = tf.stack(x_unwrap)

    # calc total loss (compare x_t to x_t+1)
    predicted_task_vector = x[:,FLAGS.seq_length-1:,:,:]
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicted_task_vector)
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

    for step in xrange(FLAGS.max_step):
      features, labels = load_data()
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t

      if step%100 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


