import cv2
import tensorflow as tf
from tensorflow import Graph,Session

cap = cv2.VideoCapture(0)

cv2.namedWindow("IMage Classification Testing")
saver = tf.train.import_meta_graph('/home/siddharth/Downloads/final_model/archive/data.pkl')
while True:
    # true or false for ret if the capture is there or not
    ret, frame = cap.read()
    #saver = tf.train.import_meta_graph('/home/siddharth/Downloads/final_model/archive/data.pkl')
    with tf.Session() as sess:
        saver.restore(sess, '/home/siddharth/Downloads/final_model/archive/data.pkl')
        detection_graph = tf.get_default_graph()
        input_tensor = detection_graph.get_tensor_by_name('input_tensor:0')  # Get the input tensor
        output_tensor = detection_graph.get_tensor_by_name('output_tensor:0')  # Get the output te
        while True:
            # true or false for ret if the capture is there or not
            ret, frame = cap.read()  # read fram from the webcam
            feed = {input_tensor: frame}
            prediction = sess.run(tf.argmax(output_tensor, 1), feed_dict=feed)  # make prediction

