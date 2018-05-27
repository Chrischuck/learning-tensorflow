import tensorflow as tf

hw = tf.constant("Hello World!")

session = tf.Session()

print(session.run(hw))

session.close()