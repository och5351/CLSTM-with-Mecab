import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.reuters.load_data(num_words=1000, test_split=0.2)