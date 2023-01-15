import tensorflow as tf

def cnn(x_train, y_train, x_test, y_test, rows, columns):
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape= (rows, columns)))
    cnn_model.add(layers.MaxPooling2D(2, 2))
    cnn_model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (rows, columns)))
    cnn_model.add(layers.MaxPooling2D(2, 2))
