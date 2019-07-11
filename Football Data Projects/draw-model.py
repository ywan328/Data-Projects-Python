import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam

from ann_visualizer.visualize import ann_viz

def load_data():
    print('Loading features.csv...')
    features = np.loadtxt('features.csv', delimiter=',', dtype=np.float32)
    print('Loading labels.csv...')
    labels = np.loadtxt('labels.csv', delimiter=',', dtype=np.float32)
    return features, labels

def build_model(optimizer, n1, n2, n3, keep_prob):
    model = Sequential()
    model.add(BatchNormalization())

    # hidden layer 1
    model.add(Dense(n1, input_dim=691, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(keep_prob))

    # hidden layer 2
    model.add(Dense(n2, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(keep_prob))

    # # hidden layer 3
    model.add(Dense(n3, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(keep_prob))

    # output layer
    model.add(Dense(3, activation='softmax', kernel_initializer='glorot_normal'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
	model = build_model('sgd', 512, 256, 128, 0.9)
	X, Y = load_data()
	n_instances = 60000
	model.fit(X[:n_instances], Y[:n_instances], validation_split=0.005, epochs=20, batch_size=1)
	ann_viz(model, title="");