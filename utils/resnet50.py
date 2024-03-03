import numpy as np
from keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.models import Model
from keras.layers import UpSampling2D, GlobalAveragePooling2D,Input

from keras.models import Model
from keras.layers import Input, UpSampling2D, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

def extract_resnet50_features(x, trainable_layers=0):
    im_h = 32  # Set the desired size

    # Resize the input images to 224x224
    inputs = Input(shape=(im_h, im_h, 3))
    resize = UpSampling2D(size=(7, 7))(inputs)

    model = ResNet50(include_top=False, weights='imagenet', input_tensor=resize)

    # Freeze layers up to the specified index
    for layer in model.layers[:trainable_layers]:
        layer.trainable = False

    # Remove the final classification layer
    base_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Add Global Average Pooling layer
    final_model = GlobalAveragePooling2D()(base_model.output)

    # Create the modified model
    feature_model = Model(inputs=base_model.input, outputs=final_model)

    print('Extracting features...')
    x_resized = preprocess_input(x)
    features = feature_model.predict(x_resized)

    print('Features shape = ', features.shape)

    return features


def load_cifar10_resnet50():
    # Load CIFAR-10 dataset
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    # Concatenate train and test sets
    x = train_x
    y = train_y.reshape((50000,))

    # Extract features(x_train) using ResNet50
    features = np.zeros((50000, 2048))
    for i in range(50):  # Adjusted the loop limit
        idx = range(i * 1000, (i + 1) * 1000)
        print("The %dth 1000 samples" % i)
        features[idx] = extract_resnet50_features(x[idx])
        
    # Extract features(x_test) using ResNet50
    features_test = np.zeros((10000, 2048))
    for i in range(10):  # Adjusted the loop limit
        idx = range(i * 1000, (i + 1) * 1000)
        print("The %dth 1000 samples" % i)
        features_test[idx] = extract_resnet50_features(test_x[idx])

    # Scale to [0,1]
    features = features/255
    features_test = features_test/255

    return features,features_test,y,test_y
