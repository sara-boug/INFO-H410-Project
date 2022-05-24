from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, \
    UpSampling2D, Concatenate, MaxPool2D
from tensorflow.keras.models import Model
from source.config import network_input_shape


class SegNet:
    """
    Defines the segnet model architecture

    """

    def __init__(self, labels_num: int, input_shape):
        self.labels_num = labels_num
        self.input_shape = input_shape

    def get_model(self) -> Model:
        input = Input(batch_input_shape=network_input_shape)
        # Encoder layer
        output_conv1 = self.__layer_conv2(input, 64)
        output_conv2 = self.__layer_conv2(output_conv1, 128)
        output_conv3 = self.__layer_conv3(output_conv2, 256)
        output_conv4 = self.__layer_conv3(output_conv3, 512)
        output_conv5 = self.__layer_conv3(output_conv4, 512)

        # Decoder layer
        output_deconv1 = self.__layer_deconv3(output_conv5, None, 512 + 256)
        output_deconv2 = self.__layer_deconv3(output_deconv1, output_conv4, 512 + 256)
        output_deconv3 = self.__layer_deconv3(output_deconv2, output_conv3, 256 + 128)
        output_deconv4 = self.__layer_deconv3(output_deconv3, output_conv2, 128 + 64)
        output = self.__last_layer_deconv(output_deconv4, output_conv1, 64 + 32, self.labels_num)

        return Model(name="SegNet", inputs=[input], outputs=[output])

    @staticmethod
    def __layer_conv2(input, depth):
        output = Conv2D(depth, (3, 3), padding="same", activation='relu', )(input)
        output = BatchNormalization()(output, training=True)
        output = Conv2D(depth, (3, 3), padding="same", activation='relu', )(output)
        output = BatchNormalization()(output, training=True)
        output = MaxPool2D((2, 2), strides=2, )(output)  # No overlap
        return output

    @staticmethod
    def __layer_conv3(input, depth):
        output = Conv2D(depth, (3, 3), padding="same", activation='relu')(input)
        output = BatchNormalization()(output, training=True)
        output = Conv2D(depth, (3, 3), padding="same", activation='relu')(output)
        output = BatchNormalization()(output, training=True)
        output = Conv2D(depth, (3, 3), padding="same", activation='relu')(output)
        output = BatchNormalization()(output, training=True)
        output = MaxPool2D((2, 2), strides=2)(output)  # No overlap
        return output

    @staticmethod
    def __layer_deconv3(input1, input2, depth):
        output = input1
        if output is None:
            output = Concatenate()([input1, input2])
        output = UpSampling2D(size=(2, 2))(output)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = BatchNormalization()(output, training=True)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = BatchNormalization()(output, training=True)
        return output

    @staticmethod
    def __layer_deconv2(input1, input2, depth):
        output = Concatenate()([input1, input2])
        output = UpSampling2D(size=(2, 2))(output)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = BatchNormalization()(output, training=True)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = BatchNormalization()(output, training=True)
        return output

    @staticmethod
    def __layer_deconv3(input1, input2, depth):
        output = input1
        if output is None:
            output = Concatenate()([input1, input2])
        output = UpSampling2D(size=(2, 2))(output)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = BatchNormalization()(output, training=True)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = BatchNormalization()(output, training=True)
        return output

    @staticmethod
    def __last_layer_deconv(input1, input2, depth, num_labels):
        output = Concatenate()([input1, input2])
        output = UpSampling2D(size=(2, 2))(output)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = BatchNormalization()(output, training=True)
        output = Conv2D(num_labels, (3, 3), padding="same", activation="softmax")(output)
        return output
