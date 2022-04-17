from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense
from tensorflow.keras.models import Model


class CNN:

    def __init__(self, labels_num: int, input):
        self.labels_num = labels_num
        self.input = input

    def get_model(self) ->Model:
        input = Input(self.input)
        output= self.__layer_conv2(input,64)
        output= self.__layer_conv2(output,128)
        output= self.__layer_conv3(output,256)
        output = self.__layer_conv3(output,512)
        output = self.__layer_conv3(output,512)
        return  Model(name="VGG-16", inputs=[input], outputs=[output])

    @staticmethod
    def __layer_conv2(input, depth):
        output = Conv2D(depth, (3, 3), padding="same", )(input)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = MaxPooling2D((2, 2))(output)
        return output

    @staticmethod
    def __layer_conv3(input, depth):
        output = Conv2D(depth, (3, 3), padding="same", )(input)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = Conv2D(depth, (3, 3), padding="same", )(output)
        output = MaxPooling2D((2, 2))(output)
        return output

    @staticmethod
    def __last_layer(input, depth):
        output = Dense(depth)(input)
        output = Dense(depth, activation="softmax")(output)
        return output
    @staticmethod
    def __deconv_layer(input,depth):
        pass


