# demod for conv kernel

import tensorflow as tf
import numpy as np
import svbrdf.module_svbrdf as tfHelpers
import utils.helpers as helpers
from utils.warp_const import dense_image_warp as warp_const

#Define the the model class, this contains our network definition. Quite similar to the first project.
class Model:
    generatorOutputs = None
    output = None
    inputTensor = None
    useSecondary = True
    useCoordConv = True
    ngf = 64 #Number of filter for the generator
    generatorOutputChannels = 64
    reuse_bool=False
    last_convolutions_channels =[64,32,9]
    pooling_type = "max"
    dynamic_batch_size = None
    firstAsGuide = False
    NoMaxPooling = False

    def __init__(self, input, input_flows, dyn_batch_size, useSecondary=True, ngf=64, generatorOutputChannels=64, reuse_bool=False,  pooling_type="moment", last_convolutions_channels=[64,64,48,32,16,9], useCoordConv = True, firstAsGuide = False, NoMaxPooling = False):
        self.inputTensor = input
        self.input_flows = input_flows
        self.useSecondary = useSecondary
        self.ngf = ngf
        self.generatorOutputChannels = generatorOutputChannels
        self.reuse_bool = reuse_bool
        self.pooling_type = pooling_type
        self.last_convolutions_channels = last_convolutions_channels
        self.dynamic_batch_size = dyn_batch_size
        self.useCoordConv = useCoordConv
        self.firstAsGuide = firstAsGuide
        self.NoMaxPooling = NoMaxPooling

    #Secondary network block, used in the submission
    def _addSecondaryNetBlock(self, input, inputMean, lastGlobalNetworkValue, currentChannels, nextChannels, layerCount):
        if self.useSecondary:
            if inputMean is None:
                inputMean = tf.reduce_mean(input, axis=[1, 2], keep_dims=True)
            summed = input
            if not lastGlobalNetworkValue is None:
                summed = input + tfHelpers.GlobalToGenerator(lastGlobalNetworkValue, input.get_shape()[-1])
            with tf.variable_scope("globalNetwork_fc_%d" % (layerCount + 1)):
                nextGlobalInput = inputMean
                if not lastGlobalNetworkValue is None:
                    nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(lastGlobalNetworkValue, axis = 1), axis=1), inputMean], axis = -1)
                globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, nextChannels, True, "globalNetworkLayer" + str(layerCount + 1))

            return summed, tf.nn.selu(globalNetwork_fc) #returns the sum of this layer + last globalNet output and a new globalNetValue
        else:
            return input, None

    #Encoder of the generator, used in the submission
    def __create_encoder(self, input):
        layers = []
        #input shape is [batch * nbRenderings, height, width, 3]
        if self.useCoordConv:
            coords = helpers.generateCoords(tf.shape(input))
            input = tf.concat([input, coords], axis = -1)

        with tf.variable_scope("encoder_1"):
            convolved = tfHelpers.conv_down(input, self.ngf, useXavier=False)
            convolved, lastGlobalNet = self._addSecondaryNetBlock(convolved, None, None, None ,self.ngf * 2, 1)
            layers.append(convolved)
        #Default ngf is 64
        layer_specs = [
            self.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
        for layerCount, out_channels in enumerate(layer_specs):
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = tfHelpers.lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = tfHelpers.conv_down(rectified, out_channels, useXavier=False)
                #here mean and variance will be [batch, 1, 1, out_channels]
                layers_specs_GlobalNet = layerCount + 1
                if layerCount + 1 >= len(layer_specs) - 1:
                    layers_specs_GlobalNet = layerCount
                outputs, lastGlobalNet = self._addSecondaryNetBlock(convolved, None, lastGlobalNet, out_channels, layer_specs[layers_specs_GlobalNet], len(layers) + 1)
                layers.append(outputs)

        return layers, lastGlobalNet
    
    #Decoder of the generator, used in the submission
    def __create_decoder(self, encoder_results, lastGlobalNet, output_channels):
        layer_specs = [
            (self.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8]
            (self.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 ] => [batch, 4, 4, ngf * 8]
            (self.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 ] => [batch, 8, 8, ngf * 8] #Dropout was 0.5 until here
            (self.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 ] => [batch, 16, 16, ngf * 8]
            (self.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 ] => [batch, 32, 32, ngf * 4]
            (self.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4] => [batch, 64, 64, ngf * 2]
            (self.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
        ]
        decoder_results = []

        num_encoder_layers = len(encoder_results)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = encoder_results[-1]
                else:
                    input = tf.concat([decoder_results[-1], encoder_results[skip_layer]], axis=3)

                rectified = tfHelpers.lrelu(input, 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = tfHelpers.deconv(rectified, out_channels)
                output, lastGlobalNet = self._addSecondaryNetBlock(output, None, lastGlobalNet, out_channels, out_channels, num_encoder_layers + len(decoder_results))
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                decoder_results.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, output_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([decoder_results[-1], encoder_results[0]], axis=3)
            rectified = tfHelpers.lrelu(input, 0.2)
            deconved = tfHelpers.deconv(rectified, output_channels)
            decoder_results.append(deconved)

        return decoder_results[-1], lastGlobalNet


    #This function creates the convolutional neural net taking as input the pooled features from the generator.
    def __createLastConvs(self, input, secondaryNet_input, output_channels, last_channels, reuse_bool = True):
        input, lastGlobalNet = self._addSecondaryNetBlock(input, None, secondaryNet_input, last_channels, output_channels[0], 0)
        #if self.useCoordConv:
        #    coords = helpers.generateCoords(tf.shape(input))
        #    input = tf.concat([input, coords], axis = -1)
        layers = [input]
        for layerCount, chanCount in enumerate(output_channels[:-1]):
            with tf.variable_scope("final_conv_" + str(layerCount)):
                rectified = tfHelpers.lrelu(layers[-1], 0.2)
                convolved = tfHelpers.conv_same(rectified, chanCount, initScale=0.02, useXavier=False, paddingSize = 1)
                output, lastGlobalNet = self._addSecondaryNetBlock(convolved, None, lastGlobalNet, chanCount, output_channels[layerCount + 1], len(layers))
                layers.append(output)
        with tf.variable_scope("final_conv_last"):
            rectified = tfHelpers.lrelu(layers[-1], 0.2)
            #convolved = tfHelpers.conv(layers[-1], output_channels[-1], stride=1, filterSize=3, initScale=0.02, useXavier=True, paddingSize = 1,useBias= True)
            convolved = tfHelpers.conv_same(rectified, output_channels[-1], initScale=0.02, useXavier=True, paddingSize = 1, useBias= True, normKernel = False)
            #convolved, _ = self._addSecondaryNetBlock(convolved, None, lastGlobalNet, output_channels[-1], output_channels[-1], len(layers))
            outputs = tf.tanh(convolved)
            #outputs should be [batch, W, H, C]
            return outputs
            
    
    #call the proper functions to create the generator.
    def __create_generator(self, input, output_channels, reuse_bool = True):
        with tf.variable_scope("generator", reuse=reuse_bool) as scope:
            #generator_output, secondary_output = self.create_generator(input, output_channels, reuse_bool)
            encoder_results, lastGlobalNet = self.__create_encoder(input)
            decoder_results, lastGlobalNet = self.__create_decoder(encoder_results, lastGlobalNet, output_channels)
            #output = tf.tanh(decoder_results)
            generator_output = decoder_results
        return generator_output, lastGlobalNet

    #create the full model
    def create_model(self):
        with tf.variable_scope("trainableModel", reuse=self.reuse_bool) as scope:
            #get all the generator outputs
            generator_output, secondary_output = self.__create_generator(self.inputTensor, self.generatorOutputChannels, self.reuse_bool)
            firstOutput = warp_const(generator_output, self.input_flows)
            secondOutput = secondary_output
            #If no max pooling all images are treated separately. else, process all images and pull them.
            if not self.NoMaxPooling:
                #Separate again the dimension of the batch and the dimension of the number of images.

                tmpOutputs = tf.reshape(firstOutput, [self.dynamic_batch_size, -1, tf.shape(generator_output)[1], tf.shape(generator_output)[2], int(generator_output.get_shape()[3])])
                tmpSecondary = tf.reshape(secondOutput, [self.dynamic_batch_size, -1, int(secondary_output.get_shape()[1])])

                if self.pooling_type == 'moment':
                    output_max = tf.reduce_max(tmpOutputs, axis=1)
                    mask = tf.cast(tf.greater(tmpOutputs, -3e38), tf.float32)
                    nozeros = tf.reduce_sum(mask, axis=1)
                    output_mean = tf.divide(tf.reduce_sum(mask * tmpOutputs, axis=1), nozeros)
                    output_var = tf.divide(tf.reduce_sum(mask * tf.pow(mask * tmpOutputs - tf.expand_dims(output_mean, 1), 2), axis=1), nozeros)
                    pooledGeneratorOutput = tf.concat([output_max, output_mean, output_var], axis=-1)
                    global_max = tf.reduce_max(tmpSecondary, axis=1)
                    global_mean = tf.reduce_mean(tmpSecondary, axis=1)
                    global_var = tf.reduce_mean(tf.pow(tmpSecondary - tf.expand_dims(global_mean, 1), 2), axis=1)
                    pooledSecondaryOutput = tf.concat([global_max, global_mean, global_var], axis=-1)
                elif self.pooling_type == 'mean':
                    output_max = tf.reduce_max(tmpOutputs, axis=1)
                    mask = tf.cast(tf.greater(tmpOutputs, -3e38), tf.float32)
                    output_mean = tf.divide(tf.reduce_sum(mask * tmpOutputs, axis=1), tf.reduce_sum(mask, axis=1))
                    pooledGeneratorOutput = tf.concat([output_max, output_mean], axis=-1)
                    global_max = tf.reduce_max(tmpSecondary, axis=1)
                    global_mean = tf.reduce_mean(tmpSecondary, axis=1)
                    pooledSecondaryOutput = tf.concat([global_max, global_mean], axis=-1)
                elif self.pooling_type == 'max':
                    pooledGeneratorOutput = tf.reduce_max(tmpOutputs, axis=1)
                    pooledSecondaryOutput = tf.reduce_max(tmpSecondary, axis=1)

            
            #Create the final convolutions to process the pooled features and output the maps
            partialOutput = self.__createLastConvs(pooledGeneratorOutput, pooledSecondaryOutput, self.last_convolutions_channels, self.generatorOutputChannels)
            
            #Process the outputs to have 3 channels for all parameter maps.
            self.output = helpers.deprocess_outputs(partialOutput)