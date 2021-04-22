import os
import tensorflow as tf
import numpy as np
import math

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keep_dims=True))
    return tf.div(tensor, Length)

# Computes the dot product between 2 tensors (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_DotProduct(tensorA, tensorB):
    return tf.reduce_sum(tf.multiply(tensorA, tensorB), axis = -1, keep_dims=True)

#Physically based lamp attenuation
def tf_lampAttenuation_pbr(distance):
    return 1.0 / tf.square(distance)

#Clip values between min an max
def squeezeValues(tensor, min, max):
    return tf.clip_by_value(tensor, min, max)

# Generate an array grid between -1;1 to act as the "coordconv" input layer (see coordconv paper)
def generateCoords(inputShape):
    crop_size = inputShape[-2]
    firstDim = inputShape[0]

    Xcoords= tf.expand_dims(tf.lin_space(-1.0, 1.0, crop_size), axis=0)
    Xcoords = tf.tile(Xcoords,[crop_size, 1])
    Ycoords = -1 * tf.transpose(Xcoords) #put -1 in the bottom of the table
    Xcoords = tf.expand_dims(Xcoords, axis = -1)
    Ycoords = tf.expand_dims(Ycoords, axis = -1)
    coords = tf.concat([Xcoords, Ycoords], axis=-1)
    coords = tf.expand_dims(coords, axis = 0)#Add dimension to support batch size and nbRenderings should now be [1, 256, 256, 2].
    coords = tf.tile(coords, [firstDim, 1, 1, 1]) #Add the proper dimension here for concat
    return coords

# Generate an array grid between -1;1 to act as each pixel position for the rendering.
def generateSurfaceArray(crop_size, pixelsToAdd = 0):
    totalSize = crop_size + (pixelsToAdd * 2)
    surfaceArray=[]
    XsurfaceArray = tf.expand_dims(tf.lin_space(-1.0, 1.0, totalSize), axis=0)
    XsurfaceArray = tf.tile(XsurfaceArray,[totalSize, 1])
    YsurfaceArray = tf.transpose(XsurfaceArray) #put -1 in the bottom of the table
    XsurfaceArray = tf.expand_dims(XsurfaceArray, axis = -1)
    YsurfaceArray = tf.expand_dims(YsurfaceArray, axis = -1)

    surfaceArray = tf.concat([XsurfaceArray, YsurfaceArray, tf.zeros([totalSize, totalSize,1], dtype=tf.float32)], axis=-1)
    surfaceArray = tf.expand_dims(tf.expand_dims(surfaceArray, axis = 0), axis = 0)#Add dimension to support batch size and nbRenderings
    return surfaceArray

#Adds a little bit of noise
def addNoise(renderings):
    shape = tf.shape(renderings)
    stddevNoise = tf.exp(tf.random_normal((), mean = np.log(0.005), stddev=0.3))
    noise = tf.random_normal(shape, mean=0.0, stddev=stddevNoise)
    return renderings + noise

#Generate a random direction on the upper hemisphere with gaps on the top and bottom of Hemisphere. Equation is described in the Global Illumination Compendium (19a)
def tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.05):
    r1 = tf.random_uniform([batchSize, nbRenderings, 1], 0.0 + lowEps, 1.0 - highEps, dtype=tf.float32)
    r2 =  tf.random_uniform([batchSize, nbRenderings, 1], 0.0, 1.0, dtype=tf.float32)
    r = tf.sqrt(r1)
    phi = 2 * math.pi * r2
    #min alpha = atan(sqrt(1-r^2)/r)
    x = r * tf.cos(phi)
    y = r * tf.sin(phi)
    z = tf.sqrt(1.0 - tf.square(r))
    finalVec = tf.concat([x, y, z], axis=-1) #Dimension here should be [batchSize,nbRenderings, 3]
    return finalVec
    
#Generate a distance to compute for the specular renderings (as position is important for this kind of renderings)
def tf_generate_distance(batchSize, nbRenderings):
    gaussian = tf.random_normal([batchSize, nbRenderings, 1], 0.5, 0.75, dtype=tf.float32) # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return (tf.exp(gaussian))
    
#generate the diffuse rendering for the loss computation
def tf_generateDiffuseRendering(batchSize, nbRenderings, targets, outputs, renderer):
    currentViewPos = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightPos = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)

    wi = currentLightPos
    wi = tf.expand_dims(wi, axis=2)
    wi = tf.expand_dims(wi, axis=2)

    wo = currentViewPos
    wo = tf.expand_dims(wo, axis=2)
    wo = tf.expand_dims(wo, axis=2)

    #Add a dimension to compensate for the nb of renderings
    #targets = tf.expand_dims(targets, axis=-2)
    #outputs = tf.expand_dims(outputs, axis=-2)

    #Here we have wi and wo with shape [batchSize, height,width, nbRenderings, 3]
    renderedDiffuse = renderer.tf_Render(targets,wi,wo, None, "diffuse", useAugmentation = False, lossRendering = True)[0]

    renderedDiffuseOutputs = renderer.tf_Render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]#tf_Render_Optis(outputs,wi,wo)
    #renderedDiffuse = tf.Print(renderedDiffuse, [tf.shape(renderedDiffuse)],  message="This is renderings targets Diffuse: ", summarize=20)
    #renderedDiffuseOutputs = tf.Print(renderedDiffuseOutputs, [tf.shape(renderedDiffuseOutputs)],  message="This is renderings outputs Diffuse: ", summarize=20)
    return [renderedDiffuse, renderedDiffuseOutputs]

#generate the specular rendering for the loss computation
def tf_generateSpecularRendering(batchSize, nbRenderings, surfaceArray, targets, outputs, renderer):
    currentViewDir = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightDir = currentViewDir * tf.expand_dims([-1.0, -1.0, 1.0], axis = 0)
    #Shift position to have highlight elsewhere than in the center.
    currentShift = tf.concat([tf.random_uniform([batchSize, nbRenderings, 2], -1.0, 1.0), tf.zeros([batchSize, nbRenderings, 1], dtype=tf.float32) + 0.0001], axis=-1)

    currentViewPos = tf.multiply(currentViewDir, tf_generate_distance(batchSize, nbRenderings)) + currentShift
    currentLightPos = tf.multiply(currentLightDir, tf_generate_distance(batchSize, nbRenderings)) + currentShift

    currentViewPos = tf.expand_dims(currentViewPos, axis=2)
    currentViewPos = tf.expand_dims(currentViewPos, axis=2)

    currentLightPos = tf.expand_dims(currentLightPos, axis=2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=2)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    #targets = tf.expand_dims(targets, axis=-2)
    #outputs = tf.expand_dims(outputs, axis=-2)
    #targets = tf.Print(targets, [tf.shape(targets)],  message="This is targets in specu renderings: ", summarize=20)
    renderedSpecular = renderer.tf_Render(targets,wi,wo, None, "specu", useAugmentation = False, lossRendering = True)[0]
    renderedSpecularOutputs = renderer.tf_Render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]
    #tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)

    #renderedSpecularOutputs = tf.Print(renderedSpecularOutputs, [tf.shape(renderedSpecularOutputs)],  message="This is renderings outputs Specular: ", summarize=20)
    return [renderedSpecular, renderedSpecularOutputs]

def tf_generateTopRendering(batchSize, nbRenderings, surfaceArray, targets, outputs, renderer):
    fov = tf.random_uniform([batchSize, nbRenderings, 1], 25.0, 35.0)
    dist = 1 / tf.tan(fov / 2 / 180 * np.pi)
    xy = tf.random_uniform([batchSize, nbRenderings, 2], -1.0, 1.0)
    pos = tf.concat([xy, dist], axis=-1)

    pos = tf.expand_dims(pos, axis=2)
    pos = tf.expand_dims(pos, axis=2)

    wi = pos - surfaceArray

    #targets = tf.expand_dims(targets, axis=-2)
    #outputs = tf.expand_dims(outputs, axis=-2)
    #targets = tf.Print(targets, [tf.shape(targets)],  message="This is targets in specu renderings: ", summarize=20)
    renderedTop = renderer.tf_Render(targets,wi,wi, None, "specu", useAugmentation = False, lossRendering = True)[0]
    renderedTopOutputs = renderer.tf_Render(outputs,wi,wi, None, "", useAugmentation = False, lossRendering = True)[0]
    #tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)

    #renderedSpecularOutputs = tf.Print(renderedSpecularOutputs, [tf.shape(renderedSpecularOutputs)],  message="This is renderings outputs Specular: ", summarize=20)
    return [renderedTop, renderedTopOutputs]

#Put the normals and roughness back to 3 channel for easier processing.
def deprocess_outputs(outputs):
    partialOutputedNormals = outputs[:,:,:,0:2] * 3.0 #The multiplication here gives space to generate direction with angle > pi/4
    outputedDiffuse = outputs[:,:,:,2:5]
    outputedRoughness = outputs[:,:,:,5]
    outputedSpecular = outputs[:,:,:,6:9]
    normalShape = tf.shape(partialOutputedNormals)
    newShape = [normalShape[0], normalShape[1], normalShape[2], 1]
    #normalShape[-1] = 1
    tmpNormals = tf.ones(newShape, tf.float32)

    normNormals = tf_Normalize(tf.concat([partialOutputedNormals, tmpNormals], axis = -1))
    outputedRoughnessExpanded = tf.expand_dims(outputedRoughness, axis = -1)
    return tf.concat([normNormals, outputedDiffuse, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedSpecular], axis=-1)
