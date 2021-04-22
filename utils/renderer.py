import tensorflow as tf
import utils.helpers
import math
import numpy as np

#A renderer which implements the Cook-Torrance GGX rendering equations
class GGXRenderer:
    includeDiffuse = True

    def __init__(self, includeDiffuse = True):
        self.includeDiffuse = includeDiffuse

    #Compute the diffuse part of the equation
    def tf_diffuse(self, diffuse, specular):
        #return diffuse * (1.0 - specular) / math.pi
        return diffuse / math.pi

    #Compute the distribution function D driving the statistical orientation of the micro facets.
    def tf_D(self, roughness, NdotH):
        alpha = tf.square(roughness)
        underD = 1/tf.maximum(0.001, (tf.square(NdotH) * (tf.square(alpha) - 1.0) + 1.0))
        return (tf.square(alpha * underD)/math.pi)

    #Compute the fresnel approximation F
    def tf_F(self, specular, VdotH):
        sphg = tf.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
        return specular + (1.0 - specular) * sphg

    #Compute the Geometry term (also called shadowing and masking term) G taking into account how microfacets can shadow each other.
    def tf_G(self, roughness, NdotL, NdotV):
        return self.G1(NdotL, tf.square(roughness)/2) * self.G1(NdotV, tf.square(roughness)/2)

    def G1(self, NdotW, k):
        return 1.0/tf.maximum((NdotW * (1.0 - k) + k), 0.001)

    #This computes the equations of Cook-Torrance for a BRDF without taking light power etc... into account.
    def tf_calculateBRDF(self, svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight):

        h = helpers.tf_Normalize(tf.add(wiNorm, woNorm) / 2.0)
        #Put all the parameter values between 0 and 1 except the normal as they should be used between -1 and 1 (as they express a direction in a 360° sphere)        
        diffuse = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,3:6]), 0.0,1.0), axis = 1)
        normals = tf.expand_dims(svbrdf[:,:,:,0:3], axis=1)
        normals = tf.stack([normals[...,0], -normals[...,1], normals[...,2]], axis=-1)
        normals = helpers.tf_Normalize(normals)
        specular = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,9:12]), 0.0, 1.0), axis = 1)
        roughness = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,6:9]), 0.0, 1.0), axis = 1)
        #Avoid roughness = 0 to avoid division by 0
        roughness = tf.maximum(roughness, 0.001)

        #If we have multiple lights to render, add a dimension to handle it.
        if multiLight:
            diffuse = tf.expand_dims(diffuse, axis = 1)
            normals = tf.expand_dims(normals, axis = 1)
            specular = tf.expand_dims(specular, axis = 1)
            roughness = tf.expand_dims(roughness, axis = 1)

        NdotH = helpers.tf_DotProduct(normals, h)
        NdotL = helpers.tf_DotProduct(normals, wiNorm)
        NdotV = helpers.tf_DotProduct(normals, woNorm)

        VdotH = helpers.tf_DotProduct(woNorm, h)

        diffuse_rendered = self.tf_diffuse(diffuse, specular)
        D_rendered = self.tf_D(roughness, tf.maximum(0.0, NdotH))
        G_rendered = self.tf_G(roughness, tf.maximum(0.0, NdotL), tf.maximum(0.0, NdotV))
        F_rendered = self.tf_F(specular, tf.maximum(0.0, VdotH))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered
        
        #Add the diffuse part of the rendering if required.        
        if self.includeDiffuse:
            result = result + diffuse_rendered
        return result, NdotL

    #Main rendering function, this is the computer graphics part, it generates an image from the parameter maps. For pure deep learning purposes the only important thing is that it is differentiable.
    def render(self, svbrdf, wi, wo, currentConeTargetPos, currentLightPos, light_factor=None):
        wiNorm = helpers.tf_Normalize(wi)
        woNorm = helpers.tf_Normalize(wo)

        result, NdotL = self.tf_calculateBRDF(svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, False)
        
        currentConeTargetDir = currentLightPos - currentConeTargetPos
        coneTargetNorm = helpers.tf_Normalize(currentConeTargetDir)
        distanceToConeCenter = (tf.maximum(0.0, helpers.tf_DotProduct(wiNorm, coneTargetNorm)))
        lampDistance = tf.sqrt(tf.reduce_sum(tf.square(wi), axis = -1, keep_dims=True))

        if light_factor is None:
            light_factor = [0.1, 0.1, 0.1, np.exp(1.6)]

        lampFactor = tf.reshape(light_factor[0:3], [1,1,1,1,3]) * helpers.tf_lampAttenuation_pbr(lampDistance) * tf.pow(distanceToConeCenter, light_factor[3])
        result = result * lampFactor
        result = result * tf.maximum(0.0, NdotL)
        return result

    #Main rendering function, this is the computer graphics part, it generates an image from the parameter maps. For pure deep learning purposes the only important thing is that it is differentiable.
    def render_ambient(self, svbrdf, wi, wo, ambient_factor=None):
        wiNorm = helpers.tf_Normalize(wi)
        woNorm = helpers.tf_Normalize(wo)

        result, NdotL = self.tf_calculateBRDF(svbrdf, wiNorm, woNorm, None, None, False)

        if ambient_factor is None:
            ambient_factor = 0.1

        lampFactor = tf.reshape(ambient_factor, [1,1,1,1,1])
        result = result * lampFactor
        result = result * tf.maximum(0.0, NdotL)
        return result
