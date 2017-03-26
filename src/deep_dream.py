#!/usr/bin/env python3

from __future__ import print_function
import os
from io import BytesIO
import numpy as np
import PIL.Image
from IPython.display import clear_output, Image, display
import tensorflow as tf
import time
import uuid

class DeepDream(object):
    """
    DeepDream Image Processer
    Please declare model_fn & layer of analysis
    """

    def __init__(self,
                 model_fn='tensorflow_inception_graph.pb',
                 layer='mixed4d_3x3_bottleneck_pre_relu'):
        self.load_graph(os.path.expandvars(model_fn))
        self.k5x5 = self.setup_k()
        self.layer = layer

    def load_graph(self, model_fn='tensorflow_inception_graph.pb'):
        # creating TensorFlow session and loading the model
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)
        with tf.gfile.FastGFile(model_fn, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # define the input tensor
        self.t_input = tf.placeholder(np.float32, name='input')
        # Unclear why imagenet_mean variable standard in deep dream
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(self.t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input':t_preprocessed})

    def setup_k(self):
        k = np.float32([1,4,6,4,1])
        k = np.outer(k, k)
        return k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

    def showarray(self, a, fmt='jpeg'):
        a = np.uint8(np.clip(a, 0, 1)*255)
        f = BytesIO()
        PIL.Image.fromarray(a).save(f, fmt)
        display(Image(data=f.getvalue()))

    def visstd(self, a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

    def T(self, layer):
        '''Helper for getting layer output tensor'''
        return self.graph.get_tensor_by_name("import/%s:0"%self.layer)

    def tffunc(self, *argtypes):
        '''
        Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    # Helper function that uses TF to resize an image
    def resize(self, img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]

    def calc_grad_tiled(self, img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = self.sess.run(t_grad, {self.t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def lap_split(self, img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, self.k5x5, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, self.k5x5*4, tf.shape(img), [1,2,2,1])
            hi = img-lo2
        return lo, hi

    def lap_split_n(self, img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = self.lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(self, levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, self.k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
        return img

    def normalize_std(self, img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img/tf.maximum(std, eps)

    def lap_normalize(self, img, scale_n=4):
        '''Perform the Laplacian pyramid normalization.'''
        img = tf.expand_dims(img,0)
        tlevels = sel.flap_split_n(img, scale_n)
        tlevels = list(map(self.normalize_std, tlevels))
        out = self.lap_merge(tlevels)
        return out[0,:,:,:]
    
    def render_deepdream(self,t_obj, img0,
                         iter_n=10, step=1.5, 
                         octave_n=4, octave_scale=1.4,
                         show_image=False):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, self.t_input)[0] # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = img0
        octaves = []
        for i in range(octave_n-1):
            hw = img.shape[:2]
            lo = self.resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-self.resize(lo, hw)
            #img = lo
            img = lo.eval(session=self.sess)
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = (self.resize(img, hi.shape[:2])+hi).eval(session=self.sess)
                
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))
                print('.',end = ' ')
            clear_output()
            #Added as a flag to not have to show the image every iteration
            if show_image == True:
                self.showarray(img/255.0)
        return img/255.0
    

    def load_parameters_run_deep_dream_return_image(self,
                image,
                name=uuid.uuid4(),
                t_obj_filter= 139,
                iter_n=10,
                step=1.5,
                octave_n=4,
                octave_scale=1.4,
                show_image=False):
        '''
        Image must be an np_float32 datatype
        '''
        assert isinstance(image,np.ndarray)
        start_time=time.time()
        output_image=self.render_deepdream(
                              self.T(self.layer)[:,:,:,t_obj_filter],
                              image,
                              iter_n=int(iter_n),
                              step=int(step),
                              octave_n=int(octave_n),
                              octave_scale=float(octave_scale),
                              show_image=show_image)
        print ('Processing time:', time.time()-start_time)
        return output_image


def load_image_into_memory_from_file(filename='pilatus800.jpg', show_image=False):
    '''
    Load an image into memory as a numpy.ndarray
    '''
    img0 = PIL.Image.open(os.path.expandvars(filename))
    img = np.float32(img0)
    if show_image:
        showarray(img)
    return img


# to run in console
if __name__ == '__main__':

    #Use CPU only --> Temporary Flag
    #os.environ['CUDA_VISIBLE_DEVICES']=""

    deepdream = DeepDream(model_fn='$DD_STREAM/data/models/tensorflow_inception_graph.pb',
                          layer='mixed4d_3x3_bottleneck_pre_relu')

    #Load image into file
    filename='$DD_STREAM/data/pilatus800.jpg'
    print ('Loading image into memory')
    img=load_image_into_memory_from_file(filename=filename,show_image=False)

    print ('Running Deep Dream...')
    output_image=deepdream.load_parameters_run_deep_dream_return_image(img,octave_n=4,show_image=False)
