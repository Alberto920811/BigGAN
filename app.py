import os
from flask import Flask, request, render_template, jsonify
import sys, struct
import io
import numpy as np
from PIL import Image
from scipy.stats import truncnorm
import tensorflow as tf
import tensorflow_hub as hub
import base64

__author__ = 'EMULINT'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/receive_data", methods=['POST'])
def receive_data():

    num_samples = int(request.form.get('num_samples'))
    truncation = float(request.form.get('truncation'))
    category = request.form.get('category')
    noise_seed = None

    tf.reset_default_graph()
    module = hub.Module('TARS') 
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k) 
          for k, v in module.get_input_info_dict().items()}
    output = module(inputs) 

    input_z = inputs['z']
    input_y = inputs['y']
    input_trunc = inputs['truncation']
    dim_z = input_z.shape.as_list()[1]
    vocab_size = input_y.shape.as_list()[1]

    def truncated_z_sample(batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
        return truncation * values
    
    def one_hot(index, vocab_size=vocab_size):
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output
    
    def one_hot_if_needed(label, vocab_size=vocab_size):
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = one_hot(label, vocab_size)
            assert len(label.shape) == 2
        return label
    
    def sample(sess, noise, label, truncation=1., batch_size=8, vocab_size=vocab_size):
        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
            if label.shape[0] != num:
                raise ValueError('Got # noise samples ({}) != # label samples ({})'.format(noise.shape[0], label.shape[0]))
        label = one_hot_if_needed(label, vocab_size)
        ims = []
        for batch_start in range(0, num, batch_size):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
            ims.append(sess.run(output, feed_dict=feed_dict))
        ims = np.concatenate(ims, axis=0)
        assert ims.shape[0] == num
        ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
        ims = np.uint8(ims)
        return ims
    
    def imgrid(imarray, cols=5, pad=1):
        if imarray.dtype != np.uint8:
            raise ValueError('imgrid input imarray must be uint8')
        pad = int(pad)
        assert pad >= 0
        cols = int(cols)
        assert cols >= 1
        N, H, W, C = imarray.shape
        rows = N // cols + int(N % cols != 0)
        batch_pad = rows * cols - N
        assert batch_pad >= 0
        post_pad = [batch_pad, pad, pad, 0]
        pad_arg = [[0, p] for p in post_pad]
        imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
        H += pad
        W += pad
        grid = (imarray.reshape(rows, cols, H, W, C).transpose(0, 2, 1, 3, 4).reshape(rows*H, cols*W, C))
        if pad:
            grid = grid[:-pad, :-pad]
        return grid
    
    def imshow(a, format='png', jpeg_fallback=True):
        a = np.asarray(a, dtype=np.uint8)
        data = io.BytesIO()
        Image.fromarray(a).save(data, format)
        try:    
            im_data = data.getvalue()
        except IOError:
            if jpeg_fallback and format != 'jpeg':
                print(('Warning: image was too large to display in format "{}"; '
                'trying jpeg instead.').format(format))
                return imshow(a, format='jpeg')
            else: raise
        return im_data

    def save_image(num_samples, truncation, noise_seed, category):
        initializer = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initializer)
        z = truncated_z_sample(num_samples, truncation, noise_seed)
        y = int(category.split(')')[0])
        ims = sample(sess, z, y, truncation=truncation)
        img = imshow(imgrid(ims, cols=min(num_samples, 5)))
        stream = io.BytesIO(img)
        image = Image.open(stream).convert("RGBA")
        stream.close()
        target = os.path.join(APP_ROOT, 'images/')
        if not os.path.isdir(target):
            os.mkdir(target)
        else: print("Couldn't create upload directory: {}".format(target))
        filename = 'out.png'
        destination = "/".join([target, filename])
        return image.save(destination )

    save_image(num_samples, truncation,noise_seed, category)
    with open("images/out.png", "rb") as img_file:
        my_string = str(base64.b64encode(img_file.read()))
        json = {'images':[
               {'mime':'jpeg', 
                'width':100, 
                'height':100,
                'data': my_string
                }]}
    return jsonify(json)
    #return render_template("success.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
