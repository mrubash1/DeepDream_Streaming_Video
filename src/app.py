from flask import Flask,request,Response
import base64
import numpy as np
import PIL.Image
import StringIO
import cStringIO
import json
# from deep_dream import DeepDream

app = Flask(__name__)

# deep_dream = DeepDream()
filter_map = {'Tornado': 84, 'Flowers': 139, 'Fireworks':50,
              'Caves': 38,'Mountains': 142, 'Van Gogh': 1 }


@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"

@app.route('/deepdream', methods=['POST'])
def deep_dream():
    data = request.form
    img_bytes = base64.decodestring(data['file_data'])
    img = PIL.Image.open(StringIO.StringIO(img_bytes))

    octaves = data['octaves']
    iterations = data['iterations']
    filter = filter_map[data['filter']]

    img = np.float32(img)

    processed_array = deep_dream.load_parameters_run_deep_dream_return_image(img,iter_n=iterations,octave_n=octaves,t_obj_filter=filter)
    distortedimage = PIL.Image.fromarray(processed_array)

    buffer = cStringIO.StringIO()
    distortedimage.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue())

@app.route('/test', methods=['POST'])
def test():

    data = dict(request.get_json())
    img_bytes = base64.decodestring(data['image']['file_data'])
    img = PIL.Image.open(StringIO.StringIO(img_bytes))
    # PIL.Image._show(img)
    img = np.float32(img)

    formatted_img = (img * 255 / np.max(img)).astype('uint8')

    distortedimage = PIL.Image.fromarray(formatted_img)
    buffer = cStringIO.StringIO()
    distortedimage.save(buffer, format="JPEG")
    d = {"image_data" : base64.b64encode(buffer.getvalue())}
    return json(data)



@app.route('/styletransfer', methods=['POST'])
def runStyleTransfer():
    #run style transfer here
    print "test"

if __name__ == '__main__':
    app.run(debug=True)
