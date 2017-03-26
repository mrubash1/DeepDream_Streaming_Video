from flask import Flask,request,Response
import base64
import numpy as np

app = Flask(__name__)


filter_map = {'Tornado': 84, 'Flowers': 139, 'Fireworks':50,
              'Caves': 38,'Mountains': 142, 'Van Gogh': 1 }

@app.route('/deepdream', methods=['POST'])
def deep_dream():
    data = request.form
    img_bytes = base64.decodestring(data['file_data'])
    octaves = data['octaves']
    iterations = data['iterations']
    filter = filter_map[data['filter']]
    image = np.asarray(bytearray(img_bytes), dtype="uint8")

    #run deep dream here

    #distortedimage = deepDream()

    return base64.asbytes()

@app.route('/styletransfer', methods=['POST'])
def runStyleTransfer():
    #run style transfer here
    print "test"

if __name__ == '__main__':
    app.run()
