from flask import Flask, render_template, request, redirect,session
import os
import cv2
import numpy as np
import cv2
import dlib
import numpy as np
from imutils import face_utils
from face import face_swap
from PIL import Image
import base64
app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER='images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        file =request.files['closedEyesImage']
        # session['eyes_closed']=np.array(Image.open(file)).tolist()
       

        file.save(os.path.join(app.config['UPLOAD_FOLDER'],'eye_close.jpg'))
        return redirect ('/second_page')
    return render_template('main.html')

@app.route('/second_page',methods=['GET','POST'])
def process_image():

    if request.method == 'POST':
        file =request.files['openEyesImage']
        # session['eyes_opened']=np.array(Image.open(file)).tolist()

        # check=session.get('eyes_closed')
        # print(np.array(check))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],'eye_open.jpg'))
        return redirect('/final_page')
    
    return render_template('second_page.html')


@app.route('/final_page')
def final():
    eye_open='images/eye_open.jpg'
    eye_closed='images/eye_close.jpg'
    image=face_swap(eye_open,eye_closed)
    print(type(image),image.shape)

    _, encoded_image = cv2.imencode('.png', image)
    image_data = base64.b64encode(encoded_image).decode('utf-8')

    return render_template('/final_page.html',image_data=image_data)



if __name__ == '__main__':
    app.run(debug=True)
