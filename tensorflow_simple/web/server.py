
from flask import Flask
from flask import request
from flask import render_template

import cPickle
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imageQA_tensorflow import *

app = Flask(__name__, static_url_path = "/images", static_folder='images')

config = get_config()


# http://127.0.0.1:5000/
@app.route('/')
def my_form():
    return render_template('iqa.html')


@app.route('/', methods=['POST'])
def my_form_post(answer=None):
    
    filename='/home/seonhoon/Desktop/workspace/ImageQA/data/dict.pkl'
    
    with open(filename, 'rb') as fp:
        idx2word, word2idx, idx2answer, answer2idx = cPickle.load(fp)

    text = request.form['text']
    print text
    
    question=text.split()
    
    q_idx=[]
    for i in range(len(question)):
        q_idx.append(word2idx[question[i]])
    q_idx=np.array(q_idx)
    
    print q_idx

    #running caffe and tensorflow seems not so easy simultaneously
    pd.read_pickle('/home/seonhoon/Desktop/workspace/ImageQA_Web/cnn.pkl')
    x_img = np.array([pd.read_pickle('/home/seonhoon/Desktop/workspace/ImageQA_Web/cnn.pkl')['cnn_feature'][0].tolist()])
 


    x , x_mask = prepare_data([q_idx], config.steps)

            
    y = test_sample(x, x_mask, x_img)
    
    print idx2answer[y[0]]
    
    params = {'answer' : idx2answer[y[0]], 'text' : text}
    
    return render_template('iqa.html', **params) 

    
if __name__ == '__main__':
    app.run()
