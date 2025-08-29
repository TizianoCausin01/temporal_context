from scipy.ndimage import zoom
#from scipy.misc import logsumexp
from scipy.special import logsumexp
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
sns.set_style('white')
import tensorflow.compat.v1 as tf
import torch
import cv2
#from google.colab.patches import cv2_imshow # Import cv2_imshow

#changes to the directory with all the functions... to be evaluated in the command window
%cd /Users/tizianocausin/Desktop/backUp20240609/summer2024/dondersInternship/code/myScripts/python_scripts/deep_gaze/

for i in range(3):
    run=i+1
    fn_stim=f'/Users/tizianocausin/Desktop/dataRepository/RepDondersInternship/project1917/stimuli/Project1917_movie_part{run}_24Hz.mp4'
    print('part',i+1)
    cap = cv2.VideoCapture(fn_stim)
    if not cap.isOpened():
      print("Error opening video file")
    fsVid=round(cap.get(cv2.CAP_PROP_FPS),3)
    frames_to_skip = int(fsVid * 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-frames_to_skip
    tVid = np.arange(0,frame_count/fsVid,1/fsVid)
    centerbias_template=np.load('centerbias_mit1003.npy')
    #alternatively, I could use a uniform centerbias:
    #centerbias_template=np.zeros(1024,1024)
    # rescale to match image size
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    centerbias = zoom(centerbias_template, (height/1024,width/1024), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    check_point = 'DeepGazeII.ckpt'  # DeepGaze II
    #check_point = 'ICF.ckpt'  # ICF
    new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))
    input_tensor = tf.get_collection('input_tensor')[0]
    centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
    log_density = tf.get_collection('log_density')[0]
    log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]
    resize_tuple = (int(width*0.1),round(height*.1))
    dg_saliency_model=np.zeros((resize_tuple[1]*resize_tuple[0],frame_count)) #preallocation
    # Read and display frames
    count=0
    while cap.isOpened():
      ret, frame = cap.read()
    
      image_data = frame[np.newaxis, :, :, :]  # BHWC, three channels (RGB), vectorizes the three channels
    
        #tf.reset_default_graph()
    
      with tf.Session() as sess:
          new_saver.restore(sess, check_point)
          log_density_prediction = sess.run(log_density, {
          input_tensor: image_data,
          centerbias_tensor: centerbias_data,
          })
      dg_prediction = np.exp(np.squeeze(log_density_prediction))
      dg_prediction_resized= np.round((cv2.resize(dg_prediction,resize_tuple))*100,9)
      del dg_prediction
      del log_density_prediction
      #print(log_density_prediction.shape)
      #plt.gca().imshow(frame, alpha=0.5)
      #m = plt.gca().matshow(dg_prediction_resized, alpha=0.2, cmap=plt.cm.RdBu)
      #plt.colorbar(m)
      #plt.title('density prediction')
      #plt.axis('off');
      #plt.show()
      dg_prediction_resized=dg_prediction_resized.reshape(-1)
      dg_saliency_model[:,count]=dg_prediction_resized
      count+=1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    # Data to be saved
fn2save= f"/Users/tizianocausin/Desktop/dataRepository/RepDondersInternship/project1917/data/models/Project1917_dg_run{'0'+str(run)}_movie24Hz.mat"
# Save the data to a .mat file
sio.savemat(fn2save, {'dg_saliency_model': dg_saliency_model, 'fsVid': fsVid, 'tVid': tVid })
