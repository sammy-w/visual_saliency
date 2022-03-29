#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import os
import matplotlib.pyplot as plt
import cv2
import progressbar 

#get videos

img = cv2.imread("ilse_meme.jpg")
# converting BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.imshow(img)
plt.axis('off')
plt.show()


''' Overall goal: 

##calculate saliency maps per frame (per video)

## save these saliency maps to a folder

## later: compute measure of spatial dispersion vs. concentration for the frame-by-frame density maps

'''

os.chdir('.\deep_gaze')
# load precomputed log density over a 1024x1024 image
centerbias_template = np.load('centerbias.npy')  
# rescale to match image size
centerbias = zoom(centerbias_template, (img.shape[0]/1024, img.shape[1]/1024), order=0, mode='nearest')
# renormalize log density
centerbias -= logsumexp(centerbias)

image_data = img[np.newaxis, :, :, :]  # BHWC, three channels (RGB)
centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)

#use pre-trained deep gaze model
tf.reset_default_graph()
check_point = 'DeepGazeII.ckpt'  # DeepGaze II
#check_point = 'ICF.ckpt'  # ICF
new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

input_tensor = tf.get_collection('input_tensor')[0]
centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
log_density = tf.get_collection('log_density')[0]
log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]


vid = '33.mp4'
i = 1

with tf.Session() as sess:
    
    new_saver.restore(sess, check_point)

    #os.chdir()

    #cv2.cap #read video start looping through frames
    cap = cv2.VideoCapture(vid)
    
    amount_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #show progressbar
    bar = progressbar.ProgressBar(maxval=amount_of_frames, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    #writing numpy array of saliency maps per frame to this folder (chunking frame by frame instead of holding the whole array in memory)
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        
        #if i>5: #debugging
        #    break


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        centerbias = zoom(centerbias_template, (frame.shape[0]/1024, frame.shape[1]/1024), order=0, mode='nearest')
        # renormalize log density
        centerbias -= logsumexp(centerbias)

        image_data = frame[np.newaxis, :, :, :]  # BHWC, three channels (RGB)
        centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)

        log_density_prediction = sess.run(log_density, {
            input_tensor: image_data,
            centerbias_tensor: centerbias_data,
        })
        
        #save saliency map to an outputfile (2d array), should somehow be possible to append this to one file?
        os.chdir('./density_maps')
        np.savez_compressed(f'{i}',np.exp(log_density_prediction[0, :, :, 0]))        
        os.chdir('..')

        #update progress bar
        bar.update(i+1)
        i += 1

print(log_density_prediction.shape)

#show output
plt.gca().imshow(frame, alpha=0.2)
m = plt.gca().matshow((log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.title('log density prediction')
plt.axis('off');
plt.show()

plt.gca().imshow(frame, alpha=0.2)
m = plt.gca().matshow(np.exp(log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.title('density prediction')
plt.axis('off');
plt.show()

#plt.savefig("output.png", dpi=300)

np.savez_compressed('test',test)

test2 = np.load('./density_maps/1.npz') #dictionary object

np.size(frame)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.gca().imshow(frame, alpha=0.2)
m = plt.gca().matshow(test2['arr_0'], alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.title('density prediction')
plt.axis('off');
plt.savefig('deep_gaze_output.png', bbox_inches='tight')
plt.show()


