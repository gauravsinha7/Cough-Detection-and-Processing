#!/usr/bin/python
# Use argument to provide the file name as  .wav file format
import sys
from vggish_input import waveform_to_examples, wavfile_to_examples
import numpy as np
import tensorflow as tf
from keras.models import load_model
import vggish_params
from pathlib import Path
import label
import wget

MODEL_PATH = "models/audio_detection.hdf5"
print("=====")
print("Checking model... ")
print("=====")
model_filename = "models/audio_detection.hdf5"
label_model = Path(model_filename)

###########################
# Load Model
###########################
context = label.everything
context_mapping = label.context_mapping
trained_model = model_filename
other = True
selected_file = str(sys.argv[1])
selected_context = 'everything'




print("Using deep learning model: %s" % (trained_model))

def audio_predict (selected_files):
    model = load_model(trained_model)
    context = context_mapping[selected_context]
    graph = tf.get_default_graph()

    labels = dict()
    for k in range(len(context)):
        labels[k] = context[k]

    ###########################
    # Read Wavfile and Make Predictions
    ###########################
    x = wavfile_to_examples(selected_file)
    with graph.as_default():

        x = x.reshape(len(x), 96, 64, 1)
        predictions = model.predict(x)

        for k in range(len(predictions)):
            prediction = predictions[k]
            m = np.argmax(prediction)
            ## compare the label and use it
            if(label.to_human_labels[labels[m]]=="Coughing"):
                return ("Prediction: %s (%0.2f)" % ("Cough Detected", prediction[m]))
            else:
                return "No Cough Sample in Audio"

print(audio_predict(selected_file))
