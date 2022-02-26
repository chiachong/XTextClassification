import sys
import spacy
import numpy as np
import tensorflow as tf
from flask import Flask, request
from flask_cors import cross_origin
from flask_restful import Api
sys.path.append('srcs')
import utils

SPACY_MODEL = 'en_core_web_md'
TF_MODEL = 'model'
EMBED_DIM = 300
MAX_LEN = 200
LABEL_DICT = {
    'positive': 0,
    'negative': 1,
}

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)

nlp = spacy.load(SPACY_MODEL, disable=['tagger', 'parser', 'ner'])
tf_model = utils.load_tf_model(TF_MODEL)


@app.route('/api/v1/predict_sentiment', methods=['POST'])
@cross_origin()
def predict():
    """ """
    text = request.get_json()['text']
    embedded, doc = utils.encode_text(text, nlp, MAX_LEN, EMBED_DIM, True)
    x = np.expand_dims(embedded, axis=0)
    with tf.GradientTape() as gtape:
        attn_output, predictions = tf_model(x)
        pred_class_id = np.argmax(predictions[0])
        pred_class = list(LABEL_DICT.keys())[pred_class_id]
        pred_score = float(predictions[0][pred_class_id])
        loss = predictions[:, pred_class_id]
        # get the gradient
        grads = gtape.gradient(loss, attn_output)
        # Each entry of this tensor is the mean intensity of the
        # gradient over a specific feature map channel
        pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1))
        # Multiply each channel in the feature map array by
        # "how important this channel is"
        # then normalize the heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, attn_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
    # truncate heatmap and convert to list
    heatmap = heatmap[0, :len(doc)].tolist()
    # list of tokenize words
    tokens = [d.text for d in doc]
    return {'tokens': tokens, 'sentiment': pred_class, 'score': pred_score,
            'heatmap': heatmap}


if __name__ == '__main__':
    app.run(debug=False)
