import numpy as np
import tensorflow as tf


def encode_text(input_text: str, nlp: 'spacy.lang.en.English', max_len: int,
                embed_dim: int, return_doc: bool = False):
    """ Encode and apply padding or truncate the sentence. """
    doc = nlp(input_text)
    embedding = np.zeros((max_len, embed_dim), dtype=np.float32)
    for i in range(min(len(doc), max_len)):
        if doc[i].has_vector:
            embedding[i] = doc[i].vector
    if return_doc:
        return embedding, doc
    return embedding


def encode_label(input_label: str, label_dict: dict):
    """ """
    label_int = label_dict[input_label]
    return (np.arange(len(label_dict)) == label_int).astype(np.float32)


def load_tf_model(text_cat_model: str):
    """
    Load and modify the text classification model to add the last attention
    layer output to the model outputs.
    """
    model = tf.keras.models.load_model(text_cat_model)
    # get last attention layer by name
    attn_layers = []
    for layer in model.layers:
        if 'attention' in layer.name:
            attn_layers.append(layer.name)
    last_layer = model.get_layer(attn_layers[-1])
    heatmap_model = tf.keras.models.Model(
        [model.inputs], [last_layer.output, model.output]
    )
    return heatmap_model
