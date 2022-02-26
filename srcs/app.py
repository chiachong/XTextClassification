import json
import requests
import numpy as np
import streamlit as st

TF_ENDPOINT = 'http://127.0.0.1:5000/api/v1/predict_sentiment'
st.set_page_config('XTextClassification')


def main():
    """ """
    load_css()
    st.title('Explainable Text Classification')
    st.subheader('Sentiment Analysis')
    st.write("""\
        Sentiment analysis is one of the common natural language understanding \
        techniques used to determine the subjective information in the data. \
        The result of this analysis would be the emotional tone the data carry, \
        whether they are positive, negative or neutral. This technique could \
        help businesses in marketing, advertising, market research, etc, by \
        having a closer look at the sentiment in customer feedback.
    """)
    st.write("""\
        This demo would predict the sentiment of the input text and highlight \
        the words that are contributing to the results. The darker the words, \
        the more influential the words are.
    """)

    sample_text = 'This is a great phone, it packs good spec and manages to '\
                  'do so for a lower cost than many would expect.'
    text = st.text_area('Input text', sample_text)
    submit_predict = st.button('Predict', key='button_submit_predict')
    if submit_predict and text:
        st.subheader('Predictions')
        result_holder = st.empty()
        result_holder.text('Predicting...')
        try:
            pred_results = predict_sentiment(TF_ENDPOINT, text)
            if pred_results is not None:
                html = prediction_html(**pred_results)
                result_holder.write(html, unsafe_allow_html=True)
            else:
                result_holder.error('Error predicting sentiment!')
        except requests.exceptions.ConnectionError:
            result_holder.error('Error connecting to server. Please try again later.')


def load_css():
    """ """
    html = f"""
        <style>
            #results {{
                border: none;
                border-radius: 5px;
                margin-top: 1em;
                padding: 20px;
                height: auto;
                box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            }}
            #results:hover {{
                box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.35);
            }}
        </style>
    """
    st.write(html, unsafe_allow_html=True)


@st.cache(show_spinner=False)
def predict_sentiment(endpoint: str, text: str):
    """ """
    headers = {
        'content-type': 'application/json',
        'Accept-Charset': 'UTF-8',
    }
    data = json.dumps({'text': text})
    r = requests.post(endpoint, data=data, headers=headers).json()
    if 'sentiment' in r:
        return r
    return None


@st.cache(show_spinner=False)
def prediction_html(tokens, sentiment: str, score: float, heatmap: np.ndarray) -> str:
    """ HTML scripts to display text to be labelled. """
    sentiment_color = '171, 245, 217' if sentiment == 'positive' else '245, 183, 177'
    sentiment_style = f"""
        border-radius: 5px;
        border-width: 0px;
        background-color: rgb({sentiment_color});
        padding: 3px 6px;
    """
    return f"""
        <div>
            Sentiment:
            <span style="{sentiment_style}">
                {sentiment}, {score:.2f}
            </span>
        </div>
        <div id="results">
            {highlight_text(tokens, heatmap)}
        </div>
    """


@st.cache(show_spinner=False)
def highlight_text(tokens, heatmap):
    """ """
    heatmap = np.clip(heatmap, 0.3, 1.0)
    highlights = []
    for i in range(len(tokens)):
        heat = heatmap[i]
        highlights.append(
            f"""\
                <span style="color:rgba(0, 0, 0, {heat});">{tokens[i]}</span>\
            """
        )
        # if heat != np.max(heatmap):
        #     highlights.append(
        #         f"""\
        #         <span style="color:rgba(0, 0, 0, {heat});">{tokens[i]}</span>\
        #         """
        #     )
        # else:
        #     highlights.append(
        #         f"""\
        #         <span style="color:rgba(0, 0, 0, 1);"><b>{tokens[i]}</b></span>\
        #         """
        #     )
    return ' '.join(highlights)


if __name__ == '__main__':
    main()
