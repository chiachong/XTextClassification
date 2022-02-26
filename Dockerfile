FROM python:3.7-slim

# install python dependencies
RUN pip3 install --upgrade pip setuptools
COPY ./requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# download spacy model
RUN python3 -m spacy download en_core_web_md

# copy configurations
COPY ./.streamlit /.streamlit

# copy model
COPY ./model /model

# copy source codes
COPY ./srcs /srcs
COPY ./start.sh /start.sh

CMD ["./start.sh"]
