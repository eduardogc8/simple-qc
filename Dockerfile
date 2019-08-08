FROM ufoym/deepo:all-py36-cu100

RUN pip install -U --no-cache-dir \
        nltk==3.4.4 \
        pytorch-transformers \
        numpy \
        pandas \
        scikit-learn==0.21.3 \
        tqdm

RUN python -c "import nltk; nltk.download('punkt')"
RUN python download_word_embeddings.py

WORKDIR /docker-share
ENV PYTHONPATH /docker-share
