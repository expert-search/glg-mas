FROM python:3.9.6

MAINTAINER "spencer.nelle@gmail.com"

COPY . /app

WORKDIR /app

#Install Package Dependencies
RUN pip install -r requirements.txt

# Uncompress LDA Model Directory
RUN unzip -d ./models/ ./LDA.zip 

# Copy the uncompressed LDA Model files to the appropriate directory for serving
RUN pip install -e .

# Install NLTK dependencies
RUN python ./src/models/init_nltk.py

# Install SpaCy dependencies
RUN python -m spacy download en_core_web_sm

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["./src/visualization/app/glg_web_app.py"]