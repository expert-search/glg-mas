FROM python:3.9.6

MAINTAINER "spencer.nelle@gmail.com"

COPY . /app

WORKDIR /app

#Install Package Dependencies
RUN pip install -r requirements.txt

# Uncompress LDA Model
RUN unzip ./models/lda_250k/250k_ldamodel_185_files.zip -d ./models/lda_250k

# Install NLTK dependencies
RUN python ./src/models/init_nltk.py

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["./src/visualization/app/glg_web_app.py"]