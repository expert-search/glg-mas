FROM python:3.9.6

MAINTAINER "spencer.nelle@gmail.com"

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["./src/visualization/app/glg_web_app.py"]