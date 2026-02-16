FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m nltk.downloader vader_lexicon

ENV PORT=8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
