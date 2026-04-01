FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader -d /var/task/nltk_data stopwords vader_lexicon

COPY . .

CMD ["main.handler"]
