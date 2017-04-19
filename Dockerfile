FROM python:latest

# ADD .
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN wget https://s3.amazonaws.com/aml-sample-data/banking.csv

# Chop up the training data into train and test
RUN cat banking.csv | head -n 30001 > banking-train.csv && \
    cat banking.csv | tail -n 11188 > banking-test.csv && \
    rm banking.csv

COPY *.py /app/
RUN mv *.csv /app/
WORKDIR /app

CMD [ "python", "./myonn-banking.py" ]
