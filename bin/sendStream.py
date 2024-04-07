#!/usr/bin/env python

"""Generates a stream to Kafka from a time series csv file.
"""

import argparse
import csv
import json
import sys
import time
import datetime
from confluent_kafka import Producer
import socket
import random
import string
import re

# SQL Injection has now changed to a different dataset.
# The underlying task is still the same.
def data_drift():
    pass


# SQL Injection has now changed into a hypothetical attack pattern
# where if there are any special characters in the input string it
# constitutes an attack.
def concept_drift():
    letters = string.ascii_letters + string.punctuation + string.digits
    inputstr = ''.join(random.choice(letters) for i in range(random.randint(1, 30)))
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    
    if regex.search(inputstr) == None:
        label = 'Safe'
    else:
        label = 'Injection'
    
    return (inputstr, label)


def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg.value()), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', type=str,
                        help='Time series csv file.')
    parser.add_argument('data_drift_filename', type=str,
                        help='Alternate time series csv file')
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')
    parser.add_argument('--speed', type=float, default=1, required=False,
                        help='Speed up time series by a given multiplicative factor.')
    args = parser.parse_args()

    topic = args.topic
    p_key = args.filename

    conf = {'bootstrap.servers': "localhost:9092",
            'client.id': socket.gethostname()}
    producer = Producer(conf)
    
    step = 0
    START_DATA_DRIFT_STEP = 500
    START_CONCEPT_DRIFT_STEP = 1000
    
    rdr = csv.reader(open(args.filename))
    next(rdr)  # Skip header
    #firstline = True
    rdr2 = csv.reader(open(args.data_drift_filename, encoding='utf8'))

    while True:

        try:

            if step < START_DATA_DRIFT_STEP:
                line = next(rdr, None)
                time.sleep(random.randint(1,10)) #sleep from 1-10s
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                
                value_tuple = (line[0], "Injection" if int(line[1])==1 else "Safe")
                result = {}
                result[timestamp] = value_tuple
            
            # Simulate data drift
            elif step >= START_DATA_DRIFT_STEP and step < START_CONCEPT_DRIFT_STEP:
                line = next(rdr2, None)
                time.sleep(random.randint(1,10)) #sleep from 1-10s
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                
                value_tuple = (line[0], "Injection" if int(line[1])==1 else "Safe")
                result = {}
                result[timestamp] = value_tuple
                
            # Simulate concept drift
            elif step >= START_CONCEPT_DRIFT_STEP:
                time.sleep(random.randint(1,10)) #sleep from 1-10s
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                
                value_tuple = concept_drift()
                result = {}
                result[timestamp] = value_tuple
                
            jresult = json.dumps(result)
            producer.produce(topic, key=p_key, value=jresult, callback=acked)
            step = step + 1
            
            producer.flush()

        except TypeError:
            sys.exit()


if __name__ == "__main__":
    main()
