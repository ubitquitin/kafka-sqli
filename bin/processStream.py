#!/usr/bin/env python

"""Consumes stream for printing all messages to the console.
"""

import argparse
import json
import sys
import time
import socket
from confluent_kafka import Consumer, KafkaError, KafkaException
import joblib
from collections import deque

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


# Model retraining parameters
threshold_accuracy = 0.9  # Threshold accuracy below which retraining is triggered
consecutive_steps = 5  # Number of consecutive steps with accuracy below the threshold to trigger retraining
current_consecutive_steps = 0  # Counter for consecutive steps with low accuracy
N = 10  # Number of previous datapoints to use for accuracy check

# queue to hold the past N datapoints the model has seen for retraining.
HISTORICAL_DATA_MEM = 500
q = deque()

class_dict = {'Safe': 0,
              'Injection': 1}

client = MlflowClient()

# Initiates model retraining if accuracy drops too low
def check_retraining_required(model):
    global current_consecutive_steps
    run = mlflow.active_run()

    name, step, ts, val = zip(*client.get_metric_history(run.info.run_id, key='accuracy'))
    val = [x[1] for x in val]
    ts = [x[1] for x in ts]

    runs = pd.DataFrame({'value': val, 'timestamp': ts})
    runs = runs.sort_values(by="timestamp", ascending=False)
    
    # Get the latest N runs
    latest_metrics = runs.iloc[:N]    
    # Filter out None values
    accuracies = latest_metrics.value
    print('avg acc: ', np.mean(accuracies))
    
    if len(accuracies) > 0 and np.mean(accuracies) < threshold_accuracy:
        current_consecutive_steps += 1
        if current_consecutive_steps >= consecutive_steps:
            print("Initiating model retraining...")
            # Extract the last N datapoints for retraining
            X_train, y_train = extract_last_N_datapoints(model)
            
            # Retrain the model
            retrain_model(X_train, y_train)
            current_consecutive_steps = 0  # Reset the counter
            return True
    else:
        current_consecutive_steps = 0  # Reset the counter
        return False
    
        
# Extracts datapoints from mlflow to retrain model on.        
def extract_last_N_datapoints(model):

    historical_x, historical_y = zip(*q)
    historical_x = [model.encode(x) for x in historical_x]
    return np.array(historical_x), np.array(historical_y)


# Retrains model
def retrain_model(X_train, y_train):
    #print(X_train.shape)
    #print(y_train.shape)
    new_lr = LogisticRegression()
    new_lr.fit(X_train, y_train)
    joblib.dump(new_lr, 'joblib_retrained_LR.pkl')
    return


def msg_process(msg, model, lr, answers, predictions, mlflowstep):

    # Print the current time and the message.
    time_start = time.strftime("%Y-%m-%d %H:%M:%S")
    val = msg.value()
    dval = json.loads(val)
    for key, value in dval.items():
        answer = value[1]
        if len(q) == HISTORICAL_DATA_MEM:
            q.popleft()
        q.append((value[0], class_dict[value[1]]))
        a = model.encode(value[0]).reshape(1, -1)
        prediction = lr.predict(a)
    print(dval, 'Pred: ', prediction)
    # Assumes dval is 1 key and 1 value...
    if class_dict[answer] == prediction[0]:
        print('\x1b[1;32;40m' +'Model Correct!' + '\x1b[0m')
    else:
        print('\x1b[1;31;40m' +'Model Wrong!' + '\x1b[0m')
    
    answers.append(class_dict[answer])
    predictions.append(prediction)
    
    # Calculate performance metrics
    accuracy = accuracy_score(answers, predictions)
    precision = precision_score(answers, predictions)
    recall = recall_score(answers, predictions)
    f1 = f1_score(answers, predictions)


    mlflow.log_param("model_type", "binary_classifier")
    mlflow.log_metric("accuracy", accuracy, step=mlflowstep)
    mlflow.log_metric("precision", precision, step=mlflowstep)
    mlflow.log_metric("recall", recall, step=mlflowstep)
    mlflow.log_metric("f1_score", f1, step=mlflowstep)
    # Log other relevant information or metrics

def main():
    iteration = 0
    START_DATA_DRIFT_STEP = 500
    START_CONCEPT_DRIFT_STEP = 1000
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')

    args = parser.parse_args()

    conf = {'bootstrap.servers': 'localhost:9092',
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': socket.gethostname()}

    consumer = Consumer(conf)

    running = True
    
    model = SentenceTransformer('C:/Users/ROHAN/time-series-kafka-demo/model/')
    lr = joblib.load('joblib_RL_Model.pkl')
    answers = []
    predictions = []
    
    # # Start the MLflow server with default local file-based backend store
    # mlflow.server.start(
    #     backend_store_uri="./mlflow",
    #     default_artifact_root="./mlflow",
    #     host="0.0.0.0"
    # )
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080") 
    mlflow.set_experiment("sqli_detection_experiment")
    
    # Log predictions and metrics with MLflow
    mlflow.start_run()

    try:
        while running:
            consumer.subscribe([args.topic])

            msg = consumer.poll(1)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write('Topic unknown, creating %s topic\n' %
                                     (args.topic))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                if iteration > N and check_retraining_required(model):
                    lr = joblib.load('joblib_retrained_LR.pkl')
                msg_process(msg, model, lr, answers, predictions, iteration)
                iteration += 1
                if iteration == START_DATA_DRIFT_STEP:
                    print('DATA DRIFT STARTED!')
                    
                if iteration == START_CONCEPT_DRIFT_STEP:
                    print('CONCEPT DRIFT STARTED!')

    except KeyboardInterrupt:
        pass

    finally:
        # Close down consumer to commit final offsets.
        consumer.close()


if __name__ == "__main__":
    main()
