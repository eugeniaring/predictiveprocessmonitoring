import os
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log import log as lg
import pandas as pd
from pm4py.objects.log.log import Trace, EventLog
from datetime import datetime as dt
import sklearn
from sklearn.preprocessing import LabelEncoder
import dateutil.parser

######################simple index#################################

def main_si():

    #----------->specify path of training and test sets
    train_path = '../data/BPIChallenge2011_training_0-80.xes'
    test_path = '../data/BPIChallenge2011_testing_80-100.xes'

    #import training log
    train_log = xes_importer.apply(train_path)
    activity_names_1 = get_activity_names(train_log)

    #import test log
    test_log = xes_importer.apply(test_path)
    activity_names_2 = get_activity_names(test_log)

    #create list of unique activities in all the dataset
    activity_names = activity_names_1 + activity_names_2
    activity_names = sorted(set(activity_names), key=lambda x: activity_names.index(x))

    #obtain simple index encoding for each case in training log
    train_encoded_log = []
    for case_index, case in enumerate(train_log):
        encoded_row = [case.attributes["concept:name"]]
        encoded_row.extend(encode_trace_simple_index(case, activity_names))
        encoded_row.append(case.attributes["label"])
        train_encoded_log.append(encoded_row)

    #obtain simple index encoding for each case in test log
    test_encoded_log = []
    for case_index, case in enumerate(test_log):
        encoded_row = [case.attributes["concept:name"]]
        encoded_row.extend(encode_trace_simple_index(case, activity_names))
        encoded_row.append(case.attributes["label"])
        test_encoded_log.append(encoded_row)

    print('Number of features: %d' % len(activity_names))
    print('Feature list: ', activity_names)
    print("\n")

    #check if all the cases have only 20 events
    print(train_encoded_log[:4])
    print(len(train_encoded_log))
    for i in range(len(train_encoded_log)):
        print(len(train_encoded_log[i]))

    #create the dataframe for training set
    train_df = pd.DataFrame(data=train_encoded_log, columns=compute_columns())
    print(train_df.head())
    train_df.to_csv("train_si.csv")

    #create the dataframe for test set
    test_df = pd.DataFrame(data=test_encoded_log, columns=compute_columns())
    print(test_df.head())
    test_df.to_csv("test_si.csv")

    return train_df, test_df



def encode_trace_simple_index(trace:Trace,activity_names,prefix_length=20):

    #create list of 0s, that will be useful to contain the events of a specific case
    activity_names_case=[str(0)]*prefix_length
    for event_idx, event in enumerate(trace):
             #if the event index is smaller than a specific length threshold (20)
             if event_idx < prefix_length:
                #extract name of the event
                activity_names_case[event_idx]=event['concept:name']
             else:
               break;
    #convert the names of the events into numeric features
    le = LabelEncoder()
    le.fit([str(0)]+activity_names)
    encoding = le.transform(activity_names_case)
    return encoding


#function to obtain the list of all the activities of the event log
def get_activity_names(log):
    activity_names = []
    for case_index, case in enumerate(log):
        for event_index, event in enumerate(case):
            activity_names.append(event['concept:name'])
    return sorted(set(activity_names), key=lambda x: activity_names.index(x))

#function that calculates the columns of the simple index encoded dataframe
def compute_columns(prefix_length=20):
    return ['trace_id']+["event_"+str(i+1) for i in range(prefix_length)]+['label']

if __name__== "__main__":
  df_train,df_test=main_si()

#########################timestamp encoding##################################à


def main_ts():

    # specify path of training and test sets
    train_path = '../data/BPIChallenge2011_training_0-80.xes'
    test_path = '../data/BPIChallenge2011_testing_80-100.xes'

    #import training log
    train_log = xes_importer.apply(train_path)
    activity_names_1 = get_activity_names(train_log)

    #import test log
    test_log = xes_importer.apply(test_path)
    activity_names_2 = get_activity_names(test_log)

    #create list of unique activities in all the dataset
    activity_names = activity_names_1 + activity_names_2
    activity_names = sorted(set(activity_names), key=lambda x: activity_names.index(x))

    #create list of unique timestamps in all the dataset
    all_ts = get_ts_list(train_log) + get_ts_list(test_log)
    all_ts = sorted(set(all_ts), key=lambda x: all_ts.index(x))

    print('Number of features: %d' % len(activity_names))
    print('Feature list: ', activity_names)
    print("\n")

    #obtain timestamp encoding for each case in the training log
    enc_si_ts_train = []
    for case_index, case in enumerate(train_log):
        encoded_row = [case.attributes["concept:name"]]
        enc_si_ts = encode_trace_simple_index_with_timestamp(case, activity_names, all_ts)
        encoded_row.extend(np.reshape(enc_si_ts, -1))
        encoded_row.append(case.attributes["label"])
        enc_si_ts_train.append(encoded_row)

    #obtain timestamp encoding for each case in the test log
    enc_si_ts_test = []
    for case_index, case in enumerate(test_log):
        encoded_row = [case.attributes["concept:name"]]
        enc_si_ts = encode_trace_simple_index_with_timestamp(case, activity_names, all_ts)
        encoded_row.extend(np.reshape(enc_si_ts, -1))
        encoded_row.append(case.attributes["label"])
        enc_si_ts_test.append(encoded_row)

    print("First case with timestamp informations: \n", enc_si_ts_train[0])

    #create training dataframe with timestamp encoding
    df_train = pd.DataFrame(data=enc_si_ts_train, columns=compute_columns())
    print(df_train.head())
    df_train.to_csv('train_si_ts.csv')

    #create test dataframe with timestamp encoding
    df_test = pd.DataFrame(data=enc_si_ts_test, columns=compute_columns())
    print(df_test.head())
    df_test.to_csv('test_si_ts.csv')

    return df_train, df_test

#function that returns 8 temporal features extract from the timestamp of a specific event
def encode_timestamp(ts):
    # timestamp = event['time:timestamp']
     vector = [int(ts.timestamp()),str(ts),ts.day,dt.weekday(ts),
               ts.month,ts.hour,ts.minute,
               int(str(ts.hour)+str(ts.minute))]

     return vector

#function that combines name of the event with timestamp vector encoding returned by the encode_timestamp function
def encode_event_simple_index_with_timestamp(event):
    event_name = event['concept:name']
    ts = event['time:timestamp']
    return [event_name] + encode_timestamp(ts)

#function that returns vector encoding the trace using the simple index with timestamp encoding for a specific execution trace
def encode_trace_simple_index_with_timestamp(trace: Trace, all_activities, all_timestamps, prefix_length=20):
   enc_si_ts_case = [[str(0)]+[0]+[str(0)]+[0]*6]*prefix_length
   for event_idx, event in enumerate(trace):
      if event_idx < prefix_length:
         event_si_ts = encode_event_simple_index_with_timestamp(event)
         event_name = [event_si_ts[0]]
         time_stam = [event_si_ts[2]]

         le1 = LabelEncoder()
         le1.fit([str(0)] + all_activities)
         enc_e = le1.transform(event_name)

         le2 = LabelEncoder()
         le2.fit([str(0)] + all_timestamps)
         enc_t = le2.transform(time_stam)
         event_si_ts = list(enc_e) + [event_si_ts[1]] + list(enc_t) + event_si_ts[3:]
         enc_si_ts_case[event_idx] = event_si_ts
      else:
        break;
   return enc_si_ts_case

#obtain list of all unique timestamp
def get_ts_list(log: lg.EventLog):
    ts_l=[]
    for case_index,case in enumerate(log):
        for event_index,event in enumerate(case):
            ts_l.append(str(event['time:timestamp']))
    return sorted(set(ts_l), key=lambda x:ts_l.index(x))

#function that returns all the columns of the timestamp encoded dataframe
def compute_columns(prefix_length=20):
    columns = ['trace_id']
    for i in range(prefix_length):

        columns.append("event_"+str(i+1))
        columns.append("ets_" + str(i + 1))
        columns.append("ts_" + str(i + 1))
        columns.append("dy_" + str(i + 1))
        columns.append("dw_" + str(i + 1))
        columns.append("mon_" + str(i + 1))
        columns.append("h_" + str(i + 1))
        columns.append("min_" + str(i + 1))
        columns.append("conc_" + str(i + 1))

    columns.append('label')
    return columns

if __name__== "__main__":
  df_train,df_test=main_ts()