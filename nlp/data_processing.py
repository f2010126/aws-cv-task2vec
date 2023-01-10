import os
import pandas as pd
import json

class LoadingData():
    def __init__(self):
        print(f"where? {os.getcwd()}")
        train_file_path = os.path.join("./data","benchmarking_data", "Train")
        validation_file_path = os.path.join("./data","benchmarking_data", "Validate")
        category_id = 0
        self.cat_to_intent = {}
        self.intent_to_cat = {}

        for dirname, _, filenames in os.walk(train_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                self.cat_to_intent[category_id] = intent_id
                self.intent_to_cat[intent_id] = category_id
                category_id += 1
        print(self.cat_to_intent)
        print(self.intent_to_cat)
        '''Training data'''
        training_data = list()
        for dirname, _, filenames in os.walk(train_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                training_data += self.make_data_for_intent_from_json(file_path, intent_id,
                                                                     self.intent_to_cat[intent_id])
        self.train_data_frame = pd.DataFrame(training_data, columns=['query', 'intent', 'category'])
        self.train_data_frame = self.train_data_frame.sample(frac=1)

        '''Validation data'''
        validation_data = list()
        for dirname, _, filenames in os.walk(validation_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                validation_data += self.make_data_for_intent_from_json(file_path, intent_id,
                                                                       self.intent_to_cat[intent_id])
        self.validation_data_frame = pd.DataFrame(validation_data, columns=['query', 'intent', 'category'])

        self.validation_data_frame = self.validation_data_frame.sample(frac=1)

    def make_data_for_intent_from_json(self, json_file, intent_id, cat):
        json_d = json.load(open(json_file))

        json_dict = json_d[intent_id]

        sent_list = list()
        for i in json_dict:
            each_list = i['data']
            sent = ""
            for i in each_list:
                sent = sent + i['text'] + " "
            sent = sent[:-1]
            for i in range(3):
                sent = sent.replace("  ", " ")
            sent_list.append((sent, intent_id, cat))
        return sent_list
