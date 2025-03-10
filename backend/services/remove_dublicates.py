import os
import pickle
import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta




class MessageManager:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', index_path='faiss.index', data_path='data.pkl'):
        self.model = SentenceTransformer(model_name)
        self.messages = []
        self.timestamps = [] 
        self.embeddings = np.empty((0, self.model.get_sentence_embedding_dimension()), dtype='float32')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index_path = index_path
        self.data_path = data_path
        self.index = faiss.IndexFlatL2(self.dimension)
        self.load_data()
    
    def add_new_message(self, new_message , current_time):
        new_embedding = self.model.encode([new_message]).astype('float32')
        self.index.add(new_embedding)
        self.messages.append(new_message)
        self.timestamps.append(current_time)
        self.embeddings = np.vstack([self.embeddings, new_embedding])
    
    def is_similar(self, new_message, threshold=0.8):
        if len(self.messages) == 0:
            return False, None

        new_embedding = self.model.encode([new_message]).astype('float32')
        D, I = self.index.search(new_embedding, k=1)
        nearest_index = I[0][0]

        if nearest_index == -1:
            return None, None
        # get the date and content of the nearest message
        nearest_message = self.messages[nearest_index]
        nearest_timestamp = self.timestamps[nearest_index]  

        existing_embedding = self.embeddings[nearest_index]
        similarity = np.dot(new_embedding, existing_embedding) / (np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding))
        similar = similarity > threshold

        return (nearest_message, nearest_timestamp), similar[0]
    
    def is_similar_in_time_range(self, new_message, new_timestamp, day_range=1, threshold=0.8):
        if len(self.messages) == 0:
            print("база пуста - запис")
            return False
        
        new_dt = datetime.strptime(new_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        start_dt = new_dt - timedelta(days=day_range)
        
        valid_indices = [
            i for i, ts in enumerate(self.timestamps)
            if start_dt <= datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") <= new_dt
        ]

        if not valid_indices:
            print("подібних немає - запис")
            return False 

        filtered_embeddings = np.array([self.embeddings[i] for i in valid_indices])
        new_embedding = self.model.encode([new_message]).astype('float32')
        
        new_embedding = new_embedding.squeeze() 
        similarities = np.dot(filtered_embeddings, new_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(new_embedding)
        )

        if similarities.size == 0:
            print("подібних немає - запис")
            return False 

        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        if best_similarity > threshold:
                    print("подібні є  - не запис")
                    return  True
        else :
                    print("подібних немає  - запис")
                    return  False
        #     nearest_index = valid_indices[best_idx]
        #     return (self.messages[nearest_index], self.timestamps[nearest_index]), True


    def save_data(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, 'wb') as f:
            pickle.dump({'messages': self.messages, 'timestamps': self.timestamps, 'embeddings': self.embeddings}, f)
    
    def load_data(self):
        if os.path.exists(self.data_path) and os.path.exists(self.index_path):
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                self.messages = data.get('messages', [])
                self.timestamps = data.get('timestamps', [])  
                self.embeddings = data.get('embeddings', np.empty((0, self.dimension), dtype='float32'))
            self.index = faiss.read_index(self.index_path)

    def shutdown(self):
        self.save_data()





def rm_dublicates(manager , test_messages ,test_times ,test_ids ,day_range = 1):


     res_ids = []

     for i in range(len(test_messages)):
          new_msg = test_messages[i]
          new_timestamp = test_times[i]
          new_id= test_ids[i]

          result, is_similar = manager.is_similar(new_msg)
          if result == None:
               print(new_msg)
               print("the nearest neighbor is not found - write ")
               #the nearest neighbor is not found - write 
               manager.add_new_message(new_msg , new_timestamp)
               res_ids.append(new_id)
          elif result == False :
               print(new_msg)
               print("the database is empty - write")
               manager.add_new_message(new_msg , new_timestamp)
               res_ids.append(new_id)
               # the database is empty - write
          elif is_similar == False :
               print(new_msg)
               print("no similar  - write ")
               manager.add_new_message(new_msg , new_timestamp)
               res_ids.append(new_id)
               # no similar  - write 
          else: 
               print(new_msg)
               print("There are similar ones - check the time")

               # There are similar ones - check the time
            #    nearest_message, nearest_timestamp = result

            #    nearest_dt = datetime.strptime(nearest_timestamp, "%Y-%m-%d %H:%M:%S.%f")
            #    #end_dt time of the current message
            #    end_dt = datetime.strptime(new_timestamp, "%Y-%m-%d %H:%M:%S.%f")
            #    # start_dt the day_range before the current day
            #    start_dt = end_dt - timedelta(days=day_range)
            #    if not (start_dt <= nearest_dt <= end_dt) :
            #         # there are no similar ones in the specified time range - write 
            #         print("there are no similar ones in the specified time range - write ")
            #         manager.add_new_message(new_msg , new_timestamp)
            #         res_ids.append(new_id)
            #    else:
            #         print("similar record in the specified time range - don't write")
                 
     manager.shutdown()
     return res_ids


def rm_duplicates_time_range(manager, test_messages, test_times, test_ids, day_range=1):
    res_ids = []

    for new_msg, new_timestamp, new_id in zip(test_messages, test_times, test_ids):
        print("для new_msg " + new_msg + " " + str(new_timestamp))
        is_similar = manager.is_similar_in_time_range(new_msg, new_timestamp, day_range)
        if is_similar == False :
               manager.add_new_message(new_msg , new_timestamp)
               res_ids.append(new_id)

        else:
             print(f"{new_msg}\nСхожий запис у межах часу знайдено. Не записую.")
    
    manager.shutdown()
    return res_ids