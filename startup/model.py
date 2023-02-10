import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class AnimeRecommender():
    index = faiss.read_index('animes.index')
    data = pd.read_csv('processed.csv', sep='/')
    model = SentenceTransformer('model/')
    
    query = ""
    top_k = 0
    results = list()

    def fetch_anime(self, dataframe_idx):
        info = self.data.iloc[dataframe_idx]
        meta = dict()
        meta['title'] = info['title']
        return meta
    
    def search(self):
        query_vector = self.model.encode([self.query])
        self.top_k = self.index.search(query_vector, self.top_k)
        result_id = self.top_k[1].tolist()[0]
        result_id = list(np.unique(result_id))
        results =  [self.fetch_anime(idx) for idx in result_id]
        return results
    
    def recommend(self, query, top_k = 5):
        self.top_k = top_k
        self.query = query
        
        self.results = self.search()
        
        print("Query : ", self.query)
        print("Result : ")
        for result  in self.results:
            print("\t", result)
    
if __name__ == '__main__':
    recommender = AnimeRecommender()
    recommender.recommend(query = "Anime about isekai using sword", top_k = 5)