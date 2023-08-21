from vertexgenaieval.classes import models
from vertexgenaieval.classes import data
from abc import ABC, abstractmethod
from vertexai.preview.language_models import TextEmbeddingModel
import numpy as np
import sklearn
from sklearn.metrics import f1_score
import pdb
from google.cloud import language_v1
from nltk.translate.bleu_score import sentence_bleu
import tenacity
from tenacity import * 


class Evaluator(ABC):
  
  def __init__(self, data:data.Data):
    self.set_data(data)    

  def set_data(self, data):
    self.data = data

  def evaluation_job(self):
    self.data.df["score"] = self.data.df.apply(self.evaluate_row, axis =1) 
    mean_score = self.data.df["score"].mean()
    return mean_score

  @abstractmethod
  def evaluate_row(self):
    pass	


class SemanticSimilarityEvaluator(Evaluator):

  def __init__(self, data:data.Data):
    self.set_data(data)    
    self.emb_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
 
  @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
  def evaluate_row(self, row):
    if row["model_response"] == '':
      return 0
    
    try:
      embedding_response = self.emb_model.get_embeddings([row["ground_truth"], row["model_response"]])
    except Exception as e: 
      print(e)
      raise

    embeddings = [embedding.values for embedding in embedding_response]    
    return np.dot(embeddings[0], embeddings[1])


class ExactMatchEvaluator(Evaluator):

  def evaluate_row(self, row):
    if str(row["ground_truth"]) == str(row["model_response"]):
      return 1
    else:
      return 0	

class SentimentEvaluator(Evaluator):
  
  def __init__(self, data:data.Data):
    self.set_data(data)    
    self.nlp_model = language_v1.LanguageServiceClient()
 
  def sample_analyze_sentiment(self, content):

    if isinstance(content, bytes):
        content = content.decode("utf-8")

    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {"type_": type_, "content": content}

    response = self.nlp_model.analyze_sentiment(request={"document": document})
    sentiment = response.document_sentiment.score
    if sentiment >= 0.0:
       return "positive"
    else:
       return "negative"

  def evaluate_row(self, row):
    if row["model_response"] == '':
      return 0
    if (self.sample_analyze_sentiment(row["ground_truth"]) == self.sample_analyze_sentiment(row["model_response"])):
      return 1
    else:
      return 0	


class BleuEvaluator(Evaluator):
  
  def evaluate_row(self, row):
    reference = [row["ground_truth"].split()]
    candidate = row["model_response"].split() 
    return sentence_bleu(reference, candidate, weights=(1,0,0,0))


