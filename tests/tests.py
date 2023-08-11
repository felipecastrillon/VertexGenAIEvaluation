import sys
sys.path.append('../src')
from vertexgenaieval.classes import models 
from vertexgenaieval.classes import data 
from vertexgenaieval.classes import evaluators 
import pytest

def test_simple():
  assert 1 == 1

def test_with_message():
  assert 1 == 1, "This test should pass"

def test_palm_bison_model():
  model_instance = models.PalmBisonModel(parameters = {"temperature":0.0,
                                                "top_k":5,
                                                "top_p":0.8,
                                                "max_output_tokens":1000}  )
  response = model_instance.generate_response(prompt="hello how are you?") 
  assert(len(response.text)>0)

@pytest.mark.skip
def test_data_class():
  prefix = "summarize the following article: "
  model_instance =  models.PalmBisonModel(parameters = {"temperature":0.2,
                                                "top_k":40,
                                                "top_p":0.8,
                                                "max_output_tokens":1000}  )
  data_instance = data.Data("summarization_sample.jsonl", prefix, "article", "summary", model_instance)
  print(data_instance.df.iloc[0])
  assert (data_instance.df.iloc[0, 0].startswith(prefix))


def test_summary_evaluator():
  prefix = "summarize the following article: "
  model_instance =  models.PalmBisonModel(parameters = {"temperature":0.0,
                                                "top_k":5,
                                                "top_p":0.8,
                                                "max_output_tokens":1000}  )
  data_instance = data.Data("summarization_sample.jsonl", prefix, "article", "summary", model_instance)
  evaluator_instance = evaluators.SemanticSimilarityEvaluator(data_instance)
  mean_score = evaluator_instance.evaluation_job()
  print(mean_score) 
  evaluator_bleu_instance = evaluators.BleuEvaluator(data_instance)
  mean_bleu_score = evaluator_bleu_instance.evaluation_job()
  print(mean_bleu_score)
  assert mean_score > 0.80  
  assert (mean_bleu_score > 0.2)

@pytest.mark.skip
def test_classification_evaluator():
  prefix = "categorize the following text based on four categories: \
              Books \
              Household \
              Electronics \
              Clothing & Accessories \
              \
              Only output the category from the list above. \
              Do not create any new categories"
  model_instance =  models.PalmBisonModel(parameters = {"temperature":0.0,
                                                "top_k":5,
                                                "top_p":0.8,
                                                "max_output_tokens":5}  )
  data_instance = data.Data("classification_sample.csv", prefix, "Description", "Category", model_instance)
  evaluator_instance = evaluators.ExactMatchEvaluator(data_instance)
  mean_score = evaluator_instance.evaluation_job()
  assert mean_score > 0.50  

@pytest.mark.skip
def test_extraction_evaluator():
  prefix = "Make sure you state Yes/No somewhere in your answer. See context and question below: \n\n"
  model_instance =  models.PalmBisonModel(parameters = {"temperature":0.8,
                                                "top_k":5,
                                                "top_p":0.8,
                                                "max_output_tokens":100}  )
  data_instance = data.Data("extraction_sample.csv", prefix, "Question", "Answer", model_instance)
  evaluator_instance = evaluators.SentimentEvaluator(data_instance)
  mean_score = evaluator_instance.evaluation_job()
  assert mean_score > 0.50  

 
