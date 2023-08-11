from abc import ABC, abstractmethod
import pandas as pd
from vertexgenaieval.classes import models

class Data(ABC):
  
  def __init__(self, dataloc:str, prefix_question:str, context_col: str, ground_truth_col: str, llm_model:models):
    self.model_response_col = "model_response"
    self.set_data(dataloc, prefix_question, context_col, ground_truth_col, llm_model)

  def set_data(self, dataloc, prefix_question, context_col, ground_truth_col, llm_model):
    
    if (".csv" in dataloc):
      df = pd.read_csv(dataloc, header=0, quotechar='"')
    elif (".json" in dataloc):
      df = pd.read_json(dataloc, lines=True)
    else:
      print("ERROR: file should be .csv or .json/.jsonl")
      exit()
    
    df = df[[context_col, ground_truth_col]]
    df = df.rename(columns={context_col: "prompt", ground_truth_col: "ground_truth"})
    df['prompt'] = df['prompt'].apply(lambda x: prefix_question + "\n" + x)
    self.df = df
    self.generate_model_responses(llm_model)

  def generate_model_responses(self, llm_model):
    self.df[self.model_response_col] = self.df["prompt"].apply(lambda x: llm_model.generate_response(x).text)  


