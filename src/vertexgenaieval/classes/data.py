import pandas as pd
from vertexgenaieval.classes import models
import tenacity
from tenacity import * 
import pdb

class Data():
  
  def __init__(self, dataloc:str, prefix_question:str, context_col: str, ground_truth_col: str):
    self.model_response_col = "model_response"
    self.set_data(dataloc, prefix_question, context_col, ground_truth_col)

  def set_data(self, dataloc, prefix_question, context_col, ground_truth_col):
    
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
  
  def generate_model_responses(self, llm_model):
    self.df[self.model_response_col] = self.df["prompt"].apply(lambda x: self.generate_response(x,llm_model).text)  
  


  @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
  def generate_response(self, prompt, model):
    try:
      answer=model.generate_response(prompt)
      return answer 
    except Exception as e:
      print(e)
      raise

    
