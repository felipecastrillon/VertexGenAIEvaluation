from abc import ABC, abstractmethod
from vertexai.language_models import TextGenerationModel
from enum import Enum
from typing import Any
from typing import Dict
import os

class Model(ABC):
  
  @abstractmethod
  def __init__(self, parameters:Dict[str, Any]):
    self.set_model()
    self.set_parameters(parameters) 

  @abstractmethod
  def set_model(self):
    pass

  def set_parameters(self, parameters):
    self.parameters = {
      "temperature" : parameters["temperature"],
      "top_k" : parameters["top_k"],
      "top_p" : parameters["top_p"],
      "max_output_tokens" : parameters["max_output_tokens"] 
    }

  def generate_response(self, prompt):
    return self.model.predict(
        prompt,
        **self.parameters,
      ) 

class PalmBisonModel(Model):

  def __init__(self, parameters:Dict[str, Any]):
    super().__init__(parameters)

  def set_model(self):
    self.model=TextGenerationModel.from_pretrained("text-bison@001") 

class PalmBisonTunedModel(Model):
  
  def __init__(self, endpoint_path:str, parameters:Dict[str,Any]):
    set_model(endpoint_path)
    set_parameters(parameters) 

  def set_model(self, endpoint_path):
    self.model = TextGenerationModel.get_tuned_model(endpoint_path)

  

