# VertexGenAIEvaluation

This package provides utilities that facilite evaluation tasks with Google Cloud PaLM LLM models. Ideally you will only need a few lines of code to perform a task.

&nbsp;&nbsp;
## Requirements ##

* A Google Cloud Platform (GCP) project
* IAM user access to VertexAI
* Authentication to GCP (see below section to auth via service account)
* A "golden dataset" containing a row with the prompt context and another row with "ground truth" answers
* If evaluating a tuned model, [the tuned model should already be created and deployed](https://cloud.google.com/vertex-ai/docs/generative-ai/models/tune-models). 

&nbsp;&nbsp;
## How to Run ##

First, we need to authenticate via the terminal. One way to authenticate to GCP is to [create a service account](https://cloud.google.com/iam/docs/keys-create-delete) and download the key file. Then, you can run the terminal command:

&nbsp;
```bash
export GOOGLE_APPLICATION_CREDENTIALS=<service_account_file_location>
```
&nbsp;

Now, we can install the package. Open up a python environment (i.e. a notebook or a virtual environment), clone the repo and pip install it:

&nbsp;
``` bash
git clone https://github.com/felipecastrillon/VertexGenAIEvaluation.git
cd VertexGenAIEvaluation
pip install dist/vertexgenaieval-1.0-py3-none-any.whl
```
&nbsp;

To begin using the package let's instantiate a model object. there are two types of models that can be created: a vanilla Palm Text model (not tuned) and a tuned Palm Text model. For the latter you will have to [tune the model on the console or via the API/SDK](https://cloud.google.com/vertex-ai/docs/generative-ai/models/tune-models) before instantiation. 

&nbsp; 
```python
from vertexgenaieval.classes import models

model_instance =  models.PalmBisonModel(parameters = {"temperature":0.0,
                                                "top_k":5,
                                                "top_p":0.8,
                                                "max_output_tokens":1000}  )
# or...
tuned_model_instance = models.PalmBisonModel(parameters = {"temperature":0.0,
                                                "top_k":5,
                                                "top_p":0.8,
                                                "max_output_tokens":1000},
                                             endpoint_path=<MODEL_ENDPOINT>  )
```
&nbsp;

Now let's read the data file and populate generated answers for each row. The prompt for each row will be created by concatenating the prefix question and the value from the context_col, the answers will be generated by passing each prompts from each row to the model object. 

&nbsp;
```python
from vertexgenaieval.classes import data

prefix = "summarize the following article: "
data_instance = data.Data(dataloc="path/to/summarization_sample.jsonl", prefix_question=prefix, context_col="article", ground_truth_col="summary", llm_model=model_instance)

```
&nbsp;

Finally let's do an evaluation task comparing the generated response vs the "ground truth" response and get the mean score of the task:

&nbsp;
```python
from vertexgenaieval.classes import evaluators

evaluator_instance = evaluators.SemanticSimilarityEvaluator(data=data_instance)
mean_score = evaluator_instance.evaluation_job()

```
&nbsp;&nbsp;
## Evaluation Metrics ##


This package contains 4 evaluator classes to compare the "generated response" vs the "ground truth":

- **SemanticSimilarityEvaluator()** - This metric is useful when determining how similar the two texts are on a semantic basis. This will give a score between 0 and 1. This is a great metric for  evaluating summarization or content retrieval tasks.
- **BleuEvaluator()** - The [BLEU metric](https://en.wikipedia.org/wiki/BLEU) can help evaluate the token similarity between two texts. In this implementation we are only using the 1-gram BLEU evaluator. This metric could be used for summarization of content retrieval tasks. 
- **ExactMatchEvaluator()** - This metric evalutes an exact match between the two texts. This is useful for classification tasks or tasks expecting exact answers. 
- **SentimentEvaluator()** - Evaluate sentiment tasks or Yes/No tasks. 

&nbsp; &nbsp;
## Some Ideas on Package Extension ##


- Integration with Langchain?
- Ability to create a tuned model with some of the input data before evaluating the model
- Add Enterprise Search as a model
- Add 3rd party models
- Add more Evalutors
- Import and export data from BQ/GCS
- Run evaluators on Dataflow for better scale
