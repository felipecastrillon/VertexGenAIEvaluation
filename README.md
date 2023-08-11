# VertexGenAIEvaluation

The purpose of this package is to create some utilities that make it easy to perform an evaluation task with Google Cloud VertexAI models. 

### Requirements 

The following requirements are necessary to run this project
- A Google Cloud Platform (GCP) project
- User access to VertexAI
- Authentication to GCP, (see above section to auth via service account)
- A "golden dataset" which with a row for the prompt context and another row with the "ground truth" 

### How to Run

First, we need to authenticate via the terminal. One way to authenticate to GCP is to create a servie account key via https://cloud.google.com/iam/docs/keys-create-delete and download it. Then run the command:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=<file_location>
```

Now let's create a data object. This object assumes that your file 

```python
# This is a Python code block
print("Hello, world!")
```

### Classes
