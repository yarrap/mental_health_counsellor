# mental_health_counsellor
Web application used for creating a mental health consellor

The idea is to implement a web application which involves users entering free-text that describes the challenge they are
facing, using the user input we need to invoke an LLM and return a suggestion to the user on
how to best help him with his condition. I was planning to build a mental health counsellor which
can help a patient using the above methodology.


The steps involved Fine-tuning a LLM using Kaggle dataset and building a web application
based on 3 different models like mistral-7B(mlabonne/NeuralHermes-2.5-Mistral-7B) which is
already pretrained on medical dataset and this model is available from hugging face, llama 3.2
1B instruct model and Llama 3.2 1b instruct finetuned model. The idea is to give the option to
the user to select one of the 3 models and generate an advice based on his condition.


The steps involved:
1. Step1 (Data Processing):
We are processing the data and removing all the duplicate entries and null values from
the dataset, and did a clear analysis on word and token distribution across columns.
Also, I checked if different languages are present in the columns, only english was taken
and others were discarded.

3. Step2 (Fine-tuning Llama 3.2 1B Instruct Model):
In this step, we are using the processed data set and using the tokenizer as the llama
model itself. We are generating the tokenized input for train and test samples. We are
then defining our training parameters and training our model based on the defined
parameters.

4. Step3 (Building Web Application):
This is the final step which involves building a web application. So, in this step we are
using streamlit to build our application. This involves working on the UI design which
makes sure that the application looks visually promising. Once the user inputs a problem
they are facing we will use the user input and infer our model(one of the three which the
user selects) then we will give a response.


Instructions to run:

To start the web application you need to run the following commands from src folder
Run the below command for fine-tuning
```
python3 main.py
```
Run the below command for running the web application
```
streamlit run app.py
```
