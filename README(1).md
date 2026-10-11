# BERT-Based Question Answering System

## 1. Project Overview

This project is a simple Question Answering System built with a
pretrained BERT model. It reads a passage of text and answers a question
by finding the most relevant part of that passage.

For example, if the passage says that the Pragyan rover explored the
lunar surface and the question is "Which rover explored the lunar
surface?", the system should return "Pragyan rover".

The project includes a Jupyter notebook for preparing the data, training
and evaluating the model, and a Streamlit web app that lets a user try
the system in a browser.

## 2. Problem Statement

Finding a specific fact inside a long paragraph can take time. A
Question Answering System aims to make this easier by allowing a person
to ask a question in normal language and receive a short answer based on
the supplied text.

The goal of this project is to build a working prototype that can
identify answer spans from a passage. It is an extractive question
answering system: it selects text from the passage rather than writing a
completely new answer.

## 3. Objectives

-   Understand how a pretrained BERT model can be used for question
    answering.
-   Prepare question-and-answer examples from the SQuAD dataset.
-   Fine-tune the model to predict the start and end of an answer in a
    passage.
-   Evaluate predictions using Exact Match and Token F1.
-   Build a simple Streamlit interface for testing questions.
-   Save the trained model, evaluation metrics, and sample predictions.

## 4. Technologies Used

-   **Python:** Main programming language.
-   **JupyterLab / Jupyter Notebook:** Used for data preparation,
    training, and evaluation.
-   **PyTorch:** Used to train and run the model.
-   **Hugging Face Transformers:** Used to load the pretrained BERT
    tokenizer and question-answering model.
-   **SQuAD 1.1:** Dataset containing passages, questions, and answers.
-   **Streamlit:** Used to create the browser-based application.
-   **CUDA-enabled GPU:** Used to speed up model training when
    available.

## 5. Dataset

This project uses the Stanford Question Answering Dataset (SQuAD 1.1).
It contains paragraphs from various topics, questions about those
paragraphs, and answer text with its location in the paragraph.

The project files used are:

-   `dataset/squad/train-v1.1.json`
-   `dataset/squad/dev-v1.1.json`

The training file contains 87,599 extracted question-answer examples in
the current project setup. For the expanded training run, the first
5,000 examples were selected. Because long passages may be split into
overlapping token windows, those examples produced 5,090 training
features.

The development file was used to create a 200-example evaluation subset.
Therefore, the reported evaluation scores apply to this subset only, not
to the full SQuAD development set.

## 6. How the System Works

The project follows these main steps:

1.  **Load the dataset:** Read the SQuAD JSON files and extract each
    question, passage, and reference answer.
2.  **Tokenize the text:** Convert the question and passage into tokens
    that BERT can process.
3.  **Prepare answer positions:** Map the correct answer's character
    position to token start and end positions. Longer passages may
    create more than one feature window.
4.  **Fine-tune BERT:** Train the question-answering model on prepared
    examples so it learns to identify answer spans.
5.  **Evaluate the model:** Compare predicted answers with the reference
    answers using Exact Match and Token F1.
6.  **Save the model:** Store the model and tokenizer in
    `models/bert_qa/`.
7.  **Run the web app:** The Streamlit app accepts a passage and a
    question, runs the model, and displays the selected answer.

## 7. Model Details

The project uses `bert-base-uncased` as the starting model with a
question-answering output head. "Uncased" means the tokenizer treats
uppercase and lowercase text in a normalized way.

For the expanded training run, the model was trained on features
prepared from 5,000 SQuAD training examples. The training run completed
with an average training loss of **1.9592**. Training loss shows how
well the model is fitting the training examples; by itself, it does not
prove that the model answers unseen questions accurately.

The saved model files are stored in:

`models/bert_qa/`

This folder contains the model configuration, model weights, and
tokenizer files.

## 8. Evaluation Method

Two standard metrics were used:

-   **Exact Match (EM):** Counts a prediction as correct when its
    normalized text exactly matches the reference answer.
-   **Token F1:** Measures the overlap between the words in the
    predicted answer and the reference answer. It can give partial
    credit when the prediction contains some, but not all, of the
    correct answer words.

### Current Results

The updated model was evaluated on 200 examples from the development
subset.

  Metric                  Result
  --------------------- --------
  Evaluation examples        200
  Exact Match             60.50%
  Token F1                67.74%

An earlier evaluation on the same 200 examples, before the expanded
training run, recorded 19.50% Exact Match and 23.80% Token F1.

These results suggest that the additional training helped on this
subset. However, the evaluation is limited to 200 examples and should be
treated as a preliminary project result, not as a full SQuAD benchmark
score.

## 9. Manual Testing

The Streamlit application was tested with a short paragraph about the
Chandrayaan-3 mission.

  -----------------------------------------------------------------------
  Question                Observed answer         Test outcome
  ----------------------- ----------------------- -----------------------
  Where was Chandrayaan-3 Sriharikota, Andhra     Relevant location
  launched from?          Pradesh                 returned, but the
                                                  answer omitted the
                                                  launch centre's name

  What was the rover's    Pragyan rover           Correct
  name?                                           

  On what date did the    23 August 2023          Correct
  Vikram lander touch                             
  down near the Moon's                            
  south pole?                                     
  -----------------------------------------------------------------------

Manual tests are useful for checking whether the app works, but a few
examples are not enough to measure overall model performance.

## 10. Project Folder Structure

The main files are organized as follows:

``` text
Question_Answering_WTP/
├── dataset/
│   └── squad/
│       ├── train-v1.1.json
│       └── dev-v1.1.json
├── models/
│   └── bert_qa/
├── results/
│   ├── evaluation_metrics.json
│   └── predictions.json
├── app/
│   └── app.py
└── Question_Answering.ipynb
```

The `models/bert_qa/` folder contains the saved model and tokenizer
files. The `results/` folder contains the evaluation metrics and
predictions. The notebook contains the data preparation, training, and
evaluation work, while `app/app.py` contains the Streamlit interface.

## 11. How to Run the Project

These instructions assume that Python and the project's existing virtual
environment are already set up on Windows.

### Step 1: Open the project folder

Open a terminal in the `Question_Answering_WTP` project root. Activate
the existing `.venv` if it is not already active.

### Step 2: Check that the required packages are available

The environment needs PyTorch, Transformers, and Streamlit. The app
should be run using the same Python environment in which the compatible
project dependencies are installed.

### Step 3: Start the Streamlit app

From the project root, run:

``` powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Streamlit prints a local URL in the terminal, usually
`http://localhost:8501`. Open that URL in a browser.

### Step 4: Ask a question

1.  Paste a paragraph into the context field.
2.  Enter a question whose answer is stated in that paragraph.
3.  Click **Find Answer**.
4.  Read the answer displayed by the application.

The model works best when the answer is clearly stated in the supplied
passage. Very long passages may be truncated by the current app
configuration, so keep test passages reasonably short.

### Step 5: Stop the app

When finished, return to the terminal running Streamlit and press
`Ctrl + C`.

## 12. Limitations

-   The model was fine-tuned on a subset of the available training
    examples rather than the full training set.
-   The reported evaluation uses only 200 development examples, so it
    may not represent performance across the entire dataset.
-   The system extracts a span of text; it does not generate
    explanations or combine facts from different parts of a passage.
-   It may return an answer that is related to the question but not
    precise enough.
-   The current app is designed for a single supplied passage. It is not
    a search engine over a large collection of documents.
-   The model's answer depends on the quality and clarity of the input
    passage.

## 13. Future Improvements

-   Train on more SQuAD training examples and compare the results
    carefully.
-   Evaluate on a larger development subset or the full development set
    when resources allow.
-   Improve handling of long passages by processing multiple windows and
    comparing candidate answers.
-   Add more sample questions and automated tests.
-   Improve the user interface with example passages, clearer error
    messages, and an option to show the answer in context.
-   Record experiment settings and results so future training runs can
    be compared fairly.

## 14. Conclusion

This project demonstrates how a pretrained BERT model can be adapted for
extractive question answering. It combines dataset preparation, model
fine-tuning, evaluation, and a Streamlit interface in one workflow.

The current model achieved 60.50% Exact Match and 67.74% Token F1 on a
200-example development subset. It also answered several manual test
questions correctly, although some answers were incomplete. The results
show that the prototype is working, while also highlighting
opportunities to improve training coverage, evaluation, and answer
selection.

The project provided practical experience with transformer models,
token-level answer prediction, evaluation metrics, GPU-based training,
and building a small machine learning web application.

## 15. References

-   Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine
    Comprehension of Text*. https://arxiv.org/abs/1606.05250
-   Hugging Face Transformers documentation:
    https://huggingface.co/docs/transformers/
-   Streamlit documentation: https://docs.streamlit.io/
-   PyTorch documentation: https://pytorch.org/docs/
