
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Page configuration
st.set_page_config(
    page_title="BERT Question Answering System",
    page_icon="📚",
    layout="wide"
)

MODEL_DIR = "models/bert_qa"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


st.title("📚 BERT Question Answering System")
st.write(
    "Ask a question about a passage of text. "
    "The fine-tuned BERT model will extract the answer from the passage."
)

try:
    tokenizer, model, device = load_model()
    st.caption(f"Model loaded successfully | Device: {device}")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

context = st.text_area(
    "Enter the context passage",
    placeholder="Paste a paragraph here...",
    height=220
)

question = st.text_input(
    "Enter your question",
    placeholder="What would you like to know?"
)

if st.button("Find Answer", type="primary"):
    if not context.strip() or not question.strip():
        st.warning("Please enter both a context passage and a question.")
    else:
        try:
            inputs = tokenizer(
                question,
                context,
                max_length=384,
                truncation="only_second",
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            offsets = inputs.pop("offset_mapping")[0].tolist()
            sequence_ids = inputs.sequence_ids(0)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            best_score = float("-inf")
            best_start = 0
            best_end = 0
            max_answer_length = 30

            for start_idx in range(len(offsets)):
                if sequence_ids[start_idx] != 1:
                    continue

                end_limit = min(
                    len(offsets),
                    start_idx + max_answer_length
                )

                for end_idx in range(start_idx, end_limit):
                    if sequence_ids[end_idx] != 1:
                        break

                    score = (
                        start_logits[start_idx].item()
                        + end_logits[end_idx].item()
                    )

                    if score > best_score:
                        best_score = score
                        best_start = start_idx
                        best_end = end_idx

            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            answer = context[start_char:end_char].strip()

            if answer:
                st.subheader("Predicted Answer")
                st.success(answer)
            else:
                st.warning(
                    "No answer span was found. Try a different question "
                    "or provide a clearer passage."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")