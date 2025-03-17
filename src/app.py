# Create the Streamlit app file
import os
import streamlit as st
from vllm import LLM, SamplingParams
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Set page configuration
st.set_page_config(
    page_title="Mental Health Counselor",
    page_icon="🧠",
    layout="wide"
)

model_config = {
    "Mistral-7B": "mlabonne/NeuralHermes-2.5-Mistral-7B",
    "Llama3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama3.2-1B-Instruct-finetuned": "/project/pi_hongyu_umass_edu/n_yarrabolu/personal/results/"
}

prompt_config = {
    "Mistral-7B": """You are a mental health counsellor who provides empathetic and supportive responses to patients based on their mental health conditions.
Your role is to help the patient feel heard, understood, and supported.
Do not provide medical advice; instead, focus on offering emotional support and validation.

The patient’s mental health condition will be provided in quotes below. Use this information to tailor your response:

{patient_situation}

Respond to the patient in a compassionate and professional manner, thinking as if you are a mental health counsellor. encouraging them to share more if they feel comfortable, and suggesting general self-care practices if appropriate.
Your goal is to make the patient feel positive and supported. 

""",
    "Llama3.2-1B-Instruct": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You are a mental health counsellor who provides empathetic and supportive responses
    
<|eot_id|><|start_header_id|>user<|end_header_id|>
    
{patient_situation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
""",
    "Llama3.2-1B-Instruct-finetuned": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You are a mental health counsellor who provides empathetic and supportive responses
    
<|eot_id|><|start_header_id|>user<|end_header_id|>
    
{patient_situation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
"""
}

if not os.path.exists(model_config["Llama3.2-1B-Instruct-finetuned"]):
    model_config.pop("Llama3.2-1B-Instruct-finetuned")



@st.cache_resource
def load_model(model_name):
    st.cache_data.clear()
    with torch.no_grad():
        torch.cuda.empty_cache()
    model_path = model_config[model_name]
    try:
        return LLM(
            model= model_path,
            # tensor_parallel_size=1,
            trust_remote_code=True,
            download_dir="/tmp/model_cache",
        )
    except Exception as e:
        st.error(f"Error loading local model: {e}")
        return None
    

def generate_counselor_guidance(model_name, patient_situation):
    model = load_model(model_name=model_name)
    
    if model is None:
        return "Model failed to load. Please check your installation and try again."
    
    full_prompt = prompt_config[model_name].format(patient_situation=patient_situation)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )
    
    try:
        outputs = model.generate([full_prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


st.title("🧠 Mental Health Counselor")
st.subheader("Talk with your personal mental health counselor")

# App description
st.markdown("""
This is a confidential space where you can discuss your thoughts, feelings, and concerns.
I'm here to listen and provide support based on therapeutic principles.
""")

# Input form
with st.form("counselor_form"):
    model_name = st.selectbox('Model to use', model_config.keys())
    patient_situation = st.text_area(
        "Describe your situation:",
        height=150,
        placeholder="Include relevant details about symptoms, and current challenges."
    ) 
    submit_button = st.form_submit_button("Get Guidance")


# Generating and displaying response
if submit_button and patient_situation:
    with st.spinner("Generating guidance..."):
        response = generate_counselor_guidance(model_name, patient_situation)
        
    st.subheader("Therapeutic Guidance")
    st.markdown(response)
    
    # Disclaimer
    st.info("""
    **Disclaimer:** This tool provides suggestions based on AI analysis and should not replace professional judgment. 
    Always use your clinical expertise and follow ethical guidelines when working with patients.
    """)
else:
    if submit_button and not patient_situation:
        st.warning("Please provide information about the patient situation.")

# for Sidebar with additional information
with st.sidebar:
    st.header("About this tool")
    st.markdown("""
    This application uses an AI model specially trained to provide mental health support and guidance.
    
    **Remember:**
    - This is not a replacement for professional therapy
    - If you're experiencing a crisis, please contact emergency services
    - Your conversations are processed locally and not stored permanently
    """)