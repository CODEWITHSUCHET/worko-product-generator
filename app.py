import os
import json
import sys
from typing import Optional, Dict

# 1. Install/Import Check
try:
    import streamlit as st
    from openai import OpenAI
    from pydantic import BaseModel
    from dotenv import load_dotenv
except ImportError:
    # Fallback if libraries are missing
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "openai", "pydantic", "python-dotenv"])
    import streamlit as st
    from openai import OpenAI
    from pydantic import BaseModel
    from dotenv import load_dotenv

# 2. Setup & Configuration
load_dotenv()

# Force page config to be the very first Streamlit command
st.set_page_config(page_title="AI Product Describer", layout="wide")

# Connect to Groq (using OpenAI client structure)
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("🚨 GROQ_API_KEY is missing! Please create a .env file with your key.")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# --- CLASS DEFINITIONS ---
class ProductInput(BaseModel):
    name: str
    category: str
    attributes: Dict[str, str]
    tone: Optional[str] = "Professional"

# --- ENGINE LOGIC (Generation) ---
def get_system_prompt(category: str, tone: str):
    base_prompts = {
        "Electronics": "You are a tech copywriter. Focus on specifications, performance, and compatibility.",
        "Fashion": "You are a fashion stylist. Focus on fabric feel, fit, and occasion.",
        "Home & Kitchen": "You are an interior designer. Focus on aesthetics, durability, and usage.",
        "General": "You are a professional copywriter. Focus on benefits and value proposition."
    }
    selected_prompt = base_prompts.get(category, base_prompts["General"])
    return f"{selected_prompt} Write in a {tone} tone. Do not invent features not listed."

def generate_description(product: ProductInput):
    specs_text = "\n".join([f"- {k}: {v}" for k, v in product.attributes.items()])
    
    user_message = f"""
    Product Name: {product.name}
    Category: {product.category}
    Features:
    {specs_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": get_system_prompt(product.category, product.tone)},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- EVALUATOR LOGIC (Quality Check) ---
def evaluate_quality(input_data: dict, generated_text: str):
    system_prompt = """
    You are a Quality Assurance AI. Compare the input data to the generated description.
    Return a valid JSON object with these keys:
    - "score": (integer 1-10)
    - "consistency_check": (string) "Pass" if all input facts are present, else "Fail".
    - "tone_feedback": (string) Brief comment on the tone.
    """

    user_message = f"""
    INPUT DATA: {json.dumps(input_data)}
    GENERATED TEXT: {generated_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": 0, "consistency_check": "Error", "tone_feedback": str(e)}

# --- UI LAYER (Streamlit) ---
st.title("🛍️ AI Product Description Generator")

with st.sidebar:
    st.header("Product Details")
    cat = st.selectbox("Category", ["Electronics", "Fashion", "Home & Kitchen", "Other"])
    p_name = st.text_input("Product Name", "Sony WH-1000XM5")
    
    st.subheader("Attributes")
    attr_input = st.text_area("Features (Format: Key: Value)", 
                              "Color: Midnight Blue\nNoise Cancellation: Active\nBattery: 30 Hours")
    
    tone = st.select_slider("Tone", options=["Professional", "Casual", "Luxury", "Witty"])
    
    generate_btn = st.button("✨ Generate Description", type="primary")

if generate_btn:
    # Parse attributes
    attributes = {}
    for line in attr_input.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            attributes[key.strip()] = val.strip()

    # Create Object
    product_data = ProductInput(name=p_name, category=cat, attributes=attributes, tone=tone)

    # Generate
    with st.spinner("Drafting copy..."):
        description = generate_description(product_data)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generated Description")
        st.markdown(f"> {description}")

    # Evaluate
    with col2:
        st.subheader("Quality Score")
        with st.spinner("Evaluating..."):
            eval_result = evaluate_quality(product_data.model_dump(), description)
            
            st.metric("Consistency Score", f"{eval_result.get('score')}/10")
            
            if eval_result.get('consistency_check') == "Pass":
                st.success("✅ Specs Match")
            else:
                st.error("❌ Missing Info")
                
            st.info(f"**Tone:** {eval_result.get('tone_feedback')}")