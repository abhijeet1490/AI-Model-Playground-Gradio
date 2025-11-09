import gradio as gr
import os
import json
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

# --- API Configuration ---
# Load API keys from environment variables or Gradio secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Hugging Face Inference API - using InferenceClient for automatic endpoint routing
# The client will handle the new router endpoint automatically

# Model mapping for API calls
MODEL_MAPPING = {
    "Gemini 2.0 Flash": {
        "type": "gemini",
        "model_name": "gemini-2.0-flash-exp"
    },
    "Meta Llama 3.1 8B": {
        "type": "huggingface",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Hugging Face model ID
    },
    "Mistral 7B Instruct": {
        "type": "huggingface",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2"  # Hugging Face model ID
    }
}

# --- Backend Functions ---

def call_gemini_api(prompt: str, model_name: str = "gemini-2.0-flash-exp") -> str:
    """Call Google Gemini API"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Check if response has text
        if response.text:
            return response.text
        else:
            return "Error: Empty response from Gemini API. The model may have been blocked or returned no content."
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages
        if "API_KEY_INVALID" in error_msg or "api key" in error_msg.lower():
            return "Error: Invalid Gemini API key. Please check your GEMINI_API_KEY."
        elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            return f"Error: API quota or rate limit exceeded. {error_msg}"
        else:
            return f"Error calling Gemini API: {error_msg}"

def call_huggingface_api(prompt: str, model_name: str) -> str:
    """Call Hugging Face Inference API using InferenceClient"""
    try:
        # Initialize InferenceClient with the new router endpoint
        # Setting provider to "hf-inference" to use the new router endpoint
        client = InferenceClient(
            model=model_name,
            token=HUGGINGFACE_API_KEY,
            timeout=90.0,
            provider="hf-inference"  # Use the new router endpoint
        )
        
        # Use text_generation method for text generation models
        try:
            response = client.text_generation(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False,
                do_sample=True
            )
            
            # The response is already a string from text_generation
            if response:
                return response.strip()
            else:
                return "Error: Empty response from Hugging Face API."
                
        except Exception as text_gen_error:
            # If text_generation fails, try chat completion for instruction models
            if "instruct" in model_name.lower() or "chat" in model_name.lower() or "llama" in model_name.lower():
                try:
                    # For chat models, format as a conversation
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat_completion(
                        messages=messages,
                        max_tokens=512,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    # Extract text from chat completion response
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        return response.choices[0].message.content
                    elif isinstance(response, dict) and 'choices' in response:
                        return response['choices'][0]['message']['content']
                    else:
                        return str(response)
                except Exception as chat_error:
                    return f"Error: Both text generation and chat completion failed. Text Gen Error: {str(text_gen_error)}. Chat Error: {str(chat_error)}"
            else:
                raise text_gen_error
                
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific error cases
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return "Error: Invalid Hugging Face API key. Please check your HUGGINGFACE_API_KEY."
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return "Error: Rate limit exceeded. Please wait a moment and try again."
        elif "503" in error_msg or "loading" in error_msg.lower():
            return "Error: The model is currently loading. Hugging Face free tier models may take 20-30 seconds to load on first use. Please wait and try again."
        elif "timeout" in error_msg.lower():
            return "Error: Request timed out. The model may be taking too long to respond. Please try again."
        elif "404" in error_msg or "not found" in error_msg.lower():
            return f"Error: Model '{model_name}' not found. Please check the model name is correct."
        else:
            return f"Error calling Hugging Face API: {error_msg}"

def generate_response(model_choice: str, prompt_text: str) -> str:
    """Main function to generate AI response based on selected model"""
    if not prompt_text or not prompt_text.strip():
        return "Please enter a prompt before generating a response."
    
    # Get model configuration
    if model_choice not in MODEL_MAPPING:
        return f"Error: Unknown model '{model_choice}'"
    
    model_config = MODEL_MAPPING[model_choice]
    
    # Check API keys
    if model_config["type"] == "gemini":
        if not GEMINI_API_KEY:
            return "Error: Gemini API key is not set. Please set GEMINI_API_KEY environment variable or in Gradio secrets."
        response = call_gemini_api(prompt_text, model_config["model_name"])
    elif model_config["type"] == "huggingface":
        if not HUGGINGFACE_API_KEY:
            return "Error: Hugging Face API key is not set. Please set HUGGINGFACE_API_KEY environment variable or in Gradio secrets."
        response = call_huggingface_api(prompt_text, model_config["model_name"])
    else:
        return f"Error: Unknown model type '{model_config['type']}'"
    
    return response

# --- CSS for styling ---
custom_css = """
/* Center the whole app on the page */
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
    padding-top: 2rem !important;
}

/* Style the main title "AI Model Playground" */
h1#title {
    color: #6D28D9; /* A nice purple color */
    font-weight: 700;
    font-size: 2.5rem;
    text-align: center;
    padding-bottom: 1rem;
}

/* Style the "Generate Response" button */
button.gradio-button.lg.primary {
    background-color: #7C3AED !important; /* A slightly lighter purple */
    border: none !important;
    font-size: 1.1rem !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease !important;
}

/* Make the button slightly darker on hover */
button.gradio-button.lg.primary:hover {
    background-color: #6D28D9 !important;
}

/* Add a subtle shadow and rounding to the main blocks and columns */
.gradio-block.gradio-box,
.gradio-column {
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

/* Style the text areas */
.gradio-textbox textarea {
    border-radius: 8px !important;
}

/* Style the dropdown */
.gradio-dropdown {
    border-radius: 8px !important;
}
"""

# --- Define the Gradio UI using gr.Blocks ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    
    # 1. Title
    gr.Markdown("# AI Model Playground", elem_id="title")
    
    # 2. Model Selection Dropdown
    model_dropdown = gr.Dropdown(
        label="Select a Model",
        choices=[
            "Gemini 2.0 Flash", 
            "Meta Llama 3.1 8B", 
            "Mistral 7B Instruct"
        ],
        value="Gemini 2.0 Flash", # Default selection
        interactive=True
    )
    
    # Use gr.Column for layout grouping (replaces deprecated gr.Box)
    with gr.Column():
        # 3. Prompt Input Textbox
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt here... (Shift+Enter for new line)",
            lines=8,
            interactive=True
        )
        
        # 4. Generate Button
        generate_button = gr.Button(
            "Generate Response", 
            variant="primary" # Makes it the main button (blue/purple)
        )
    
    with gr.Column():
        # 5. Response Output Textbox
        response_output = gr.Textbox(
            label="Response",
            placeholder="Your AI-generated content will appear here...",
            lines=10,
            interactive=False # User shouldn't be able to type here
        )
    
    # --- Wire up the components ---
    # When the button is clicked:
    # 1. Call the `generate_response` function
    # 2. Get inputs from `model_dropdown` and `prompt_input`
    # 3. Send the output to `response_output`
    generate_button.click(
        fn=generate_response,
        inputs=[model_dropdown, prompt_input],
        outputs=[response_output]
    )
    
    # Also allow Enter key to trigger generation (when prompt is focused)
    prompt_input.submit(
        fn=generate_response,
        inputs=[model_dropdown, prompt_input],
        outputs=[response_output]
    )

# --- Launch the app ---
if __name__ == "__main__":
    app.launch(share=False)

