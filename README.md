# AI Model Playground

A beautiful Gradio-based web interface for interacting with multiple AI models through Gemini API and Hugging Face Inference API.

## Features

- 🎨 Modern, clean UI with purple theme
- 🤖 Support for multiple AI models:
  - **Gemini 2.0 Flash** (via Gemini API)
  - **Meta Llama 3.1 8B** (via Hugging Face API)
  - **Mistral 7B Instruct** (via Hugging Face API)
- ⚡ Real-time AI responses
- 🎯 Easy-to-use interface

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

#### Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

#### Hugging Face API Key
1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
2. Sign up or log in to your account
3. Create a new access token (with "read" permissions is sufficient)
4. Copy the token

### 3. Configure API Keys

You have two options:

#### Option A: Environment Variables (Recommended)

Create a `.env` file in the project root and add your API keys:

```bash
# Create .env file
cat > .env << EOF
GEMINI_API_KEY=your_actual_gemini_api_key
OPENROUTER_API_KEY=your_actual_openrouter_api_key
EOF
```

Or manually create a `.env` file with the following content:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
HUGGINGFACE_API_KEY=your_actual_huggingface_api_key
```

#### Option B: Gradio Secrets

When launching the app, you can pass API keys through Gradio's secrets:

```python
app.launch(share=False, auth=("username", "password"))
```

Or set them as environment variables before running:

```bash
export GEMINI_API_KEY="your_key_here"
export HUGGINGFACE_API_KEY="your_key_here"
python app.py
```

### 4. Run the Application

```bash
python app.py
```

The app will start on `http://127.0.0.1:7860` by default.

## Usage

1. **Select a Model**: Choose from the dropdown menu (Gemini 2.0 Flash, Meta Llama 3.1 8B, or Mistral 7B Instruct)
2. **Enter Your Prompt**: Type your question or prompt in the text area
3. **Generate Response**: Click the "Generate Response" button or press Enter
4. **View Results**: The AI-generated response will appear in the response area below

## Project Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Your actual API keys (create this, don't commit)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Adding More Models

To add more models, edit the `MODEL_MAPPING` dictionary in `app.py`:

```python
MODEL_MAPPING = {
    "Your Model Name": {
        "type": "gemini",  # or "huggingface"
        "model_name": "model-id-here"  # For Hugging Face, use format: "username/model-name"
    }
}
```

Then add the model name to the dropdown choices.

## Troubleshooting

### "API key is not set" Error
- Make sure you've created a `.env` file with your API keys
- Or set the environment variables before running the app
- Check that the API keys are correct

### Connection Errors
- Check your internet connection
- Verify your API keys are valid
- For Hugging Face, free tier models may take 20-30 seconds to load on first use (you'll get a 503 error)

### Model Loading (503 Error)
- Hugging Face free tier models load on-demand and may take time
- Wait 20-30 seconds and try again if you see a "model is loading" error
- Consider using paid inference endpoints for faster response times

### Model Not Found
- Verify the model name is correct in `MODEL_MAPPING`
- For Hugging Face models, check the model ID on [Hugging Face Models](https://huggingface.co/models)
- Ensure the model supports text generation and is publicly accessible

## License

MIT License - feel free to use this project for your own purposes!

## Credits

Built with [Gradio](https://gradio.app/), [Google Gemini API](https://ai.google.dev/), and [Hugging Face Inference API](https://huggingface.co/docs/api-inference).

