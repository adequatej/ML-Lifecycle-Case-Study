import gradio as gr
from huggingface_hub import InferenceClient
# Uncomment below for local model inference
# import torch
# from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from prometheus_client import start_http_server, Counter, Summary

# Inference client setup (API-based)
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Uncomment below for local model inference (requires GPU)
# pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Prometheus Metrics
RECOMMENDATIONS_PROCESSED = Counter('app_recommendations_processed', 'Total number of recommendation requests processed')
SUCCESSFUL_RECOMMENDATIONS = Counter('app_successful_recommendations', 'Total number of successful recommendations')
FAILED_RECOMMENDATIONS = Counter('app_failed_recommendations', 'Total number of failed recommendations')
RECOMMENDATION_DURATION = Summary('app_recommendation_duration_seconds', 'Time spent processing recommendation')
USER_INTERACTIONS = Counter('app_user_interactions', 'Total number of user interactions')
CANCELLED_RECOMMENDATIONS = Counter('app_cancelled_recommendations', 'Total number of cancelled recommendations')


def spotify_rec(track_name, artist, client_id, client_secret):
    """
    Get Spotify song recommendations based on a seed track.
    
    Args:
        track_name: Name of the seed track
        artist: Artist of the seed track
        client_id: Spotify API client ID
        client_secret: Spotify API client secret
    
    Returns:
        String containing list of recommended songs
    """
    # Validate client ID and client secret
    if not client_id or not client_secret:
        FAILED_RECOMMENDATIONS.inc()
        return "Please provide Spotify API credentials."

    # Set up Spotify credentials
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Get track 
    results = sp.search(q=f"track:{track_name} artist:{artist}", type='track')
    if not results['tracks']['items']:
        FAILED_RECOMMENDATIONS.inc()
        return "No tracks found for the given track name and artist."
    
    track_uri = results['tracks']['items'][0]['uri']

    # Get recommended tracks
    recommendations = sp.recommendations(seed_tracks=[track_uri])['tracks']
    
    if not recommendations:
        FAILED_RECOMMENDATIONS.inc()
        return "No recommendations found."
    
    recommendation_list = [f"{track['name']} by {track['artists'][0]['name']}" for track in recommendations]
    
    SUCCESSFUL_RECOMMENDATIONS.inc()
    RECOMMENDATIONS_PROCESSED.inc()
    
    return "\n".join(recommendation_list)


# Global flag to handle cancellation
stop_inference = False


@RECOMMENDATION_DURATION.time()
def respond(
    track_name,
    artist,
    history: list[tuple[str, str]],
    system_message="You are a music expert chatbot that provides song recommendations based on user preferences.",
    max_tokens=512,
    use_local_model=False,
    client_id=None,
    client_secret=None
):
    """
    Process user request and generate music recommendations using Spotify API
    and LLM-powered conversation.
    
    Args:
        track_name: Name of the song to base recommendations on
        artist: Artist of the song
        history: Chat history
        system_message: System prompt for the LLM
        max_tokens: Maximum tokens to generate
        use_local_model: If True, use local pipeline instead of API
        client_id: Spotify API client ID
        client_secret: Spotify API client secret
    """
    global stop_inference
    stop_inference = False
    USER_INTERACTIONS.inc()

    if history is None:
        history = []

    # Get Spotify recommendations
    recommendations = spotify_rec(track_name, artist, client_id, client_secret)
    
    # Build conversation with LLM to enhance recommendations
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    
    user_message = f"I like '{track_name}' by {artist}. Here are some Spotify recommendations:\n{recommendations}\n\nCan you tell me more about these songs and suggest why I might enjoy them?"
    messages.append({"role": "user", "content": user_message})

    response = f"**Spotify Recommendations based on '{track_name}' by {artist}:**\n{recommendations}\n\n"
    
    if use_local_model:
        # Local model inference (requires uncommenting imports and pipeline above)
        # This uses a locally-loaded model for inference
        try:
            for output in pipe(
                messages,
                max_new_tokens=max_tokens,
                do_sample=True,
            ):
                if stop_inference:
                    CANCELLED_RECOMMENDATIONS.inc()
                    response += "\n\n*Inference cancelled.*"
                    yield history + [(f"{track_name} by {artist}", response)]
                    return
                token = output['generated_text'][-1]['content']
                response += token
                yield history + [(f"{track_name} by {artist}", response)]
        except NameError:
            # pipe is not defined - local model not configured
            FAILED_RECOMMENDATIONS.inc()
            response += "*Error: Local model not configured. Please uncomment the torch/transformers imports and pipeline initialization in app.py, or uncheck 'Use Local Model'.*"
            yield history + [(f"{track_name} by {artist}", response)]
            return
    else:
        # API-based inference using HuggingFace Inference API
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
        ):
            if stop_inference:
                CANCELLED_RECOMMENDATIONS.inc()
                response += "\n\n*Inference cancelled.*"
                yield history + [(f"{track_name} by {artist}", response)]
                return
            token = message_chunk.choices[0].delta.content
            if token:
                response += token
            yield history + [(f"{track_name} by {artist}", response)]


def cancel_inference():
    """Cancel the ongoing inference."""
    global stop_inference
    stop_inference = True


# Custom CSS for styling
custom_css = """
#main-container {
    background-color: #121212;
    font-family: 'Circular', 'Helvetica Neue', sans-serif;
}

.gradio-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background: linear-gradient(135deg, #1DB954 0%, #191414 50%);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border-radius: 12px;
}

.gr-button {
    background-color: #1DB954;
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 24px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: bold;
}

.gr-button:hover {
    background-color: #1ed760;
    transform: scale(1.05);
}

.gr-chat {
    font-size: 16px;
    background-color: #282828;
    border-radius: 8px;
}

#title {
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 20px;
    color: #1DB954;
}
"""

# Define the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #1DB954;'>🎵 AI Music Recommendation Bot 🎵</h1>")
    gr.Markdown("""
    Enter a song and artist you like, and get personalized Spotify recommendations 
    enhanced with AI-powered insights from Zephyr-7B.
    """)

    with gr.Row():
        system_message = gr.Textbox(
            value="You are a music expert chatbot that provides insightful song recommendations based on user preferences.",
            label="System Message",
            interactive=True
        )
        use_local_model = gr.Checkbox(label="Use Local Model", value=False)

    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max Tokens")

    chat_history = gr.Chatbot(label="Recommendations")

    with gr.Row():
        track_name = gr.Textbox(label="Song Name", placeholder="Enter a song name (e.g., Shape of You)")
        artist = gr.Textbox(label="Artist", placeholder="Enter the artist (e.g., Ed Sheeran)")
    
    with gr.Row():
        client_id = gr.Textbox(label="Spotify Client ID", placeholder="Your Spotify API Client ID")
        client_secret = gr.Textbox(label="Spotify Client Secret", placeholder="Your Spotify API Client Secret", type="password")

    with gr.Row():
        submit_btn = gr.Button("Get Recommendations", variant="primary")
        cancel_button = gr.Button("Cancel", variant="secondary")

    submit_btn.click(
        respond,
        [track_name, artist, chat_history, system_message, max_tokens, use_local_model, client_id, client_secret],
        chat_history
    )
    track_name.submit(
        respond,
        [track_name, artist, chat_history, system_message, max_tokens, use_local_model, client_id, client_secret],
        chat_history
    )
    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Launch Gradio app
    demo.launch(server_port=7860, share=False)
