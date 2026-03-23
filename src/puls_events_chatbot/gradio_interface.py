import gradio as gr
import requests
import os

API_BASE = os.getenv("API_BASE")

def chat(message,*_):
    """
    Envoie le message à l'API POST /ask et retourne la réponse.
    """
    try:
        response = requests.post(f"{API_BASE}/ask", json={"message": message})
        response.raise_for_status()
        data = response.json()
        # Supposons que l'API retourne {'answer': "...", 'code': "..."}
        answer = data.get("answer", "Pas de réponse")
        code_snippet = data.get("code", "")
        if code_snippet:
            return f"{answer}\n\n```python\n{code_snippet}\n```"
        return answer
    except Exception as e:
        return f"Erreur lors de l'appel à l'API : {e}"

def rebuild_api(username):
    """
    Appelle l'API GET /rebuild pour relancer la partie de l'API sous condition du l'username.
    Ne renvoie rien.
    """
    try:
        response = requests.get(f"{API_BASE}/rebuild", json={"username": username})
        response.raise_for_status()
    except Exception as e:
        print(f"Erreur lors du rebuild : {e}")

# ------------------ Interface Gradio ------------------
def launch_frontend():
    with gr.Blocks() as frontend:
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("<center><h1>Votre assistant</h1></center>")
                
                chat_interface = gr.ChatInterface(
                    fn=chat,
                    type="messages"
                )
                with gr.Row(equal_height=True):
                    username = gr.Textbox(lines=1, show_label=True,placeholder="Username")
                    rebuild_button = gr.Button("Rebuild API")
                    rebuild_button.click(fn=rebuild_api, inputs=username, outputs=None)  

    return frontend