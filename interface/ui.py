# ui.py
# type: ignore
import gradio as gr

def build_interface(chatbot_response):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # 🔹 Stylish Gradient Header
        gr.HTML("""
            <div style="text-align: center; padding: 1.5rem; 
                        background: linear-gradient(90deg, #6a11cb, #2575fc); 
                        color: white; border-radius: 16px; margin-bottom: 2rem;">
                <h1 style="margin:0; font-size:2.2rem;">🤖 Document AI Assistant</h1>
                <p style="margin-top:0.6rem; font-size:1.1rem; line-height:1.6;">
                    ✨ Transform your documents into intelligent conversations!  
                    📂 Upload <b>PDF, DOCX, TXT, PPTX, CSV, XLSX</b> files and ask anything in any language 🌍🔥
                </p>
            </div>
        """)

        # 🔹 Feature Highlights Section
        with gr.Row():
            gr.HTML("""
                <div style="display: flex; justify-content: space-around; margin-bottom: 1.5rem;">
                    <div style="background:#fef9c3; padding:1rem; border-radius:12px; width:22%; text-align:center; box-shadow:0 3px 8px rgba(0,0,0,0.1);">
                        📂 <h4>Multi-Format Support</h4>
                        <p style="font-size:0.9rem;">PDF, DOCX, TXT, PPTX, CSV, XLSX</p>
                    </div>
                    <div style="background:#e0f7fa; padding:1rem; border-radius:12px; width:22%; text-align:center; box-shadow:0 3px 8px rgba(0,0,0,0.1);">
                        🌐 <h4>Multilingual Magic</h4>
                        <p style="font-size:0.9rem;">English, Urdu & More Languages</p>
                    </div>
                    <div style="background:#fce7f3; padding:1rem; border-radius:12px; width:22%; text-align:center; box-shadow:0 3px 8px rgba(0,0,0,0.1);">
                        ⚡ <h4>AI Powered</h4>
                        <p style="font-size:0.9rem;">Advanced Natural Language Processing</p>
                    </div>
                    <div style="background:#e0f2fe; padding:1rem; border-radius:12px; width:22%; text-align:center; box-shadow:0 3px 8px rgba(0,0,0,0.1);">
                        🔒 <h4>Secure & Safe</h4>
                        <p style="font-size:0.9rem;">Your data stays private & protected</p>
                    </div>
                </div>
            """)

        with gr.Row():
            # 🔹 Left Column (Inputs & Controls)
            with gr.Column(scale=1):
                gr.HTML("<h3>📂 Document Upload Zone</h3>")
                file_output = gr.File(
                    label="Upload your files here",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".pptx", ".csv", ".xlsx"]
                )

                gr.HTML("<h3>🌐 Response Settings</h3>")
                language_dropdown = gr.Dropdown(
                    label="Choose Response Language",
                    choices=["English", "Urdu"],
                    value="English",
                    interactive=True
                )

                clear_btn = gr.Button("🧹 Clear Chat", variant="secondary")

            # 🔹 Right Column (Chat Area)
            with gr.Column(scale=2):
                chatbot_ui = gr.Chatbot(
                    label="💬 Chat with AI",
                    bubble_full_width=False,
                    height=500
                )

                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="✍ Ask me anything about your document...",
                        scale=4,
                        lines=2
                    )
                    submit_button = gr.Button("🚀 Ask", variant="primary", scale=1)

        # 🔹 Chatbot Response Function
        def respond(files, message, language, history):
            response = chatbot_response(files, message, language)
            history = history + [(message, response[0][1])]
            return history, history, ""  # clear textbox after sending

        # 🔹 Button & Enter Key Actions
        submit_button.click(
            fn=respond,
            inputs=[file_output, text_input, language_dropdown, chatbot_ui],
            outputs=[chatbot_ui, chatbot_ui, text_input],
        )

        text_input.submit(
            fn=respond,
            inputs=[file_output, text_input, language_dropdown, chatbot_ui],
            outputs=[chatbot_ui, chatbot_ui, text_input],
        )

        clear_btn.click(lambda: None, None, chatbot_ui, queue=False)

    return demo