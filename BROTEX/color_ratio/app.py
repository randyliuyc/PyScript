import gradio as gr
from TINHDON1KEODAI import one_stretch_app
from TINHDON3KEODAN import three_stretch_app
from TINHDON4KEODAN import four_stretch_app
from TINHDON1KEODAI_zh import one_stretch_app_zh
from TINHDON3KEODAN_zh import three_stretch_app_zh
from TINHDON4KEODAN_zh import four_stretch_app_zh

def main_app():
    with gr.Blocks() as demo:
        vi_container = gr.Column(visible=True)
        zh_container = gr.Column(visible=False)

        # ===== TIẾNG VIỆT =====
        with vi_container:
            with gr.Row():
                with gr.Tabs():
                    with gr.Tab("🔁 Tính theo 3 kéo dãn"):
                        three_stretch_app.render()
                    with gr.Tab("🔍 Tính theo 4 kéo dãn"):
                        four_stretch_app.render()
                    with gr.Tab("🧵 Tính theo 1 kéo dãn"):
                        one_stretch_app.render()                    
                    with gr.Tab("🌐 Ngôn ngữ/语言"):
                        lang_vi = gr.Dropdown(
                            choices=["Tiếng Việt", "中文"], 
                            value="Tiếng Việt",
                            label="Chọn ngôn ngữ",
                            interactive=True
                        )
        # ===== TIẾNG TRUNG =====
        with zh_container:
            with gr.Row():
                with gr.Tabs():
                    with gr.Tab("🔁 三次拉伸计算"):
                        three_stretch_app_zh.render()
                    with gr.Tab("🔍 按照4个牵伸计算"):
                        four_stretch_app_zh.render()
                    with gr.Tab("🧵 一次拉伸计算"):
                        one_stretch_app_zh.render()
                    with gr.Tab("🌐 语言/Ngôn ngữ"):
                        lang_zh = gr.Dropdown(
                            choices=["Tiếng Việt", "中文"], 
                            value="中文", 
                            label="选择语言",
                            interactive=True
                        )

        # ===== XỬ LÝ CHUYỂN NGÔN NGỮ =====
        def switch_language(lang_value):
            if lang_value == "Tiếng Việt":
                return gr.update(visible=True), gr.update(visible=False), gr.update(value="Tiếng Việt")
            else:
                return gr.update(visible=False), gr.update(visible=True), gr.update(value="中文")
        
        lang_vi.change(fn=switch_language, inputs=lang_vi, outputs=[vi_container, zh_container, lang_zh])
        lang_zh.change(fn=switch_language, inputs=lang_zh, outputs=[vi_container, zh_container, lang_vi])
    return demo

app = main_app()
app.launch()
