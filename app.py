import os
import sys
import argparse
from typing import Dict, Optional, Tuple

import gradio as gr
from PIL import Image

from infer import CycleGANInferencer, PRETRAINED_MODELS_INFO, list_available_models
from config import config

HTML_HEADER = """
<div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 25px;">
    <h1 style="color: white; font-size: 2.5em; margin: 0; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
        🎨 CycleGAN 图像风格转换
    </h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin-top: 12px; font-weight: 300;">
        基于深度神经网络的无监督图像风格转换 · 无需成对训练数据
    </p>
</div>
"""

HTML_FOOTER = """
<div style="text-align: center; padding: 25px 15px; margin-top: 30px; border-top: 1px solid #e5e7eb;">
    <p style="color: #6b7280; font-size: 0.95em;">
        ✨ 基于项目：<a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 500;">
        junyanz/pytorch-CycleGAN-and-pix2pix
        </a>
    </p>
    <p style="color: #9ca3af; font-size: 0.85em; margin-top: 8px;">
        技术栈：PyTorch · torchvision · Pillow · matplotlib · Gradio
    </p>
</div>
"""

MODEL_CARD_TEMPLATE = """
<div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 15px 20px; border-radius: 10px; margin: 10px 0;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h4 style="margin: 0; color: #374151; font-size: 1em;">{icon} {name}</h4>
            <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 0.9em;">{desc}</p>
        </div>
        <div style="background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 500;">
            {direction}
        </div>
    </div>
</div>
"""

class GradioApp:
    def __init__(self, 
                 checkpoint_dir: str = './checkpoints/pretrained',
                 default_model: str = 'summer2winter_yosemite'):
        self.checkpoint_dir = checkpoint_dir
        self.default_model = default_model
        self.current_model: Optional[str] = None
        self.inferencer: Optional[CycleGANInferencer] = None
        
        self.model_descriptions: Dict[str, str] = {
            name: info['description'] for name, info in PRETRAINED_MODELS_INFO.items()
        }
        
        self.model_directions: Dict[str, Dict[str, str]] = {}
        for name, info in PRETRAINED_MODELS_INFO.items():
            self.model_directions[name] = {
                'AtoB': info['direction'][0],
                'BtoA': info['direction'][1]
            }
        
        self.model_icons = {
            'summer2winter_yosemite': '❄️',
            'apple2orange': '🍊',
            'horse2zebra': '🦓',
            'monet2photo': '🎨',
            'cezanne2photo': '🖼️',
            'ukiyoe2photo': '🎎',
            'vangogh2photo': '🌻'
        }
    
    def _get_inferencer(self, model_name: str) -> CycleGANInferencer:
        if self.current_model != model_name or self.inferencer is None:
            print(f'Loading model: {model_name}')
            self.inferencer = CycleGANInferencer(
                model_name=model_name,
                checkpoint_dir=self.checkpoint_dir
            )
            self.current_model = model_name
        
        return self.inferencer
    
    def get_direction_choices(self, model_name: str) -> Tuple[Dict[str, str], str]:
        directions = self.model_directions.get(model_name, {})
        direction_choices = {
            directions['AtoB']: 'AtoB',
            directions['BtoA']: 'BtoA'
        }
        default_direction = list(direction_choices.keys())[0]
        return direction_choices, default_direction
    
    def transform_image(self, 
                        image: Optional[Image.Image], 
                        model_name: str, 
                        direction_display: str,
                        show_cycle: bool = False) -> Tuple:
        if image is None:
            if show_cycle:
                return None, None, "⚠️ 请先上传一张图像"
            return None, "⚠️ 请先上传一张图像"
        
        try:
            inferencer = self._get_inferencer(model_name)
            
            directions = self.model_directions.get(model_name, {})
            if directions['AtoB'] == direction_display:
                direction = 'AtoB'
            else:
                direction = 'BtoA'
            
            if show_cycle:
                transformed, recovered = inferencer.transform_with_cycle(image, direction)
                icon = self.model_icons.get(model_name, '🎨')
                message = f"✅ 转换成功！{icon} 模型: {self.model_descriptions[model_name]} · 方向: {direction_display}"
                return transformed, recovered, message
            else:
                transformed = inferencer.transform(image, direction)
                icon = self.model_icons.get(model_name, '🎨')
                message = f"✅ 转换成功！{icon} 模型: {self.model_descriptions[model_name]} · 方向: {direction_display}"
                return transformed, message
                
        except Exception as e:
            error_msg = f"❌ 转换过程中出错: {str(e)}"
            print(error_msg)
            if show_cycle:
                return None, None, error_msg
            return None, error_msg
    
    def update_direction_dropdown(self, model_name: str) -> Dict:
        direction_choices, default_direction = self.get_direction_choices(model_name)
        return {
            'choices': list(direction_choices.keys()),
            'value': default_direction
        }
    
    def get_model_cards_html(self) -> str:
        html_parts = []
        for name, info in PRETRAINED_MODELS_INFO.items():
            icon = self.model_icons.get(name, '🎨')
            html_parts.append(MODEL_CARD_TEMPLATE.format(
                icon=icon,
                name=name.replace('_', ' ').title(),
                desc=info['description'],
                direction=' ↔ '.join(info['direction'])
            ))
        return ''.join(html_parts)
    
    def create_interface(self, server_name: str = "0.0.0.0", server_port: int = 7860):
        model_choices = list(self.model_descriptions.keys())
        default_model = self.default_model if self.default_model in model_choices else model_choices[0]
        
        initial_directions = self.model_directions.get(default_model, {})
        initial_direction_choices = {
            initial_directions['AtoB']: 'AtoB',
            initial_directions['BtoA']: 'BtoA'
        } if initial_directions else {}
        initial_direction = list(initial_direction_choices.keys())[0] if initial_direction_choices else ''
        
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        .tab-nav {
            background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%) !important;
            border-radius: 10px 10px 0 0 !important;
            padding: 5px 5px 0 5px !important;
        }
        
        .tab-nav button {
            border: none !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 12px 24px !important;
            font-weight: 500 !important;
            font-size: 1em !important;
            margin: 0 2px !important;
            transition: all 0.3s ease !important;
        }
        
        .tab-nav button:not(.selected) {
            background: transparent !important;
            color: #64748b !important;
        }
        
        .tab-nav button:hover:not(.selected) {
            background: rgba(102, 126, 234, 0.1) !important;
            color: #667eea !important;
        }
        
        .tab-nav button.selected {
            background: white !important;
            color: #667eea !important;
            font-weight: 600 !important;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05) !important;
        }
        
        .image-container {
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }
        
        .primary-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .primary-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
        }
        
        .secondary-btn {
            background: white !important;
            border: 2px solid #e2e8f0 !important;
            color: #64748b !important;
            font-weight: 500 !important;
            padding: 12px 24px !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        .secondary-btn:hover {
            border-color: #667eea !important;
            color: #667eea !important;
            background: rgba(102, 126, 234, 0.05) !important;
        }
        
        .accordion-header {
            background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%) !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
            margin-bottom: 10px !important;
        }
        
        .accordion-header button {
            font-weight: 600 !important;
            color: #374151 !important;
        }
        
        .dropdown-container {
            background: white !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        .status-box {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
            border-radius: 10px !important;
            border: 1px solid #bae6fd !important;
        }
        
        .status-box textarea {
            background: transparent !important;
            border: none !important;
            font-weight: 500 !important;
        }
        
        .gallery-container {
            background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%) !important;
            border-radius: 15px !important;
            padding: 20px !important;
        }
        """
        
        with gr.Blocks(title="CycleGAN 图像风格转换", css=css) as app:
            gr.HTML(HTML_HEADER)
            
            with gr.Tabs() as main_tabs:
                
                with gr.TabItem("🎬 风格转换", id=0):
                    with gr.Row(equal_height=True):
                        
                        with gr.Column(scale=1, min_width=300):
                            with gr.Box(elem_classes="image-container"):
                                gr.Markdown("""
                                <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                    <span style="color: white; font-weight: 600; font-size: 1.1em;">📸 上传原始图像</span>
                                </div>
                                """)
                                input_image = gr.Image(
                                    type="pil", 
                                    label="",
                                    height=350,
                                    elem_classes="image-upload"
                                )
                            
                            with gr.Accordion("⚙️ 转换设置", open=True, elem_classes="accordion-header"):
                                model_dropdown = gr.Dropdown(
                                    choices=model_choices,
                                    value=default_model,
                                    label="选择转换模型",
                                    info="不同的模型支持不同的风格转换",
                                    elem_classes="dropdown-container"
                                )
                                
                                direction_dropdown = gr.Dropdown(
                                    choices=list(initial_direction_choices.keys()),
                                    value=initial_direction,
                                    label="转换方向",
                                    info="选择从哪种风格转换到哪种风格",
                                    elem_classes="dropdown-container"
                                )
                                
                                show_cycle_checkbox = gr.Checkbox(
                                    label="🔄 显示循环一致性验证",
                                    value=False,
                                    info="显示 '原始 → 转换 → 恢复' 的完整循环，验证转换质量"
                                )
                            
                            with gr.Row():
                                transform_btn = gr.Button(
                                    "✨ 开始转换", 
                                    variant="primary",
                                    elem_classes="primary-btn",
                                    scale=2
                                )
                                clear_btn = gr.Button(
                                    "🔄 清空", 
                                    elem_classes="secondary-btn",
                                    scale=1
                                )
                        
                        with gr.Column(scale=1, min_width=300):
                            with gr.Box(elem_classes="image-container"):
                                gr.Markdown("""
                                <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                                    <span style="color: white; font-weight: 600; font-size: 1.1em;">🎨 风格转换结果</span>
                                </div>
                                """)
                                output_transformed = gr.Image(
                                    type="pil", 
                                    label="",
                                    height=350
                                )
                            
                            with gr.Box(elem_classes="image-container", visible=False) as recovered_box:
                                gr.Markdown("""
                                <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                                    <span style="color: white; font-weight: 600; font-size: 1.1em;">🔄 循环恢复结果</span>
                                </div>
                                """)
                                output_recovered = gr.Image(
                                    type="pil", 
                                    label="",
                                    height=250
                                )
                            
                            with gr.Box(elem_classes="status-box"):
                                status_text = gr.Textbox(
                                    label="状态信息",
                                    lines=2,
                                    interactive=False,
                                    placeholder="等待上传图像并点击转换..."
                                )
                
                with gr.TabItem("📚 模型库", id=1):
                    with gr.Box():
                        gr.Markdown("""
                        <div style="text-align: center; padding: 20px;">
                            <h3 style="margin: 0; color: #374151; font-size: 1.5em;">🎯 支持的预训练模型</h3>
                            <p style="margin: 10px 0 0 0; color: #6b7280;">
                                所有模型均来自官方开源项目，首次使用会自动下载（约80MB）
                            </p>
                        </div>
                        """)
                        
                        gr.HTML(self.get_model_cards_html())
                        
                        with gr.Accordion("📋 详细说明", open=False):
                            gr.Markdown("""
                            ### 关于CycleGAN
                            
                            CycleGAN是一种无监督的图像到图像转换方法，它的特点是：
                            
                            1. **无需成对训练数据**：只需要两个领域的独立图像集，不需要一一对应的成对图像
                            2. **循环一致性**：通过 `A→B→A` 和 `B→A→B` 的循环约束，保证转换的一致性
                            3. **身份损失**：可选的身份损失，保证风格转换时不会改变图像内容
                            
                            ### 模型说明
                            
                            | 模型 | 用途 | 适用场景 |
                            |------|------|----------|
                            | summer2winter | 季节转换 | 风景照片的季节变化 |
                            | apple2orange | 物体转换 | 水果之间的风格转换 |
                            | horse2zebra | 动物转换 | 野生动物的风格转换 |
                            | monet2photo | 艺术风格 | 莫奈油画风格和照片的互转 |
                            | cezanne2photo | 艺术风格 | 塞尚油画风格和照片的互转 |
                            | ukiyo-e2photo | 艺术风格 | 浮世绘风格和照片的互转 |
                            | vangogh2photo | 艺术风格 | 梵高油画风格和照片的互转 |
                            
                            ### 使用建议
                            
                            - **照片转油画**：推荐使用 monet2photo、cezanne2photo、vangogh2photo 等模型
                            - **油画转照片**：选择相反的方向（如 photo2monet）
                            - **季节转换**：使用 summer2winter_yosemite 模型
                            - **物体转换**：使用 apple2orange 或 horse2zebra 模型
                            """)
                
                with gr.TabItem("❓ 使用指南", id=2):
                    with gr.Box():
                        gr.Markdown("""
                        <div style="text-align: center; padding: 20px;">
                            <h3 style="margin: 0; color: #374151; font-size: 1.5em;">📖 快速上手指南</h3>
                        </div>
                        """)
                    
                    with gr.Row():
                        with gr.Column():
                            with gr.Box():
                                gr.Markdown("""
                                <div style="padding: 15px;">
                                    <h4 style="color: #667eea; margin: 0 0 10px 0;">📤 第一步：上传图像</h4>
                                    <p style="color: #6b7280; margin: 0; line-height: 1.6;">
                                        点击左侧的图像上传区域，可以：<br>
                                        • 从本地上传图片<br>
                                        • 拖拽图片到上传区域<br>
                                        • 使用摄像头拍摄（如果设备支持）<br><br>
                                        <span style="color: #9ca3af; font-size: 0.9em;">
                                        支持的格式：JPG、PNG、BMP 等常见图像格式
                                        </span>
                                    </p>
                                </div>
                                """)
                        
                        with gr.Column():
                            with gr.Box():
                                gr.Markdown("""
                                <div style="padding: 15px;">
                                    <h4 style="color: #667eea; margin: 0 0 10px 0;">⚙️ 第二步：选择设置</h4>
                                    <p style="color: #6b7280; margin: 0; line-height: 1.6;">
                                        <strong>选择模型</strong>：从下拉列表中选择一个预训练模型<br>
                                        <strong>选择方向</strong>：根据模型描述选择转换方向<br><br>
                                        <span style="color: #9ca3af; font-size: 0.9em;">
                                        例如：summer2winter 表示夏天→冬天，winter2summer 表示冬天→夏天
                                        </span>
                                    </p>
                                </div>
                                """)
                    
                    with gr.Row():
                        with gr.Column():
                            with gr.Box():
                                gr.Markdown("""
                                <div style="padding: 15px;">
                                    <h4 style="color: #667eea; margin: 0 0 10px 0;">✨ 第三步：开始转换</h4>
                                    <p style="color: #6b7280; margin: 0; line-height: 1.6;">
                                        点击"开始转换"按钮，等待几秒钟即可看到结果。<br><br>
                                        <strong>首次使用说明</strong>：<br>
                                        首次使用某个模型时，程序会自动下载预训练权重文件（约80MB）。下载完成后会缓存到本地，后续使用无需重新下载。
                                    </p>
                                </div>
                                """)
                        
                        with gr.Column():
                            with gr.Box():
                                gr.Markdown("""
                                <div style="padding: 15px;">
                                    <h4 style="color: #667eea; margin: 0 0 10px 0;">🔄 高级：循环一致性</h4>
                                    <p style="color: #6b7280; margin: 0; line-height: 1.6;">
                                        勾选"显示循环一致性验证"可以看到：<br>
                                        <strong>原始图像 → 转换结果 → 恢复图像</strong><br><br>
                                        这是CycleGAN的核心特性之一。如果转换质量好，恢复图像应该与原始图像非常相似。这可以帮助你评估转换的可靠性。
                                    </p>
                                </div>
                                """)
                    
                    with gr.Accordion("💡 小贴士", open=True):
                        gr.Markdown("""
                        ### 获得最佳效果的建议
                        
                        1. **选择合适的模型**：
                           - 风景照片 → 使用 season 或 art 相关模型
                           - 人物照片 → 可能效果一般，CycleGAN更适合风景和物体
                           - 油画/艺术作品 → 使用 art 相关模型的反向转换
                        
                        2. **图像尺寸**：
                           - 模型默认处理 256×256 像素的图像
                           - 上传的图像会自动调整尺寸
                           - 建议上传分辨率适中的图像（不要太大或太小）
                        
                        3. **内容匹配**：
                           - 图像内容应该与模型训练数据匹配
                           - 例如：summer2winter 模型最适合风景照片
                           - 如果内容不匹配，转换效果可能不理想
                        
                        4. **批量处理**：
                           - 如果需要处理大量图像，可以使用命令行工具
                           - 运行 `python infer.py --help` 查看详细用法
                        """)
            
            gr.HTML(HTML_FOOTER)
            
            model_dropdown.change(
                fn=self.update_direction_dropdown,
                inputs=[model_dropdown],
                outputs=[direction_dropdown]
            )
            
            def toggle_cycle_visibility(show_cycle):
                return gr.update(visible=show_cycle)
            
            show_cycle_checkbox.change(
                fn=toggle_cycle_visibility,
                inputs=[show_cycle_checkbox],
                outputs=[recovered_box]
            )
            
            def clear_all():
                return None, None, gr.update(visible=False), ""
            
            clear_btn.click(
                fn=clear_all,
                outputs=[input_image, output_transformed, recovered_box, status_text]
            )
            
            def transform_wrapper(image, model, direction, show_cycle):
                result = self.transform_image(image, model, direction, show_cycle)
                if show_cycle:
                    transformed, recovered, msg = result
                    return transformed, gr.update(visible=True, value=recovered), msg
                else:
                    transformed, msg = result
                    return transformed, gr.update(visible=False), msg
            
            transform_btn.click(
                fn=transform_wrapper,
                inputs=[input_image, model_dropdown, direction_dropdown, show_cycle_checkbox],
                outputs=[output_transformed, recovered_box, status_text]
            )
        
        return app
    
    def launch(self, 
               server_name: str = "0.0.0.0", 
               server_port: int = 7860,
               share: bool = False):
        print("=" * 60)
        print("🎨 CycleGAN Image Style Transfer - Gradio Interface")
        print("=" * 60)
        print(f"Device: {config.device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Available models: {list(PRETRAINED_MODELS_INFO.keys())}")
        print("=" * 60)
        
        app = self.create_interface(server_name, server_port)
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )

def main():
    parser = argparse.ArgumentParser(description='CycleGAN Gradio Web Interface')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', 
                        help='Server name/IP address')
    parser.add_argument('--server_port', type=int, default=7860, 
                        help='Server port')
    parser.add_argument('--share', action='store_true', 
                        help='Create a public shareable link')
    parser.add_argument('--model', type=str, default='summer2winter_yosemite', 
                        help='Default model to load')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/pretrained', 
                        help='Checkpoint directory for pretrained models')
    parser.add_argument('--list_models', action='store_true', 
                        help='List all available pretrained models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    gradio_app = GradioApp(
        checkpoint_dir=args.checkpoint_dir,
        default_model=args.model
    )
    
    gradio_app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )

if __name__ == '__main__':
    main()
