import os
import sys
import argparse
from typing import Dict, Optional, Tuple

import gradio as gr
from PIL import Image

from infer import CycleGANInferencer, PRETRAINED_MODELS_INFO, list_available_models
from config import config

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
                return None, None, "Please upload an image first"
            return None, "Please upload an image first"
        
        try:
            inferencer = self._get_inferencer(model_name)
            
            directions = self.model_directions.get(model_name, {})
            if directions['AtoB'] == direction_display:
                direction = 'AtoB'
            else:
                direction = 'BtoA'
            
            if show_cycle:
                transformed, recovered = inferencer.transform_with_cycle(image, direction)
                message = f"Transformation completed! Model: {self.model_descriptions[model_name]}, Direction: {direction_display}"
                return transformed, recovered, message
            else:
                transformed = inferencer.transform(image, direction)
                message = f"Transformation completed! Model: {self.model_descriptions[model_name]}, Direction: {direction_display}"
                return transformed, message
                
        except Exception as e:
            error_msg = f"Error during transformation: {str(e)}"
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
    
    def create_interface(self, server_name: str = "0.0.0.0", server_port: int = 7860):
        model_choices = list(self.model_descriptions.keys())
        default_model = self.default_model if self.default_model in model_choices else model_choices[0]
        
        initial_directions = self.model_directions.get(default_model, {})
        initial_direction_choices = {
            initial_directions['AtoB']: 'AtoB',
            initial_directions['BtoA']: 'BtoA'
        } if initial_directions else {}
        initial_direction = list(initial_direction_choices.keys())[0] if initial_direction_choices else ''
        
        with gr.Blocks(title="CycleGAN Image Style Transfer") as app:
            gr.Markdown("""
            # 🌟 CycleGAN 图像风格转换系统
            
            基于 PyTorch 的无监督图像风格转换，无需成对训练数据即可实现照片转油画、夏天转冬天等风格转换。
            
            **支持的模型：**
            - summer2winter_yosemite: 夏天 → 冬天 (优胜美地)
            - apple2orange: 苹果 → 橙子
            - horse2zebra: 马 → 斑马
            - monet2photo: 莫奈风格 → 照片
            - cezanne2photo: 塞尚风格 → 照片
            - ukiyo-e2photo: 浮世绘 → 照片
            - vangogh2photo: 梵高风格 → 照片
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 上传图像")
                    input_image = gr.Image(type="pil", label="输入图像", height=300)
                    
                    gr.Markdown("### ⚙️ 转换设置")
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=default_model,
                        label="选择预训练模型",
                        info="选择要使用的CycleGAN模型"
                    )
                    
                    direction_dropdown = gr.Dropdown(
                        choices=list(initial_direction_choices.keys()),
                        value=initial_direction,
                        label="转换方向",
                        info="选择风格转换的方向"
                    )
                    
                    show_cycle_checkbox = gr.Checkbox(
                        label="显示循环一致性结果",
                        value=False,
                        info="同时显示原始 → 转换 → 恢复的完整循环"
                    )
                    
                    transform_btn = gr.Button("🎨 开始转换", variant="primary")
                    clear_btn = gr.Button("🔄 清空", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎭 转换结果")
                    output_transformed = gr.Image(type="pil", label="转换结果", height=300)
                    
                    output_recovered = gr.Image(
                        type="pil", 
                        label="循环恢复结果 (原始图像)",
                        height=300,
                        visible=False
                    )
                    
                    status_text = gr.Textbox(
                        label="状态信息",
                        lines=2,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 📋 使用说明
                    
                    1. **上传图像**: 点击左侧的图像上传区域，选择您要转换的图片
                    2. **选择模型**: 从下拉菜单中选择一个预训练模型
                    3. **选择方向**: 根据模型描述选择转换方向 (如 summer → winter 或 winter → summer)
                    4. **可选设置**: 勾选"显示循环一致性结果"可以同时看到恢复的图像
                    5. **开始转换**: 点击"开始转换"按钮，等待几秒钟即可看到结果
                    
                    ### 📁 模型信息
                    
                    首次使用某个模型时，程序会自动下载预训练权重文件 (约 80MB)，下载后会缓存到本地，后续使用无需重新下载。
                    """)
            
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
                outputs=[output_recovered]
            )
            
            def clear_all():
                return None, None, None, ""
            
            clear_btn.click(
                fn=clear_all,
                outputs=[input_image, output_transformed, output_recovered, status_text]
            )
            
            def transform_wrapper(image, model, direction, show_cycle):
                result = self.transform_image(image, model, direction, show_cycle)
                if show_cycle:
                    transformed, recovered, msg = result
                    return transformed, recovered, msg
                else:
                    transformed, msg = result
                    return transformed, None, msg
            
            transform_btn.click(
                fn=transform_wrapper,
                inputs=[input_image, model_dropdown, direction_dropdown, show_cycle_checkbox],
                outputs=[output_transformed, output_recovered, status_text]
            )
            
            gr.Markdown("""
            ---
            
            **基于项目**: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
            
            **技术栈**: PyTorch + torchvision + Pillow + matplotlib + Gradio
            """)
        
        return app
    
    def launch(self, 
               server_name: str = "0.0.0.0", 
               server_port: int = 7860,
               share: bool = False):
        print("=" * 60)
        print("CycleGAN Image Style Transfer - Gradio Interface")
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
