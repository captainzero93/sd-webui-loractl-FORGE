import re
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from modules import scripts, shared
from modules.processing import StableDiffusionProcessing
from modules.extra_networks import parse_prompt
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Dynamic LoRA Weights script for Forge is being loaded")

def parse_dynamic_prompt(prompt: str) -> Tuple[str, Dict[str, Dict[str, float]]]:
    dynamic_loras = {}
    new_prompt = prompt

    def replace_lora(match):
        full_match = match.group(0)
        lora_name = match.group(1)
        params = match.group(2)

        if '[' in params and ']' in params:
            base_strength, dynamic_part = params.split('[')
            base_strength = float(base_strength.strip())
            instructions = parse_dynamic_instructions(dynamic_part.strip('[]'))
            dynamic_loras[lora_name] = {'base': base_strength, 'dynamic': instructions}
            return f"<lora:{lora_name}:{base_strength}>"
        return full_match

    new_prompt = re.sub(r'<lora:([^:>]+):([^>]+)>', replace_lora, new_prompt)
    return new_prompt, dynamic_loras

def parse_dynamic_instructions(instruction_str: str) -> Dict[str, float]:
    instructions = {}
    for instruction in instruction_str.split(','):
        step, weight = instruction.split(':')
        instructions[step] = float(weight)
    return instructions

def calculate_dynamic_strength(instructions: Dict[str, float], base_strength: float, current_step: int, total_steps: int) -> float:
    if not instructions:
        return base_strength

    normalized_step = current_step / total_steps * 100
    steps = sorted(map(float, instructions.keys()))

    if normalized_step <= steps[0]:
        return instructions[str(steps[0])] * base_strength
    if normalized_step >= steps[-1]:
        return instructions[str(steps[-1])] * base_strength

    for i, step in enumerate(steps):
        if step > normalized_step:
            prev_step, next_step = steps[i-1], step
            prev_weight = instructions[str(prev_step)]
            next_weight = instructions[str(next_step)]
            break

    weight_range = next_weight - prev_weight
    step_progress = (normalized_step - prev_step) / (next_step - prev_step)
    interpolated_weight = prev_weight + (weight_range * step_progress)

    return interpolated_weight * base_strength

class DynamicLoRAForForge(scripts.Script):
    def __init__(self):
        super().__init__()
        self.dynamic_loras: Dict[str, Dict[str, Dict[str, float]]] = {}

    def title(self):
        return "Dynamic LoRA Weights for Forge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enable Dynamic LoRA', value=True)
            plot_weights = gr.Checkbox(label='Plot LoRA Weights', value=False)
        return [enabled, plot_weights]

    def process(self, p: StableDiffusionProcessing, enabled, plot_weights):
        if not enabled:
            return

        self.dynamic_loras.clear()

        p.prompt, dynamic_loras_positive = parse_dynamic_prompt(p.prompt)
        p.negative_prompt, dynamic_loras_negative = parse_dynamic_prompt(p.negative_prompt)

        self.dynamic_loras['positive'] = dynamic_loras_positive
        self.dynamic_loras['negative'] = dynamic_loras_negative

        # Store original extra_network_data
        self.original_extra_network_data = p.extra_network_data

        total_steps = p.steps

        def dynamic_callback(step: int, x):
            current_step = step + 1  # step is 0-indexed, but we want 1-indexed
            self.apply_dynamic_weights(p, current_step, total_steps)
            return x

        # Add our callback to the list of callbacks
        if not hasattr(p, 'callback_map'):
            p.callback_map = {}
        p.callback_map['dynamic_lora'] = dynamic_callback

    def apply_dynamic_weights(self, p: StableDiffusionProcessing, current_step: int, total_steps: int):
        new_extra_network_data = {}
        for network_type, network_params in self.original_extra_network_data.items():
            new_params = []
            for param in network_params:
                if network_type == 'lora' and param.items[0] in self.dynamic_loras['positive']:
                    lora_name = param.items[0]
                    lora_info = self.dynamic_loras['positive'][lora_name]
                    dynamic_strength = calculate_dynamic_strength(
                        lora_info['dynamic'],
                        lora_info['base'],
                        current_step,
                        total_steps
                    )
                    new_params.append(type(param)([lora_name, str(dynamic_strength)]))
                else:
                    new_params.append(param)
            new_extra_network_data[network_type] = new_params
        
        p.extra_network_data = new_extra_network_data

    def postprocess(self, p, processed, enabled, plot_weights):
        if not enabled:
            return

        if plot_weights:
            plot = self.make_plot(p.steps)
            if plot is not None:
                processed.images.append(plot)

        # Restore original extra_network_data
        p.extra_network_data = self.original_extra_network_data

        self.dynamic_loras.clear()

    def make_plot(self, total_steps):
        if not self.dynamic_loras:
            return None

        plt.figure(figsize=(10, 6))
        for prompt_type, loras in self.dynamic_loras.items():
            for lora_name, lora_info in loras.items():
                base_strength = lora_info['base']
                dynamic_instructions = lora_info['dynamic']
                
                steps = list(range(1, total_steps + 1))
                weights = [calculate_dynamic_strength(dynamic_instructions, base_strength, step, total_steps) for step in steps]
                
                plt.plot(steps, weights, label=f"{prompt_type} - {lora_name}")

        plt.xlabel('Step')
        plt.ylabel('LoRA Weight')
        plt.title('Dynamic LoRA Weights')
        plt.legend()
        plt.grid(True)

        # Convert plot to image
        fig = plt.gcf()
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return plot_image

# Add this line at the end of the file to ensure the script is loaded
logger.info("Registering DynamicLoRAForForge script")
