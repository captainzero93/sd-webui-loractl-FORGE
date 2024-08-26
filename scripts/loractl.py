import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from modules import scripts, script_callbacks
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Dynamic LoRA Weights script is being loaded")

def calculate_dynamic_strength(instructions: Dict[int, float], base_strength: float, current_step: int, max_steps: int) -> float:
    if not instructions:
        return base_strength
    
    normalized_step = current_step / max_steps * 100  # Convert to percentage
    steps = sorted(instructions.keys())
    
    if normalized_step <= steps[0]:
        return instructions[steps[0]] * base_strength
    if normalized_step >= steps[-1]:
        return instructions[steps[-1]] * base_strength
    
    # Find the two nearest instruction points
    for i, step in enumerate(steps):
        if step > normalized_step:
            prev_step, next_step = steps[i-1], step
            prev_weight, next_weight = instructions[prev_step], instructions[step]
            break
    
    # Linear interpolation
    weight_range = next_weight - prev_weight
    step_progress = (normalized_step - prev_step) / (next_step - prev_step)
    interpolated_weight = prev_weight + (weight_range * step_progress)
    
    return interpolated_weight * base_strength

def parse_lora_name(lora_name: str) -> Tuple[str, Dict[int, float]]:
    if '[' not in lora_name:
        return lora_name, {}
    
    base_name, instructions = lora_name.split('[')
    instructions = instructions.rstrip(']')
    parsed_instructions = {}
    for instruction in instructions.split(','):
        step, weight = instruction.split(':')
        parsed_instructions[int(step)] = float(weight)
    
    return base_name.strip(), parsed_instructions

def dynamic_lora_application(unet, lora_name: str, strength: float, step: int, max_steps: int):
    base_name, instructions = parse_lora_name(lora_name)
    dynamic_strength = calculate_dynamic_strength(instructions, strength, step, max_steps)
    
    # FORGE uses UnetPatcher, so we need to update the LoRA application
    if hasattr(unet, 'set_lora_scale'):
        # Assuming UnetPatcher has a method to set LoRA scale
        unet.set_lora_scale(base_name, dynamic_strength)
    else:
        # Fallback method if set_lora_scale is not available
        for name, module in unet.named_modules():
            if hasattr(module, 'set_lora_scale'):
                module.set_lora_scale(base_name, dynamic_strength)
    
    return dynamic_strength

class DynamicLoRAForForge(scripts.Script):
    def __init__(self):
        super().__init__()
        self.weight_history: Dict[str, List[Tuple[int, float]]] = {}
        logger.info("DynamicLoRAForForge instance created")

    def title(self):
        return "Dynamic LoRA Weights"

    def show(self, is_img2img):
        logger.info(f"show method called with is_img2img={is_img2img}")
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        logger.info("ui method called")
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enable Dynamic LoRA', value=False)
            plot_weights = gr.Checkbox(label='Plot LoRA Weights', value=False)

        return [enabled, plot_weights]

    def process(self, p, enabled, plot_weights):
        logger.info(f"process called with enabled={enabled}, plot_weights={plot_weights}")
        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        original_apply_lora = unet.apply_lora

        def wrapped_apply_lora(lora_name: str, strength: float):
            logger.info(f"Applying LoRA: {lora_name} with strength {strength}")
            current_step = p.step
            max_steps = p.steps
            dynamic_strength = dynamic_lora_application(unet, lora_name, strength, current_step, max_steps)
            
            base_name, _ = parse_lora_name(lora_name)
            if base_name not in self.weight_history:
                self.weight_history[base_name] = []
            self.weight_history[base_name].append((current_step, dynamic_strength))
            
            return original_apply_lora(lora_name, dynamic_strength)

        unet.apply_lora = wrapped_apply_lora

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            dynamic_lora_enabled=enabled,
            plot_lora_weights=plot_weights,
        ))

    def postprocess(self, p, processed, enabled, plot_weights):
        logger.info(f"postprocess called with enabled={enabled}, plot_weights={plot_weights}")
        if enabled and plot_weights:
            plot = self.make_plot()
            processed.images.append(plot)

        # Restore original apply_lora method
        if hasattr(p.sd_model.forge_objects.unet, 'apply_lora'):
            del p.sd_model.forge_objects.unet.apply_lora

        self.weight_history.clear()

    def make_plot(self):
        plt.figure(figsize=(10, 6))
        for lora_name, history in self.weight_history.items():
            steps, weights = zip(*history)
            plt.plot(steps, weights, label=lora_name)
        
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

# Remove the unsupported callback
logger.info("Registering DynamicLoRAForForge script")
