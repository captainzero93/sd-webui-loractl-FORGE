import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from modules import scripts
from typing import Dict, List, Tuple

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
    
    # Apply the LoRA with the calculated strength
    # This is a placeholder - you'll need to implement the actual LoRA application
    # based on how FORGE handles LoRAs
    unet.apply_lora(base_name, dynamic_strength)
    
    return dynamic_strength

class DynamicLoRAForForge(scripts.Script):
    sorting_priority = 13  # It will be the 13th item on UI.

    def __init__(self):
        self.weight_history: Dict[str, List[Tuple[int, float]]] = {}
        super().__init__()

    def title(self):
        return "Dynamic LoRA Weights"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enable Dynamic LoRA', value=False)
            plot_weights = gr.Checkbox(label='Plot LoRA Weights', value=False)

        return enabled, plot_weights

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, plot_weights = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        original_apply_lora = unet.apply_lora

        def wrapped_apply_lora(lora_name: str, strength: float):
            current_step = p.step
            max_steps = p.steps
            dynamic_strength = dynamic_lora_application(unet, lora_name, strength, current_step, max_steps)
            
            # Record the weight for plotting
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

        return

    def postprocess(self, p, processed, *args):
        enabled, plot_weights = args

        if enabled and plot_weights:
            plot = self.make_plot()
            processed.images.append(plot)

        # Clear the weight history for the next generation
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
