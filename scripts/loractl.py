import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from modules import scripts, script_callbacks, extra_networks
from typing import Dict, List, Tuple
import logging
import re

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

    logger.debug(f"Calculated dynamic strength: {interpolated_weight * base_strength} for step {current_step}/{max_steps}")
    return interpolated_weight * base_strength

def parse_dynamic_instructions(instruction_str: str) -> Dict[int, float]:
    instructions = {}
    for instruction in instruction_str.split(','):
        step, weight = instruction.split(':')
        instructions[int(step)] = float(weight)
    return instructions

class DynamicLoRAForForge(scripts.Script):
    def __init__(self):
        super().__init__()
        self.weight_history: Dict[str, List[Tuple[int, float]]] = {}
        self.dynamic_loras: Dict[str, Dict[int, float]] = {}
        self.current_step = 0
        self.total_steps = 0
        self.original_activate = None
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

        self.weight_history.clear()
        self.dynamic_loras.clear()
        self.current_step = 0
        self.total_steps = p.steps

        # Preprocess LoRA instructions
        for prompt in [p.prompt, p.negative_prompt]:
            loras = re.findall(r'<lora:(.*?)>', prompt)
            for lora in loras:
                parts = lora.split(':')
                if len(parts) > 1 and '[' in parts[1]:
                    base_name = parts[0]
                    instruction_part = parts[1].strip('[]')
                    instructions = parse_dynamic_instructions(instruction_part)
                    self.dynamic_loras[base_name] = instructions
                    # Replace dynamic instructions with a placeholder
                    new_lora = f"{base_name}:1.0"
                    prompt = prompt.replace(f"<lora:{lora}>", f"<lora:{new_lora}>")

        p.prompt = prompt
        p.negative_prompt = prompt

        # Store the original activate method
        self.original_activate = extra_networks.extra_networks_dict['lora'].activate

        # Monkey patch extra_networks_lora.activate
        def wrapped_activate(self_lora, p, params):
            lora_name = params.positional[0]
            if lora_name in self.dynamic_loras:
                # Apply dynamic strength
                dynamic_strength = calculate_dynamic_strength(
                    self.dynamic_loras[lora_name],
                    float(params.positional[1]) if len(params.positional) > 1 else 1.0,
                    self.current_step,
                    self.total_steps
                )
                params.positional[1] = str(dynamic_strength)

                if lora_name not in self.weight_history:
                    self.weight_history[lora_name] = []
                self.weight_history[lora_name].append((self.current_step, dynamic_strength))

            return self.original_activate(p, params)

        extra_networks.extra_networks_dict['lora'].activate = wrapped_activate

        # Monkey patch the sampler to track steps
        original_callback = p.callback
        def sampler_callback(step, *args, **kwargs):
            self.current_step = step
            if original_callback:
                original_callback(step, *args, **kwargs)

        p.callback = sampler_callback

    def postprocess(self, p, processed, enabled, plot_weights):
        logger.info(f"postprocess called with enabled={enabled}, plot_weights={plot_weights}")
        logger.info(f"Final current_step: {self.current_step}")
        logger.info(f"Final weight history: {self.weight_history}")

        if enabled and plot_weights:
            plot = self.make_plot()
            if plot is not None:
                processed.images.append(plot)
            else:
                logger.warning("Plot was None, not appending to processed images")

        # Restore original methods
        if self.original_activate is not None:
            extra_networks.extra_networks_dict['lora'].activate = self.original_activate
            logger.info("Restored original lora activate method")
        else:
            logger.warning("Original activate method was not stored, unable to restore")

        self.weight_history.clear()
        self.dynamic_loras.clear()

    def make_plot(self):
        logger.info(f"make_plot called. Weight history: {self.weight_history}")
        if not self.weight_history:
            logger.warning("No weight history to plot")
            return None

        plt.figure(figsize=(10, 6))
        for lora_name, history in self.weight_history.items():
            steps, weights = zip(*history)
            plt.plot(steps, weights, label=lora_name)
            logger.info(f"Plotting {lora_name}: steps={steps}, weights={weights}")

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

        logger.info("Plot created successfully")
        return plot_image

# Add this line at the end of the file to ensure the script is loaded
logger.info("Registering DynamicLoRAForForge script")
