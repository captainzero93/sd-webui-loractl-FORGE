import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from modules import scripts, script_callbacks
from modules.processing import StableDiffusionProcessing
from typing import Dict, List, Tuple
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Dynamic LoRA Weights script for Forge is being loaded")

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
        self.dynamic_loras: Dict[str, Tuple[float, Dict[int, float]]] = {}
        self.current_step = 0
        self.total_steps = 0
        logger.info("DynamicLoRAForForge instance created")

    def title(self):
        return "Dynamic LoRA Weights for Forge"

    def show(self, is_img2img):
        logger.info(f"show method called with is_img2img={is_img2img}")
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        logger.info("ui method called")
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enable Dynamic LoRA', value=False)
            plot_weights = gr.Checkbox(label='Plot LoRA Weights', value=False)

        return [enabled, plot_weights]

    def process(self, p: StableDiffusionProcessing, enabled, plot_weights):
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
                    strength_part, instruction_part = parts[1].split('[')
                    base_strength = float(strength_part)
                    instructions = parse_dynamic_instructions(instruction_part.strip('[]'))
                    self.dynamic_loras[base_name] = (base_strength, instructions)
                    # Keep the original LoRA instruction in the prompt
                    # Forge will handle the initial loading of the LoRA

        # Monkey patch the sampler to track steps and apply dynamic weights
        original_callback = p.callback
        def sampler_callback(step, *args, **kwargs):
            self.current_step = step
            self.apply_dynamic_weights(p)
            if original_callback:
                original_callback(step, *args, **kwargs)

        p.callback = sampler_callback

    def apply_dynamic_weights(self, p: StableDiffusionProcessing):
        for lora_name, (base_strength, instructions) in self.dynamic_loras.items():
            dynamic_strength = calculate_dynamic_strength(
                instructions,
                base_strength,
                self.current_step,
                self.total_steps
            )
            
            # Update LoRA weight in Forge's model
            if hasattr(p, 'sd_model') and hasattr(p.sd_model, 'set_lora_weight'):
                p.sd_model.set_lora_weight(lora_name, dynamic_strength)
            else:
                logger.warning(f"Unable to set LoRA weight for {lora_name}. Method not found.")

            if lora_name not in self.weight_history:
                self.weight_history[lora_name] = []
            self.weight_history[lora_name].append((self.current_step, dynamic_strength))

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

        # Reset LoRA weights to their original values
        if hasattr(p, 'sd_model') and hasattr(p.sd_model, 'set_lora_weight'):
            for lora_name, (base_strength, _) in self.dynamic_loras.items():
                p.sd_model.set_lora_weight(lora_name, base_strength)

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
