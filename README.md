# LoRa Control - Dynamic Weights Controller for FORGE

This is an extension for the [FORGE Stable Diffusion web interface](https://github.com/lllyasviel/stable-diffusion-webui-forge) which implements dynamic LoRA weight control during image generation. It allows for specifying keyframe weights for LoRAs at arbitrary points during the generation process.

## Features

* Easily specify keyframe weights for LoRAs at arbitrary points
* Extends the existing LoRA syntax; no new systems to learn
* Provides separate control of LoRA weights over initial and high-res passes
* Compatible with FORGE's UnetPatcher system

## Installation

1. Clone this repository into your FORGE extensions folder:
   ```
   git clone https://github.com/yourusername/sd-webui-loractl-FORGE extensions/sd-webui-loractl-FORGE
   ```
2. Restart your FORGE instance.

## Usage

The standard LoRA syntax in FORGE is:

    <lora:network_name:strength>

This extension extends the syntax to allow for dynamic weight control:

    <lora:network_name[step1:weight1,step2:weight2,...]:base_strength>

For example:

    <lora:my_lora[0:0.5,50:1.0,100:0.75]:1.0>

This will start the LoRA at 0.5 strength, increase to full strength at 50% of the generation, and then decrease to 0.75 strength by the end.

### Step Specification

Steps can be specified in two ways:
1. As a percentage (0-100) of the total steps
2. As an absolute step number (any number greater than 100)

### High-Res Pass Control

You can use the `hr` parameter to specify weights for the high-res pass:

    <lora:network[0:0.5,100:1.0]:1.0:hr[0:1.0,100:0.5]>

This applies the LoRA with increasing strength in the base pass, and decreasing strength in the high-res pass.

## UI Controls

In the FORGE UI, you'll find a "Dynamic LoRA Weights" accordion in both txt2img and img2img tabs with the following options:

1. Enable Dynamic LoRA: Activates the dynamic LoRA weight functionality.
2. Plot LoRA Weights: When enabled, adds a graph of the LoRA weight changes to the output images.

## Examples

1. Gradual increase in LoRA strength:
   ```
   <lora:my_lora[0:0,100:1.0]:1.0>
   ```

2. LoRA warmup and cooldown:
   ```
   <lora:my_lora[0:0,25:1.0,75:1.0,100:0]:1.0>
   ```

3. Different weights for base and high-res pass:
   ```
   <lora:my_lora[0:0.5,100:1.0]:1.0:hr[0:1.0,100:0.5]>
   ```

## Compatibility Notes

This extension is specifically designed for FORGE and utilizes its UnetPatcher system. It may not be directly compatible with other Stable Diffusion Web UI implementations.

## Troubleshooting

If you encounter any issues:

1. Check the FORGE logs for any error messages related to the extension.
2. Ensure that your FORGE installation is up-to-date.
3. Verify that the LoRA syntax is correct and follows the format described above.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
