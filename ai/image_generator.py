from diffusers import StableDiffusionPipeline
import torch
import os


class ImageGenerator:
    def __init__(self):
        # Device config for mac M1
        if torch.backends.mps.is_available(): # check if mac supports cpu acceleration
            self.device = "mps" # gpu -> faster
        else:
            self.device = "cpu" # fallback on cpu

        print(f"Using device: {self.device}")

        # Load stable diffusion model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32
        )
        self.pipeline = self.pipeline.to(self.device)

        # Create repo to save images
        if not os.path.exists("../generated_images"):
            os.makedirs("../generated_images")

    def generate(self, prompt, num_images=1, guidance_scale=7.5, num_inference_steps=20):
        """generate images from prompt"""
        print(f"Generating {num_images} image/s with prompt: '{prompt}'")

        images = self.pipeline(
            prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images

        # Save images
        saved_paths = []
        for i, image in enumerate(images):
            filename = f"generated_images/image_{hash(prompt)}_{i}.png"
            image.save(filename)
            saved_paths.append(filename)
            print(f"Immagine salvata: {filename}")

        return {
            'prompt': prompt,
            'images': images,
            'saved_paths': saved_paths,
            'num_images': len(images)
        }



if __name__ == "__main__":
    generator = ImageGenerator()

    print("=== Test Generazione Immagini (===")

    # base specific test
    prompt = "A cute cat wearing a space helmet, digital art"
    result = generator.generate(prompt, num_images=1)
    print(f"Generate {result['num_images']} images with prompt: {result['prompt']}")
    print(f"Saved in in: {result['saved_paths']}")
    print("-" * 50)

    # base specific test
    prompt = "A futuristic city in the base of mountains and a sunset in the background"
    result = generator.generate(prompt, num_images=1)
    print(f"Generate {result['num_images']} images with prompt: {result['prompt']}")
    print(f"Saved in in: {result['saved_paths']}")
    print("-" * 50)
