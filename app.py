# Install required libraries
!pip install git+https://github.com/huggingface/transformers
!pip install gradio

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import gradio as gr

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

# Define the function to handle the prediction
def process_image(image):
    # The fixed question for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": "What brand name, manufacturing date, expiry date, category, MRP, weight, volume?"
                }
            ]
        }
    ]

    # Process the image and prompt
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Generate the model output
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=process_image, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="Product Information Extractor",
    description="Upload an image and get the brand name, manufacturing date, expiry date, category, MRP, weight, and volume."
)

# Launch the Gradio app
interface.launch()
