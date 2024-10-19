import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# Check if a GPU is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto" if device == "cuda" else None,  # Use auto device map only if CUDA is available
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Define a function to process the image and generate the output
def generate_output(image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is product size, manufacturing date, expiry date, brand name, box size, product description?"}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(device)  # Use the determined device (cuda or cpu)

    output_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text

# Streamlit app layout
st.title("Product Information Extractor")
st.write("Upload an image of the product:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Generate output
    output_text = generate_output(image)
    
    # Display output
    st.write("Generated Output:")
    st.write(output_text)

# Additional Streamlit configurations (if needed)
