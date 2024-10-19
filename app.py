# app.py
import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import logging

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Load the model and processor once
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
logging.info("Model and Processor loaded successfully")

# Function to process the image and generate output
def analyze_image(image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "What is the brand name, manufacturing date, expiry date, quantity, category?"
                }
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

    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    output_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text

# Streamlit UI
st.title("Image Analysis for Product Information")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded successfully!")

    if st.button("Analyze"):
        # Analyze the image and display results
        output_text = analyze_image(image)
        st.subheader("Analysis Output:")
        st.write(output_text)
