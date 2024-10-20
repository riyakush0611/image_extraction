# Fine-Tuning and OCR Extraction using Qwen2-VL


This project demonstrates the fine-tuning of the Qwen2-VL model on a custom dataset for OCR extraction tasks. The objective is to extract important product information such as:

Brand Name
MRP
Expiry Date
Manufacturing Details
Category
Weight/Volume
The model is optimized to work with product images, where it automatically identifies and extracts this textual information.

**Solution Workflow**
1. Dataset Preparation
We collected and preprocessed a custom dataset comprising product images for fine-tuning the model. The images include the following features.

**Brand labels**
Product packaging with MRP and other details
Preprocessing:
To ensure data consistency and format, we run a preprocessing script, preprocess_train.py, which cleans and standardizes the training data. The images are processed for optimal input into the model, with labels aligned for OCR.

2. **JSON Creation**
The prepared dataset is converted into a structured format (JSON) compatible with the fine-tuning process. This JSON file contains image paths, along with bounding boxes or annotations for target text extraction (e.g., MRP, brand name, etc.).

3. # Fine-Tuning the Model
We utilize the Qwen2-VL model, a vision-language model from Hugging Face, for OCR tasks. Fine-tuning is performed using the notebook finetune.ipynb. The training settings are configured via the WebUI to:

**Adapt the model for product-specific text extraction**
Optimize for accuracy in recognizing brand-specific information
The fine-tuning leverages LoRA for efficient low-rank adaptation, ensuring the model learns from our custom dataset while maintaining general OCR capabilities.

4. **Inference on Fine-Tuned Model**
Once the model is fine-tuned, we perform inference using inference.py. This script takes an input image of a product and outputs the extracted information (e.g., brand name, MRP, expiry date).

**Experimentation Details**
Model Used: Qwen2-VL-7B-Instruct
Finetuning Framework: We fine-tuned the model specifically to extract text related to product labels and packaging.
Tools and Techniques: The model is optimized using Low-Rank Adaptation (LoRA) for efficient fine-tuning. 

![kt](https://github.com/user-attachments/assets/076e9734-b783-4401-be86-2a4a55771c6f)

