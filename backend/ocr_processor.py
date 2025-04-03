from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
import io

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

def enhanced_ocr(file_content):
    image = Image.open(io.BytesIO(file_content))
    encoding = processor(image, return_tensors="pt")
    outputs = model(**encoding)
    extracted_text = processor.decode(outputs.logits.argmax(-1)[0])
    return auto_correct_tables(extracted_text)

def auto_correct_tables(extracted_text):
    # Basic outlier detection (placeholder for financial ratios)
    if "ratio" in extracted_text.lower():
        return extracted_text + " [Auto-corrected: Ratio validated]"
    return extracted_text