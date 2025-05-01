import os
import io
import logging
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import torch
from detoxify import Detoxify
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import json # For potentially pretty-printing dicts
import fitz # PyMuPDF
import time # To measure execution time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

TEXT_TOXICITY_THRESHOLD = 0.5

IMAGE_PROMPTS = {
    "sexually_explicit": "Is this image sexually explicit?",
    "dangerous": "Does this image depict dangerous acts or content (weapons, self-harm, illegal activities)?",
    "violence_gore": "Does this image contain graphic violence or gore?",
    "hate_symbol": "Does this image contain a common hate symbol (like a swastika)?"
}

# --- Initialize Models ---
text_moderator = None
image_processor = None
image_model = None
model_load_error = None

try:
    logging.info("Loading Detoxify model ('original')...")
    text_moderator = Detoxify('original', device='cuda' if DEVICE == 'cuda' else 'cpu')
    logging.info("Detoxify model loaded successfully.")

    logging.info("Loading PaliGemma model and processor (google/paligemma-3b-mix-448)...")
    model_id = "google/paligemma-3b-mix-448"
    image_processor = AutoProcessor.from_pretrained(model_id)
    logging.info("PaliGemma processor loaded.")

    try:
        image_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            revision="bfloat16",
            device_map=DEVICE,
        ).eval()
        logging.info("PaliGemma model loaded successfully with bfloat16 precision.")
    except (OSError, ValueError, ImportError) as e:
        logging.warning(f"Could not load PaliGemma with bfloat16 ({e}). Trying float32...")
        image_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=DEVICE
        ).eval()
        logging.info("PaliGemma model loaded successfully with float32 precision.")

except Exception as e:
    model_load_error = f"FATAL: Failed to load AI models: {e}"
    logging.error(model_load_error, exc_info=True)

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Helper Functions ---

def moderate_text(text):
    """Moderates text using Detoxify."""
    if not text_moderator:
        logging.error("Text moderation model not available.")
        return {"status": "error", "message": "Text moderation model not available."}
    if not text or not text.strip():
        logging.info("No text provided for moderation.")
        return {"status": "no_text"}

    try:
        start_time = time.time()
        logging.info(f"Moderating text block of length {len(text)} characters.")
        scores = text_moderator.predict(text) # This returns dict with numpy floats

        # -------- START FIX: Convert numpy floats to standard Python floats --------
        scores_serializable = {key: float(value) for key, value in scores.items()}
        # -------- END FIX -------------------------------------------------------

        toxicity_score = scores_serializable.get('toxicity', 0.0) # Use converted score
        label = "Toxic" if toxicity_score >= TEXT_TOXICITY_THRESHOLD else "Not Toxic"
        end_time = time.time()
        logging.info(f"Text moderation result: Label={label}, Score={toxicity_score:.4f} (Took {end_time - start_time:.2f}s)")
        return {
            "status": "processed",
            "label": label,
            "score": toxicity_score,
            "all_scores": scores_serializable # Return the serializable dict
        }
    except Exception as e:
        logging.error(f"Detoxify prediction failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Detoxify prediction failed: {e}"}


# (moderate_image function remains the same as the previous version)
def moderate_image(image_bytes):
    """Moderates an image using PaliGemma."""
    if not image_model or not image_processor:
        logging.error("Image moderation model/processor not available.")
        return {"status": "error", "message": "Image moderation model not available."}
    if not image_bytes:
        logging.warning("moderate_image called with empty image_bytes.")
        return {"status": "skipped", "message": "Empty image data provided."}

    try:
        start_time_img = time.time()
        try:
            raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logging.info(f"Processing image with size {raw_image.size}...")
        except Exception as img_err:
            logging.warning(f"Could not open image with Pillow: {img_err}")
            return {"status": "skipped", "message": f"Invalid or unsupported image format ({img_err})"}

        results = {}
        total_prompt_time = 0

        for key, prompt in IMAGE_PROMPTS.items():
            prompt_start_time = time.time()
            try:
                # Add <image> token and pass inputs correctly
                processed_prompt = "<image>\n" + prompt
                inputs = image_processor(text=processed_prompt, images=raw_image, return_tensors="pt").to(DEVICE)

                logging.info(f"[{key}] Calling model.generate()...")
                with torch.no_grad():
                    output = image_model.generate(**inputs, max_new_tokens=20, do_sample=False)
                logging.info(f"[{key}] model.generate() finished.")

                result_text = image_processor.decode(output[0], skip_special_tokens=True)
                answer = result_text[len(processed_prompt):].strip().lower()

                logging.info(f"Image prompt '{prompt}' -> Raw Answer: '{answer}'")

                if "yes" in answer: results[key] = "Yes"
                elif "no" in answer: results[key] = "No"
                else: results[key] = f"Uncertain ({answer[:30]}{'...' if len(answer)>30 else ''})"

            except Exception as model_err:
                 logging.error(f"Error processing prompt '{key}' for an image: {model_err}", exc_info=True)
                 results[key] = f"Error ({type(model_err).__name__})" # Show error type
            finally:
                prompt_end_time = time.time()
                duration = prompt_end_time - prompt_start_time
                total_prompt_time += duration
                logging.info(f"[{key}] Moderation prompt took {duration:.2f}s")

        end_time_img = time.time()
        logging.info(f"Image moderation final results: {results} (Total prompt time: {total_prompt_time:.2f}s, Total image time: {end_time_img - start_time_img:.2f}s)")
        return {"status": "processed", "results": results}

    except Exception as e:
        logging.error(f"General error during image moderation function: {e}", exc_info=True)
        return {"status": "error", "message": f"Image processing failed: {e}"}


# (extract_content_from_pdf function remains the same as the previous version)
def extract_content_from_pdf(pdf_bytes):
    """Extracts text and image bytes from a given PDF byte stream using PyMuPDF."""
    all_text = ""
    images_info = []
    doc = None

    try:
        start_time = time.time()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logging.info(f"Processing PDF with {doc.page_count} pages.")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_num_human = page_num + 1

            # Extract Text
            page_text = page.get_text("text")
            if page_text: all_text += page_text + "\n"

            # Extract Images
            image_list = page.get_images(full=True)
            if image_list: logging.info(f"Found {len(image_list)} image references on page {page_num_human}.")

            for img_index, img_info in enumerate(image_list):
                img_index_human = img_index + 1
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    if image_bytes:
                        images_info.append({
                            "page_num": page_num_human,
                            "img_index": img_index_human,
                            "image_bytes": image_bytes
                        })
                        logging.info(f"Successfully extracted image {img_index_human} (xref {xref}, ext: {image_ext}) from page {page_num_human}.")
                    else:
                         logging.warning(f"Extracted empty image data for image {img_index_human} (xref {xref}) on page {page_num_human}.")
                except Exception as img_e:
                    logging.error(f"Failed to extract image {img_index_human} (xref {xref}) from page {page_num_human}: {img_e}", exc_info=True)

        end_time = time.time()
        logging.info(f"Finished PDF extraction. Total text length: {len(all_text)}, Total images extracted: {len(images_info)} (Took {end_time - start_time:.2f}s)")
        return all_text.strip(), images_info

    except fitz.fitz.FitzError as fe:
        logging.error(f"PyMuPDF error processing PDF: {fe}", exc_info=True)
        raise ValueError(f"Failed to process PDF (PyMuPDF error): {fe}")
    except Exception as e:
        logging.error(f"General error processing PDF: {e}", exc_info=True)
        raise ValueError(f"An unexpected error occurred while processing the PDF: {e}")
    finally:
        if doc:
            doc.close()
            logging.info("Closed PDF document.")


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Displays the main upload page."""
    logging.info("Serving index page.")
    if model_load_error:
        flash(f"Warning: {model_load_error}. Moderation may not be available.", "error")
    return render_template('index.html',
                           error=None,
                           uploaded_filename=None,
                           text_results=None,
                           image_results=None)

@app.route('/moderate', methods=['POST'])
def moderate():
    """Handles the PDF file upload, extraction, moderation, and displays results."""
    logging.info("--- Received POST request to /moderate ---")
    start_request_time = time.time()

    if model_load_error:
        logging.error("Attempted moderation while models failed to load.")
        return render_template('index.html', error=model_load_error, uploaded_filename=None)

    pdf_file = request.files.get('pdf_input')
    overall_error = None
    text_moderation_result = None
    image_moderation_results = []
    uploaded_filename = None

    if not pdf_file or pdf_file.filename == '':
        overall_error = "No PDF file selected. Please upload a file."
        logging.warning("Moderation attempt without file upload.")
    elif not pdf_file.filename.lower().endswith('.pdf'):
         overall_error = "Invalid file type. Only PDF files (.pdf) are accepted."
         logging.warning(f"Invalid file type uploaded: {pdf_file.filename}")
         uploaded_filename = pdf_file.filename
    else:
        uploaded_filename = pdf_file.filename
        logging.info(f"Processing uploaded PDF: {uploaded_filename}")
        try:
            pdf_bytes = pdf_file.read()
            if not pdf_bytes: raise ValueError("Uploaded PDF file is empty.")

            # 1. Extract Content
            logging.info("Starting PDF content extraction...")
            extracted_text, extracted_images_info = extract_content_from_pdf(pdf_bytes)
            logging.info("PDF content extraction finished.")

            # 2. Moderate Text
            if extracted_text:
                logging.info("Starting text moderation...")
                text_moderation_result = moderate_text(extracted_text) # This now returns serializable scores
                if text_moderation_result.get("status") == "error":
                    logging.error(f"Text moderation failed: {text_moderation_result.get('message')}")
                    overall_error = (overall_error or "") + f" [Text moderation error: {text_moderation_result.get('message')}]"
                logging.info("Text moderation finished.")
            else:
                 logging.info("No text found in PDF to moderate.")
                 text_moderation_result = {"status": "no_text"}

            # 3. Moderate Images
            if extracted_images_info:
                logging.info(f"Starting image moderation for {len(extracted_images_info)} images...")

                # -------- Image Moderation Control --------
                # Set this to False to skip slow image moderation during testing
                ENABLE_IMAGE_MODERATION = True # <-- SET TO True TO ENABLE IMAGE MODERATION
                # ------------------------------------------

                if ENABLE_IMAGE_MODERATION:
                    logging.info("Image moderation is ENABLED.")
                    for img_info in extracted_images_info:
                        img_bytes = img_info["image_bytes"]
                        page_num = img_info["page_num"]
                        img_idx = img_info["img_index"]

                        logging.info(f"Moderating image {img_idx} from page {page_num}...")
                        img_mod_result = moderate_image(img_bytes)

                        img_mod_result["page_number"] = page_num
                        img_mod_result["image_index"] = img_idx
                        image_moderation_results.append(img_mod_result)

                        if img_mod_result.get("status") == "error":
                             err_msg = img_mod_result.get('message', 'Unknown image error')
                             logging.error(f"Error moderating image {img_idx} on page {page_num}: {err_msg}")
                             overall_error = (overall_error or "") + f" [Image {img_idx} (Page {page_num}) error: {err_msg}]"
                        elif img_mod_result.get("status") == "skipped":
                             warn_msg = img_mod_result.get('message', 'Skipped image')
                             logging.warning(f"Skipped image {img_idx} on page {page_num}: {warn_msg}")
                    logging.info("Image moderation processing loop finished.")
                else:
                    logging.warning("Image moderation is DISABLED by ENABLE_IMAGE_MODERATION flag.")
                    # Optionally add a placeholder message if needed for the template
                    # image_moderation_results = [{"status": "skipped", "message": "Image moderation disabled."}]


            else:
                 logging.info("No images found in PDF to moderate.")

        except ValueError as ve:
            logging.error(f"Value error during PDF processing for {uploaded_filename}: {ve}", exc_info=True)
            overall_error = str(ve)
            text_moderation_result = None
            image_moderation_results = []
        except Exception as e:
            logging.error(f"Unexpected error during moderation process for {uploaded_filename}: {e}", exc_info=True)
            overall_error = f"An unexpected error occurred during processing: {e}"
            text_moderation_result = None
            image_moderation_results = []

    # Log total time and render
    end_request_time = time.time()
    logging.info(f"--- Request finished. Total time: {end_request_time - start_request_time:.2f}s. Rendering template with results for {uploaded_filename} ---")
    return render_template(
        'index.html',
        error=overall_error,
        uploaded_filename=uploaded_filename,
        text_results=text_moderation_result,
        image_results=image_moderation_results
    )

# --- Run the Flask App ---
if __name__ == '__main__':
    # Run WITHOUT debug mode to prevent auto-restarts on long requests
    logging.info("Starting Flask app without debug mode (use Ctrl+C to stop).")
    app.run(debug=False, host='0.0.0.0', port=5000)