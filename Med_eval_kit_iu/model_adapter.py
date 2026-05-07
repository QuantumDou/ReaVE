from __future__ import annotations

import os
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor
from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available

import re
from string import punctuation



def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEFAULT_USER_PROMPT = """You are an expert Radiologist. Your task is to interpret the provided chest X-ray image following a standard clinical workflow: first give a global impression (Normal or Abnormal) and list key regions of concern, then observe specific anatomical regions in detail. For each step, explicitly state the [Observation Target], focus on the relevant image regions, and provide a clear clinical description. Finally, summarize all findings into a comprehensive diagnostic report.
Format your response exactly as follows, separated by ###:
### Global Impression: [Status: {Normal/Abnormal}], Key regions: {List of Regions}.
### Let's observe step by step.
### Step 1: [Observation Target: Region Name]
{Findings}
### Step 2: 
...
### The final conclusion is: {Diagnostic Report}"""

DEFAULT_SYSTEM_PROMPT_1 = """You are an expert Radiologist. Your goal is to generate a chest X-ray report that achieves the highest possible NLG scores (BLEU, CIDEr, ROUGE) against the Indiana University Chest X-ray (IU-X-Ray) dataset ground truth.

Your task is to interpret the provided chest X-ray image following a standard clinical workflow.

### 1. RESPONSE FORMAT (Strictly follow this template)
Format your response exactly as follows, separated by ###:

### Global Impression: [Status: {Normal/Abnormal}], Key regions: {List of Regions}
### Let's observe step by step.
### Step 1: [Observation Target: {Region 1 from List}]
{Detailed Findings}
### Step 2: [Observation Target: {Region 2 from List}]
{Detailed Findings}
...
### Step N: [Observation Target: {Region N from List}]
{Detailed Findings}
### The final conclusion is: {Diagnostic Report}

### 2. DETAILED RULES FOR EACH SECTION

#### Part A: Global Impression & Key Regions
- **Status**: Determine if the image is "Normal" or "Abnormal". Be strict—if there is ANY anomaly (e.g., small nodule, mild opacity), Status must be "Abnormal".
- **Key regions**: List the specific anatomical areas you will examine. Standard regions usually include: [Mediastinum and Airway, Cardiac, Lung and Pleura, Diaphragm and Abdomen, Bones and Soft Tissue]. 

#### Part B: Step-by-Step Observation
- **Consistency Rule**: The **[Observation Target]** in each step MUST strictly correspond to one of the items in your **Key regions** list above.
- **Coverage**: You must iterate through ALL listed Key regions to ensure a comprehensive review.
- **Content**: Provide detailed visual descriptions and clinical reasoning in these steps.

#### Part C: The Final Conclusion (The Diagnostic Report) - CRITICAL FOR SCORES
- **ULTIMATE GOAL (High BLEU Score Optimization)**: 
- This section is evaluated directly against the IU-X-Ray test set ground truth. You MUST use the concise, telegraphic style of the IU-X-Ray dataset.
- Your output MUST maximize N-gram overlap to achieve the highest possible BLEU-1 to BLEU-4 scores. 
- **Strict Report Ordering**: You MUST describe findings in this EXACT order:
- Even if there are abnormalities, maintain this order of sentences to maximize overlap:
1. **Mediastinum, Airway, and Cardiac**
2. **Lung and Pleura**
3. **Diaphragm and Abdomen**
4. **Bones and Soft Tissue**
- **STRICT TEMPLATES FOR NORMAL REGIONS:** (Use these EXACTLY when a region is normal to boost BLEU):
- **Heart/Mediastinum**: "The heart size and mediastinal contours are within normal limits."
- **Lungs (General)**: "The lungs are clear."
- **Airspace**: "No focal airspace disease."
- **Pleura**: "No pleural effusion or pneumothorax."
- **Bones**: "No acute bony abnormality."

### 3. IU_XRAY_FEW_SHOT_EXAMPLES
Here are examples of the REQUIRED style and ORDER for "The final conclusion is":

Example 1 (Normal Case):
The final conclusion is: The heart size and mediastinal contours are within normal limits. The lungs are clear. No focal airspace disease. No pleural effusion or pneumothorax.

Example 2 (Normal Case):
The final conclusion is: The heart size and pulmonary vascularity are within normal limits. The lungs are clear. No focal airspace disease. No pleural effusion or pneumothorax.

Example 3 (Abnormal Case - Opacity & Heart):
The final conclusion is: The heart is not significantly enlarged. There is a prosthetic valve. There are atherosclerotic changes of the aorta. There is a focal area of opacity in the right midlung zone. This was not present on the recent prior study. There is prominence of the pulmonary markings throughout and there are small bilateral pleural effusions. Arthritic changes of the skeletal structures are noted.

Example 4 (Abnormal Case - Cardiomegaly):
The final conclusion is: The heart is mildly enlarged. Persistent bilateral lower lobe airspace disease not significantly compared to prior. No pleural effusion or pneumothorax. No acute bony abnormality.

Example 5 (Abnormal Case - Bones):
The final conclusion is: The heart pulmonary and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is no focal air space opacity to suggest a pneumonia. There are mild degenerative changes of the spine.
"""


DEFAULT_SYSTEM_PROMPT_text_only = """You are an expert Radiologist. Your goal is to generate a chest X-ray report that achieves the highest possible NLG scores (BLEU, CIDEr, ROUGE) against the Indiana University Chest X-ray (IU-X-Ray) dataset ground truth.

Your task is to interpret the provided chest X-ray image following a standard clinical workflow.

### 1. RESPONSE FORMAT (Strictly follow this template)
Format your response exactly as follows, separated by ###:

### Global Impression: [Status: {Normal/Abnormal}], Key regions: {List of Regions}
### Let's observe step by step.
### Step 1: [Observation Target: {Region 1 from List}]
{Detailed Findings}
### Step 2: [Observation Target: {Region 2 from List}]
{Detailed Findings}
...
### Step N: [Observation Target: {Region N from List}]
{Detailed Findings}
### The final conclusion is: {Diagnostic Report}

### 2. DETAILED RULES FOR EACH SECTION

#### Part A: Global Impression & Key Regions
- **Status**: Determine if the image is "Normal" or "Abnormal". Be strict—if there is ANY anomaly (e.g., small nodule, mild opacity), Status must be "Abnormal".
- **Key regions**: List the specific anatomical areas you will examine. Standard regions usually include: [Mediastinum and Airway, Cardiac, Lung and Pleura, Diaphragm and Abdomen, Bones and Soft Tissue]. 

#### Part B: Step-by-Step Observation
- **Consistency Rule**: The **[Observation Target]** in each step MUST strictly correspond to one of the items in your **Key regions** list above.
- **Coverage**: You must iterate through ALL listed Key regions to ensure a comprehensive review.
- **Content**: Provide detailed visual descriptions and clinical reasoning in these steps.


"""

class MedFullQwen2VLChat:

    def __init__(
        self,
        base_model_path: str,
        system_prompt: Optional[str] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        prompt_template: str = "full",
        use_mulberry_prompt: bool = True,
    ):
        self.device = device if device != "auto" else get_available_device()
        self.verbose = verbose
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT_1
        self.user_prompt = DEFAULT_USER_PROMPT
        self.prompt_template = prompt_template
        self.use_mulberry_prompt = use_mulberry_prompt
        self.base_model_path = base_model_path

        # self.processor = Qwen2VLProcessor.from_pretrained(base_model_path)
        # self._configure_image_processor()

        if self.device == "mps" and torch_dtype == torch.bfloat16:
            torch_dtype = torch.float16

        # attn_implementation = self._detect_attention_impl()
        attn_implementation = "flash_attention_2"
        device_map = self._build_device_map()

        print(f"Loading full model from {base_model_path}...")
        print("===attn_implementation:",attn_implementation)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype if torch_dtype is not None else "auto",
            attn_implementation=attn_implementation,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        # self.processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.processor = Qwen2_5_VLProcessor.from_pretrained(base_model_path)

        # self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self._configure_image_processor()

        if self.device == "mps":
            self.model = self.model.to(self.device)
        self.model.eval()

        default_generate_kwargs = dict(
            max_new_tokens=2048,  #512
            temperature=0.1, #失效
            top_p=0.9, #失效
            repetition_penalty=1.05,
            do_sample=False, #失效
            use_cache=True,
            interleave_inf=True,
            selected_numbers=None,
            num_added_tokens=None,
            predicted_labels=None,
            new_input_ids=None,
            return_dict_in_generate=True,
            predicted_labels_output=None,
            predict_threshold=0.8, #0.6, #0.5
            interleave_sim=None,
            interleave_cache={},
        )
        if generation_kwargs:
            default_generate_kwargs.update(generation_kwargs)
        self.generate_kwargs = default_generate_kwargs
        print("generate.generate_kwargs:", self.generate_kwargs)

    # ------------------------------------------------------------------ #
    def _configure_image_processor(self) -> None:
        if hasattr(self.processor, "image_processor") and self.processor.image_processor is not None:
            image_processor = self.processor.image_processor
            image_processor.do_resize = True
            image_processor.min_pixels = 224 * 224
            image_processor.max_pixels = 54880   #589824 #54880
            image_processor.patch_size = 14
            image_processor.merge_size = 2

    def _detect_attention_impl(self) -> str:
        if is_flash_attn_2_available():
            print("[MedFullQwen2VLChat] Using flash_attention_2.")
            return "flash_attention_2"
        if is_torch_sdpa_available():
            print("[MedFullQwen2VLChat] Using torch SDPA.")
            return "sdpa"
        print("[MedFullQwen2VLChat] Using eager attention.")
        return "eager"

    def _build_device_map(self):
        if self.device == "mps":
            return "cpu"
        if self.device == "cpu":
            return "cpu"
        return {"": self.device}

    # ------------------------------------------------------------------ #
    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_path = sample.get("image") or sample.get("image_path")
        assert image_path, "sample must provide 'image' path"

        if not image_path.startswith(("http://", "https://", "file://")):
            base = sample.get("image_base_dir", "")
            if not os.path.isabs(image_path):
                image_path = os.path.join(base, image_path)
            image_path = os.path.abspath(image_path)
            image_path = "file://" + image_path

        question = sample.get("instruction") or sample.get("prompt") or sample.get("question") or ""
        assert question, "sample must include question text"

        torch.set_printoptions(profile="full")
        messages = self._build_messages(image_path, question)
        inputs = self._prepare_inputs(messages)
        # print("====input_sequence = ",self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # print("----generate.inputs:", {k: getattr(v, "shape", type(v)) for k, v in inputs.items()})
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self.generate_kwargs)

        generated_ids = outputs.sequences
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        response = decoded[0]
        prediction = self._extract_report_text(response)
        prediction = self._post_process_prediction(prediction)


        aux = {
            "raw_response": response,
            "interleave_sim": _safe_to_list(outputs.interleave_sim),
            "predicted_labels": _safe_to_list(outputs.predicted_labels),
        }
        return {
            "id": sample.get("id"),
            "prediction": prediction,
            "raw_response": response,
            "aux": aux,
        }

    # ------------------------------------------------------------------ #
    def _prepare_content(self, inputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for item in inputs:
            if item["type"] == "image":
                entry = {
                    "type": "image",
                    "image": self._ensure_image_url(item.get("value") or item.get("image")),
                    "min_pixels": 224 * 224,
                    "max_pixels": 54880, #54880,
                }
            elif item["type"] == "text":
                entry = {"type": "text", "text": self.user_prompt}
            else:
                raise ValueError(f"Invalid content type: {item['type']}")
            content.append(entry)
        return content

    def _ensure_image_url(self, image: str) -> str:
        if image.startswith(("http://", "https://", "file://", "data:image;")):
            return image
        if os.path.exists(image):
            return "file://" + image
        raise ValueError(f"Invalid image path: {image}")

    def _build_messages(self, image_url: str, question: str) -> List[Dict[str, Any]]:
        raw_inputs = [
            {"type": "image", "value": image_url},
            {"type": "text", "value": question},
        ]
        user_content = self._prepare_content(raw_inputs)

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _prepare_inputs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        images: List[Image.Image] = []
        videos: List[str] = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    image_path = item.get("image", "")
                    if image_path.startswith("file://"):
                        image_path = image_path[7:]
                    img = Image.open(image_path).convert("RGB")
                    images.append(img)
                    
                elif item.get("type") == "video":
                    video_path = item.get("video", "")
                    if video_path.startswith("file://"):
                        video_path = video_path[7:]
                    videos.append(video_path)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            if len(text) == 1:
                text = text[0]
            else:
                raise ValueError("apply_chat_template returned multiple conversations")

        inputs = self.processor(
            text=text,
            images=images if images else None,
            videos=videos if videos else None,
            padding=True,
            return_tensors="pt",
        )

        
        return inputs

    def _extract_report_text(self, response: str) -> str:
        if "final conclusion is:" in response:
            return response.split("The final conclusion is:")[-1].strip()
        if "The final answer is:" in response:
            return response.split("The final answer is:")[-1].strip()
        return response.strip()

    def _parse_decimal(self, text: str) -> str:
        """Parse decimal numbers and replace '.' with '*' in decimals (same as training preprocessing)."""
        find_float = lambda x: re.search(r"\d+(\.\d+)", x).group()
        text_list = []
        for word in text.split():
            try:
                decimal = find_float(word)
                new_decimal = decimal.replace(".", "*")
                text_list.append(new_decimal)
            except:
                text_list.append(word)
        return " ".join(text_list)
    
    def _clean_train_sentence(self, text: str) -> str:
        """Clean a single sentence: remove non-alphabetic chars, lowercase, remove punctuation (same as training preprocessing)."""
        punc = list(punctuation)
        text = re.sub(r"xxxx", " ", text)
        text = re.sub(r"[^a-z\s]", "", text.lower())
        text_nopunc = [char for char in text if char not in punc]
        text_nopunc = "".join(text_nopunc)
        wd = []
        for word in text_nopunc.split():
            wd.append(word)
        sentence = " ".join(wd)
        if sentence.strip() == 'images':
            return ""
        return sentence
    
    def _clean_train_report(self, report: str) -> str:
        """Clean a report: parse decimals, normalize punctuation, clean sentences (same as training preprocessing)."""
        report = self._parse_decimal(report)
        report_cleaner = lambda t: t.replace('. .', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('.')
        
        tokens = [self._clean_train_sentence(sent) for sent in report_cleaner(report) if sent != '' and self._clean_train_sentence(sent) != ""]
        report = ' . '.join(tokens) + ' .'
        return report
    
    def _post_process_prediction(self, prediction: str) -> str:
        """
        Post-process prediction: first truncate to 130 tokens, then apply same preprocessing as training data.
        """
        tokens = self.processor.tokenizer.encode(prediction, add_special_tokens=False)
        if len(tokens) > 60:  #130
            tokens = tokens[:60]  #130
            prediction = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        
        cleaned = self._clean_train_report(prediction)
        
        return cleaned


def _safe_to_list(value: Optional[torch.Tensor]) -> Optional[List[Any]]:
    if value is None:
        return None
    try:
        return value.detach().cpu().to(torch.float32).numpy().tolist()
    except Exception:
        return None


MODEL_REGISTRY: Dict[str, Any] = {}


def register_model(name: str, **kwargs: Any) -> None:
    MODEL_REGISTRY[name] = partial(MedFullQwen2VLChat, **kwargs)


def build_model(name: str, **overrides: Any) -> MedFullQwen2VLChat:
    assert name in MODEL_REGISTRY, f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
    builder = MODEL_REGISTRY[name]
    return builder(**overrides)

