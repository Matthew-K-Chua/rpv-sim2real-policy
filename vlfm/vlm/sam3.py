# Adam Scicluna, 2026, for RPV pipeline. Based on sam.py.

import gc
import os
from typing import Any, List, Optional

import numpy as np
import torch

from .server_wrapper import (
    ServerMixin,
    host_model,
    mask_to_str,
    send_request,
    str_to_image,
)

# Import any additional libraries needed for SAM3
# SAM3 Ultralytics (approx. 2x faster in testing, due to cast to half precision/float16)
from ultralytics.models.sam import SAM3SemanticPredictor

# SAM3 (not Ultralytics)
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Edit this class to load the SAM3 model and implement the segment_bbox method using the SAM3 model.
# Use the MobileSAM class in sam.py as a reference for how to structure this.
class SAM3:
    def __init__(
        self,
        sam3_ckpt: str = "checkpoints/sam3.pt",
        bpe_path: Optional[str] = None,
        conf_threshold: float = 0.7,
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        # If using the non-Ultralytics SAM3, load the model as such (this will download the model if not already cached)
        if bpe_path is not None:
            sam3_model = build_sam3_image_model(bpe_path=bpe_path)
            self.processor = Sam3Processor(sam3_model, confidence_threshold=conf_threshold, device="cuda")
            self.use_ultralytics = False

        else:
            # If using the Ultralytics SAM3, load the model as such:
            # imgsz belongs HERE, not on the per-call predictor(...) invocation:
            # a per-call imgsz forces Ultralytics to re-run setup_model on every
            # request (re-printing the device banner) and leaks ~3 GB/request
            # until OOM. Setting it once in overrides keeps the input resolution
            # the RPV pipeline was benchmarked at while setting the model up once.
            # Embodied-RPV-NOTE: SAM3_IMGSZ exists only so a memory-constrained
            # deployment can trade resolution for VRAM; leave it unset to match
            # the benchmark.
            self.predictor = SAM3SemanticPredictor(
                overrides = dict(
                    task="segment",
                    mode="predict",
                    model=sam3_ckpt,
                    conf=conf_threshold,
                    half=True,
                    device=device if device == "cpu" else "cuda",
                    imgsz=int(os.environ.get("SAM3_IMGSZ", 644)),
                    save=False
                )
            )
            self.use_ultralytics = True

        self._num_steps = 0  # For logging purposes; incremented on each segment_masks call


    # TO-DO: Check output formats for both versions of SAM3 and implement method accordingly. 
    # The output should be an array of boolean masks, along with the corresponding labels/classes
    def segment_masks(self, image: Image.Image, text_prompts: List[str]):
        """
        Segments the objects in the image corresponding to the text prompts.

        Args:
            image (PIL.Image.Image): The input image as a PIL Image (SAM3 requires PIL images as input)
            text_prompts (List[str]): A list of text prompts to segment (detections + target object descriptions)
        """
        # # Save the image passed to this function to see if it is RGB or BGR (for debugging purposes)
        # # We want the image going into Mask2Former to be RGB
        # os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        # image.save(path_to_save)

        # Define empty outputs in case no detections are made, to avoid crashing SAM3 and torch.cat when passed empty lists.
        masks, boxes, scores, labels = [], [], [], []

        if self.use_ultralytics:
            # If no classes were detected, return empty results immediately.
            # SAM3 (and torch.cat) will crash if passed an empty list of text prompts.
            if not text_prompts:
                print("No text prompts. Skipping SAM3 inference.")
                return masks, boxes, scores, labels

            # Run Inference. NOTE: do NOT pass imgsz here — a per-call imgsz forces
            # Ultralytics to re-run setup_model every request (re-prints the device
            # banner) and leaks ~3 GB/request until OOM. The size is set once in
            # overrides=dict(...) at construction instead.
            results = self.predictor(image, text=text_prompts, verbose=False)
            result = results[0]

            # os.makedirs(os.path.dirname(sam3_output_path), exist_ok=True)
            # result.save(filename=sam3_output_path)
            # self._num_steps += 1

            if result is None:
                return masks, boxes, scores, labels

            # Extract Data
            if result.masks is not None:
                masks = list(result.masks.data.cpu().numpy())
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy().tolist()
                scores = result.boxes.conf.cpu().numpy().tolist()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                # Use result.names to map IDs back to the text labels you provided
                labels = [result.names[i] for i in class_ids]

            # masks/boxes/scores/labels are now CPU copies — free the GPU result
            # (and the predictor's cached tensors) so memory doesn't creep per request.
            del results, result
            gc.collect()
            torch.cuda.empty_cache()

        else:
            # ENABLE FP16 INFERENCE HERE
            # This tells PyTorch: "Use FP16 for operations where it is safe/faster"
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Set SAM3 image to predict on (should be the original unmasked, as the masking was just for the YOLO-E step)
                inference_state = self.processor.set_image(image)
                self.processor.reset_all_prompts(inference_state)
                # Prediction
                masks, boxes, scores, labels = [], [], [], []
                for classes in text_prompts:
                    # Prompt the model with text
                    output = self.processor.set_text_prompt(state=inference_state, prompt=classes)

                    masks.extend(output["masks"])
                    boxes.extend(output["boxes"])
                    scores.extend(output["scores"])
                    labels.extend([classes] * len(output["masks"]))


        return masks, boxes, scores, labels

            


class SAM3Client:
    def __init__(self, port: int = 12183):
        self.url = f"http://localhost:{port}/sam3"

    def segment_masks(self, image: np.ndarray, text_prompts: List[str]) -> dict:
        """Return masks, boxes, scores, labels as JSON-friendly lists."""
        response = send_request(self.url, image=image, text_prompts=text_prompts)
        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12183)
    parser.add_argument("--sam3_ckpt", type=str, default=os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt"))
    parser.add_argument("--bpe_path", type=str, default=os.environ.get("SAM3_BPE_PATH"))
    parser.add_argument("--conf_threshold", type=float, default=0.6)
    args = parser.parse_args()

    print("Loading model...")

    class SAM3Server(ServerMixin, SAM3):
        def process_payload(self, payload: dict) -> dict:
            np_image = str_to_image(payload["image"])
            pil_image = Image.fromarray(np_image)  # Image is already RGB (Habitat), so just convert to PIL format for SAM3
            text_prompts = payload.get("text_prompts", [])
            masks, boxes, scores, labels = self.segment_masks(pil_image, text_prompts)

            # Bit-pack masks (compact base64 + shape) rather than nested bool
            # lists: ~30x smaller over JSON and avoids a large RAM spike.
            masks_serialisable = [mask_to_str(m) for m in masks]

            return {
                "masks": masks_serialisable,
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }

    sam3 = SAM3Server(
        sam3_ckpt=args.sam3_ckpt,
        bpe_path=args.bpe_path,
        conf_threshold=args.conf_threshold,
    )
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(sam3, name="sam3", port=args.port)