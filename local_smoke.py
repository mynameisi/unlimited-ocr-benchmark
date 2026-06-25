#!/usr/bin/env python3
"""Smoke test for local hardware: load model, measure VRAM, run single inference."""
import os, time, torch, fitz
from transformers import AutoModel, AutoTokenizer
from PIL import Image

GB = 1024**3
torch.cuda.reset_peak_memory_stats()
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}  total VRAM: {props.total_memory/GB:.2f}GB", flush=True)
free0, total0 = torch.cuda.mem_get_info()
print(f"Free VRAM before load: {free0/GB:.2f}GB", flush=True)

print("Loading model...", flush=True)
t0 = time.time()
model = AutoModel.from_pretrained('baidu/Unlimited-OCR', torch_dtype=torch.bfloat16,
                                  trust_remote_code=True, use_safetensors=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
print(f"Model loaded in {time.time()-t0:.0f}s", flush=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params/1e9:.3f}B", flush=True)
alloc = torch.cuda.memory_allocated()/GB
free1, _ = torch.cuda.mem_get_info()
print(f"VRAM allocated after load: {alloc:.2f}GB | free now: {free1/GB:.2f}GB", flush=True)

# Render c1 page to image
doc = fitz.open('test_pdfs/c1_chinese_scanned.pdf')
page = doc[0]
mat = fitz.Matrix(200/72, 200/72)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
doc.close()
os.makedirs('./ocr_output', exist_ok=True)
img.save('./ocr_output/_smoke.png')
print(f"Image: {pix.width}x{pix.height}", flush=True)

print("Running inference...", flush=True)
t1 = time.time()
try:
    res = model.infer(tokenizer, prompt='<image>document parsing.',
                      image_file='./ocr_output/_smoke.png', output_path='./ocr_output',
                      base_size=1024, image_size=640, crop_mode=True,
                      max_length=4096, no_repeat_ngram_size=35, ngram_window=128,
                      save_results=False)
    print(f"\nInference OK in {time.time()-t1:.1f}s", flush=True)
except Exception as e:
    print(f"\nInference FAILED: {type(e).__name__}: {e}", flush=True)

peak = torch.cuda.max_memory_allocated()/GB
free2, _ = torch.cuda.mem_get_info()
print(f"PEAK VRAM allocated: {peak:.2f}GB | free at peak-ish: {free2/GB:.2f}GB / {props.total_memory/GB:.2f}GB", flush=True)
