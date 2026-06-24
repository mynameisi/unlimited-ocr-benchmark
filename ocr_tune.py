#!/usr/bin/env python3
"""Hyperparameter tuning for Unlimited OCR - find optimal no_repeat_ngram_size"""
import torch, os, time, json
from transformers import AutoModel, AutoTokenizer
from PIL import Image

device = torch.device('cuda')
print('Loading model...', flush=True)
t0 = time.time()
model = AutoModel.from_pretrained('baidu/Unlimited-OCR', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

def ocr_with_params(model, tokenizer, img, no_repeat_ngram_size, ngram_window, temperature=0.0, max_length=2048):
    out_dir = './ocr_output'
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f'_tmp_{time.time()}.png')
    img.save(tmp)
    try:
        t1 = time.time()
        model.infer(tokenizer=tokenizer, image_file=tmp, prompt='document parsing.',
                     output_path=out_dir, temperature=temperature, max_length=max_length,
                     no_repeat_ngram_size=no_repeat_ngram_size, ngram_window=ngram_window)
        elapsed = time.time() - t1
        return elapsed
    finally:
        if os.path.exists(tmp): os.remove(tmp)

# Use c6_stamped_doc.pdf (contract with stamp) - had worst repetition
import fitz
pdf_path = 'test_pdfs/c6_stamped_doc.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
mat = fitz.Matrix(200/72, 200/72)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
doc.close()

# Parameter grid to search
param_grid = [
    # (no_repeat_ngram_size, ngram_window, temperature, max_length, label)
    (0,   0,   0.0, 4096,  "baseline (default)"),
    (10,  30,  0.0, 2048,  "ngram10_w30"),
    (20,  50,  0.0, 2048,  "ngram20_w50"),
    (35,  100, 0.0, 2048,  "ngram35_w100 (infer.py default)"),
    (50,  128, 0.0, 2048,  "ngram50_w128"),
    (35,  200, 0.0, 2048,  "ngram35_w200"),
    (20,  128, 0.1, 2048,  "ngram20_w128_temp0.1"),
    (35,  128, 0.1, 2048,  "ngram35_w128_temp0.1"),
    (50,  128, 0.1, 2048,  "ngram50_w128_temp0.1"),
    (35,  128, 0.0, 2048,  "ngram35_w128 (recommended)"),
    (50,  100, 0.0, 4096,  "ngram50_w100_long"),
    (70,  200, 0.0, 2048,  "ngram70_w200"),
    (80,  200, 0.0, 2048,  "ngram80_w200"),
    (100, 200, 0.0, 2048,  "ngram100_w200"),
]

results = []
print(f'\n{"="*100}')
print(f'  HYPERPARAMETER TUNING — Test: c6_stamped_doc.pdf (contract + stamp overlay)')
print(f'  Image: {img.size}')
print(f'{"="*100}')

for ngram_size, window, temp, max_len, label in param_grid:
    print(f'\n  Testing: {label}', flush=True)
    print(f'    ngram={ngram_size}, window={window}, temp={temp}, max={max_len}')
    
    elapsed = ocr_with_params(model, tokenizer, img, ngram_size, window, temp, max_len)
    
    # Get the output from the last model run (it prints to stdout during generation)
    print(f'    ⏱  {elapsed:.1f}s')
    
    results.append({
        'label': label,
        'no_repeat_ngram_size': ngram_size,
        'ngram_window': window,
        'temperature': temp,
        'max_length': max_len,
        'time_s': round(elapsed, 1)
    })

# Save results
with open('ocr_results_challenge/_tuning_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{"="*100}')
print(f'  TUNING COMPLETE — {len(results)} configurations tested')
print(f'{"="*100}')
print(f'  {"#":3s} {"Label":30s} {"ngram":>6s} {"window":>7s} {"temp":>5s} {"max":>5s} {"time":>6s}')
print(f'  {"─"*3} {"─"*30} {"─"*6} {"─"*7} {"─"*5} {"─"*5} {"─"*6}')
for i, r in enumerate(results):
    print(f'  {i:3d} {r["label"]:30s} {r["no_repeat_ngram_size"]:6d} {r["ngram_window"]:7d} {r["temperature"]:5.1f} {r["max_length"]:5d} {r["time_s"]:6.1f}s')
print(f'{"="*100}')
print(f'  See console output above for OCR quality per configuration')
