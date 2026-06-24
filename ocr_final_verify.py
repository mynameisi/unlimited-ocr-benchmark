#!/usr/bin/env python3
"""Final verification: all 6 challenge PDFs with optimal parameters"""
import torch, os, time, fitz, json
from transformers import AutoModel, AutoTokenizer
from PIL import Image

BEST_NGRAM = 50
BEST_WINDOW = 128
BEST_TEMP = 0.0
MAX_LEN = 2048

device = torch.device('cuda')
print('Loading model...', flush=True)
t0 = time.time()
model = AutoModel.from_pretrained('baidu/Unlimited-OCR', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

def test_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(200/72, 200/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    doc.close()
    
    out_dir = './ocr_output'
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f'_tmp_{time.time()}.png')
    img.save(tmp)
    try:
        t1 = time.time()
        model.infer(tokenizer=tokenizer, image_file=tmp, prompt='document parsing.',
                     output_path=out_dir, temperature=BEST_TEMP, max_length=MAX_LEN,
                     no_repeat_ngram_size=BEST_NGRAM, ngram_window=BEST_WINDOW)
        elapsed = time.time() - t1
        return elapsed, img.size
    finally:
        if os.path.exists(tmp): os.remove(tmp)

challenges = [
    ("c1_chinese_scanned.pdf",  "中文扫描件"),
    ("c2_photo_doc.pdf",        "手机拍摄文档"),
    ("c3_handwriting.pdf",      "手写体"),
    ("c4_poor_quality.pdf",     "低质量扫描"),
    ("c5_newspaper.pdf",        "多栏报纸"),
    ("c6_stamped_doc.pdf",      "盖章合同"),
]

os.makedirs('./ocr_results_final', exist_ok=True)
results = []

for fname, desc in challenges:
    path = f'test_pdfs/{fname}'
    sz = os.path.getsize(path) // 1024
    print(f'\n  📄 {fname} ({sz}KB) - {desc}', flush=True)
    elapsed, size = test_ocr(path)
    seconds = round(elapsed, 1)
    print(f'     ⏱  {seconds}s | {size[0]}x{size[1]}')
    results.append({'file': fname, 'desc': desc, 'size_kb': sz, 'time_s': seconds})

with open('ocr_results_final/_final_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{"="*80}')
print(f'  FINAL VERIFICATION — Best params: ngram={BEST_NGRAM}, window={BEST_WINDOW}, temp={BEST_TEMP}')
print(f'{"="*80}')
print(f'  {"File":30s} {"Desc":20s} {"Time":>8s}')
print(f'  {"─"*30} {"─"*20} {"─"*8}')
for r in results:
    print(f'  {r["file"]:30s} {r["desc"]:20s} {r["time_s"]:>6.1f}s')
print(f'{"="*80}')
print(f'  VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')
