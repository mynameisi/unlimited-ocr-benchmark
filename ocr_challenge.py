#!/usr/bin/env python3
"""OCR Benchmark on Real-World Challenge Images"""
import torch, os, time, fitz, json
from transformers import AutoModel, AutoTokenizer
from PIL import Image

device = torch.device('cuda')
print('Loading model...', flush=True)
t0 = time.time()
model = AutoModel.from_pretrained('baidu/Unlimited-OCR', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

def ocr_page(pdf_path):
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
                     output_path=out_dir, temperature=0.0, max_length=2048)
        elapsed = time.time() - t1
        return elapsed, pix.width, pix.height
    finally:
        if os.path.exists(tmp): os.remove(tmp)

# Challenge test files: (filename, scenario)
challenges = [
    ("c1_chinese_scanned.pdf",  "中文扫描件（带噪点+轻微倾斜）"),
    ("c2_photo_doc.pdf",        "手机拍摄文档（阴影+折痕）"),
    ("c3_handwriting.pdf",      "手写体（模拟笔迹）"),
    ("c4_poor_quality.pdf",     "低质量扫描（低对比度+模糊+泛黄）"),
    ("c5_newspaper.pdf",        "多栏报纸排版"),
    ("c6_stamped_doc.pdf",      "合同盖章/水印覆盖"),
]

os.makedirs('./ocr_results_challenge', exist_ok=True)

print(f'\n{"="*80}')
print(f'  UNLIMITED OCR — REAL-WORLD CHALLENGE')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')
print(f'{"="*80}')

results = []
for fname, scenario in challenges:
    path = os.path.join('test_pdfs', fname)
    if not os.path.exists(path):
        print(f'\n  ⚠️  {fname} not found')
        continue
    
    sz_kb = os.path.getsize(path) // 1024
    print(f'\n{"─"*80}')
    print(f'  📄 {fname} ({sz_kb}KB)')
    print(f'  🏷️  {scenario}')
    print(f'{"─"*60}')
    
    try:
        elapsed, w, h = ocr_page(path)
        print(f'  ⏱  {elapsed:.1f}s | {w}x{h}')
        results.append({'file': fname, 'scenario': scenario, 'size_kb': sz_kb,
                        'time': round(elapsed,1), 'w': w, 'h': h, 'status': 'OK'})
    except Exception as e:
        print(f'  ❌ FAILED: {e}')
        results.append({'file': fname, 'scenario': scenario, 'size_kb': sz_kb,
                        'status': f'FAILED: {str(e)[:80]}'})

with open('ocr_results_challenge/_summary.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=True)

print(f'\n{"="*80}')
print(f'  RESULTS SUMMARY')
print(f'{"="*80}')
print(f'  {"File":30s} {"Scenario":40s} {"Time":>8s} {"Status":>10s}')
print(f'  {"─"*30} {"─"*40} {"─"*8} {"─"*10}')
for r in results:
    t = f'{r["time"]}s' if 'time' in r else '-'
    print(f'  {r["file"]:30s} {r["scenario"]:40s} {t:>8s} {r["status"]:>10s}')
print(f'{"="*80}')
print(f'  VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')
