#!/usr/bin/env python3
"""Unlimited OCR - Extended Benchmark (large files + 40+ pages)"""

import torch, os, time, fitz, json, sys
from transformers import AutoModel, AutoTokenizer
from PIL import Image

device = torch.device('cuda')
print('Loading model...', flush=True)
t0 = time.time()
model = AutoModel.from_pretrained('baidu/Unlimited-OCR', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
print(f'Model loaded in {time.time()-t0:.0f}s', flush=True)


def pdf_page_to_image(pdf_path, page_num, dpi=200):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    doc.close()
    return img, pix.width, pix.height


def ocr_image(model, tokenizer, img, prompt='document parsing.'):
    output_dir = './ocr_output'
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, f'_tmp_{time.time()}.png')
    img.save(tmp_path)
    try:
        model.infer(tokenizer=tokenizer, image_file=tmp_path,
                    prompt=prompt, output_path=output_dir,
                    temperature=0.0, max_length=2048)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Test configurations: (file, desc, pages_to_test, is_long)
TEST_FILES = [
    # Original small/medium tests
    ('test_pdfs/01_arxiv_paper.pdf', 'English paper (14.5MB, 32p)', [0, 1, 2], False),
    ('test_pdfs/02_arxiv_tables.pdf', 'English tables (1.6MB, 55p)', [0, 1, 2], False),
    ('test_pdfs/03_chinese.pdf', 'Chinese paper (1.4MB, 5p)', [0, 1, 2], False),
    ('test_pdfs/04_chinese_news.pdf', 'Mixed CN/EN (3.1MB, 58p)', [0, 1, 2], False),
    ('test_pdfs/04_mixed_text.pdf', 'English LoRA paper (1.5MB, 26p)', [0, 1, 2], False),
    ('test_pdfs/05_llama_paper.pdf', 'LLaMA paper (0.7MB, 27p)', [0, 1, 2], False),
    
    # Large file test
    ('test_pdfs/08_large_images.pdf', 'LARGE-HiRes images (24MB, 19p)', [0, 1, 2, 5, 10, 15, 18], True),
    
    # Long document tests (40+ pages)
    ('test_pdfs/06_gpt4_report.pdf', 'LONG-GPT-4 report (5MB, 100p)', [0, 1, 2, 10, 20, 30, 50, 70, 90, 99], True),
    ('test_pdfs/07_deepseek_r1.pdf', 'LONG-DeepSeek-R1 (4.8MB, 86p)', [0, 1, 2, 10, 20, 30, 50, 70, 80, 85], True),
]

os.makedirs('./ocr_results_v2', exist_ok=True)

all_results = {}

for pdf_path, desc, page_nums, is_long in TEST_FILES:
    pdf_name = os.path.basename(pdf_path)
    size_mb = os.path.getsize(pdf_path) / 1024**2
    out_dir = f'./ocr_results_v2/{pdf_name.replace(".pdf","")}'
    os.makedirs(out_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    print(f'\n{"="*80}')
    print(f'  {"📄" if not is_long else "🔴"} {pdf_name}')
    print(f'     {desc}')
    print(f'{"─"*60}')
    
    page_times = []
    for pn in page_nums:
        if pn >= total_pages:
            print(f'  Skipping page {pn+1} (only {total_pages} total)')
            continue
            
        print(f'  Page {pn+1}/{total_pages} ...', end=' ', flush=True)
        t0 = time.time()
        img, w, h = pdf_page_to_image(pdf_path, pn, dpi=200)
        ocr_image(model, tokenizer, img)
        elapsed = time.time() - t0
        page_times.append({'page': pn+1, 'time': round(elapsed, 1), 'w': w, 'h': h})
        print(f'{elapsed:.1f}s ({w}x{h})', flush=True)
    
    avg = round(sum(p['time'] for p in page_times) / len(page_times), 1)
    result = {
        'desc': desc, 'size_mb': round(size_mb, 1), 'total_pages': total_pages,
        'tested': len(page_times), 'avg_time': avg,
        'pages': page_times
    }
    all_results[pdf_name] = result
    with open(f'{out_dir}/_info.json', 'w') as f:
        json.dump(result, f, indent=2)

print(f'\n{"="*80}')
print('  EXTENDED BENCHMARK SUMMARY')
print('='*80)
print(f'  {"File":30s} {"Type":30s} {"Size":>7s} {"Pages":>5s} {"Tested":>6s} {"Avg/pg":>8s}')
print(f'  {"─"*30} {"─"*30} {"─"*7} {"─"*5} {"─"*6} {"─"*8}')
for name, r in all_results.items():
    flag = '🔴' if 'LONG' in r['desc'] or 'LARGE' in r['desc'] else '  '
    note = ''
    if 'LARGE' in r['desc']:
        note = ' ★LARGE FILE'
    if 'LONG' in r['desc']:
        note = ' ★LONG DOC'
    print(f'  {flag} {name:28s} {r["desc"][:28]:28s} {r["size_mb"]:>5.1f}MB {r["total_pages"]:>3d}p {r["tested"]:>4d}p {r["avg_time"]:>6.1f}s{note}')
print(f'{"="*80}')
print(f'  VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')
print(f'  Results: ocr_results_v2/')
