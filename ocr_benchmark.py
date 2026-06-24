#!/usr/bin/env python3
"""Unlimited OCR Benchmark - Comprehensive test across diverse PDFs"""

import torch, os, time, fitz, sys
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
        result = model.infer(
            tokenizer=tokenizer, image_file=tmp_path,
            prompt=prompt, output_path=output_dir,
            temperature=0.0, max_length=4096
        )
        # Also read the output file if saved
        return result if result else '(see console output)'
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Test PDF configurations
TEST_FILES = [
    ('test_pdfs/01_arxiv_paper.pdf',  'English arXiv paper (math/formulas)'),
    ('test_pdfs/02_arxiv_tables.pdf', 'English paper with tables'),
    ('test_pdfs/03_chinese.pdf',      'Chinese paper'),
    ('test_pdfs/04_chinese_news.pdf',  'Mixed Chinese/English doc'),
    ('test_pdfs/04_mixed_text.pdf',   'English mixed text'),
]

print('\n' + '='*80)
print('  UNLIMITED OCR — BENCHMARK REPORT')
print('='*80)
print(f'  GPU:     {torch.cuda.get_device_name(0)}')
print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')
print(f'  Model:   3.34B params, bfloat16')
print(f'  Time:    {time.strftime("%Y-%m-%d %H:%M")}')
print('='*80)

summary = []
os.makedirs('./ocr_results', exist_ok=True)

for pdf_path, desc in TEST_FILES:
    if not os.path.exists(pdf_path):
        print(f'\n  ⚠️  {pdf_path} not found, skipping')
        continue
    
    pdf_name = os.path.basename(pdf_path)
    size_kb = os.path.getsize(pdf_path) // 1024
    out_dir = f'./ocr_results/{pdf_name.replace(".pdf","")}'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f'\n{"─"*80}')
    print(f'  📄 {pdf_name}')
    print(f'     Type: {desc}')
    print(f'     Size: {size_kb}KB')
    print(f'{"─"*60}')
    
    # Get page count
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    page_results = []
    # Process max 5 pages per PDF to keep test time reasonable
    pages_to_process = min(total_pages, 5)
    
    for page_num in range(pages_to_process):
        print(f'\n  --- Page {page_num+1}/{total_pages} ---', flush=True)
        
        t0 = time.time()
        img, w, h = pdf_page_to_image(pdf_path, page_num, dpi=200)
        prep_time = time.time() - t0
        
        print(f'  Image: {w}x{h} ({prep_time:.1f}s to render)', flush=True)
        
        t1 = time.time()
        result = ocr_image(model, tokenizer, img)
        ocr_time = time.time() - t1
        total_time = time.time() - t0
        
        # Get actual text length from generated output (the function prints to console)
        char_count = len(result) if result and result != '(see console output)' else 0
        
        print(f'  ⏱  {ocr_time:.1f}s | Chars: {char_count}' , flush=True)
        
        page_results.append({
            'page': page_num + 1,
            'time': round(total_time, 1),
            'width': w,
            'height': h,
        })
    
    # Save summary for this PDF
    with open(f'{out_dir}/_info.json', 'w') as f:
        import json
        json.dump({'file': pdf_name, 'desc': desc, 'size_kb': size_kb, 
                   'pages': total_pages, 'processed': pages_to_process,
                   'results': page_results}, f, indent=2)
    
    avg_time = sum(r['time'] for r in page_results) / len(page_results)
    summary.append({
        'name': pdf_name, 'desc': desc, 'size_kb': size_kb,
        'pages': total_pages, 'processed': pages_to_process,
        'avg_time': round(avg_time, 1)
    })

print(f'\n{"="*80}')
print(f'  📊 SUMMARY')
print(f'='*80)
print(f'  {"File":30s} {"Type":30s} {"Size":>6s} {"Pages":>5s} {"Avg/pg":>8s}')
print(f'  {"─"*30} {"─"*30} {"─"*6} {"─"*5} {"─"*8}')
for s in summary:
    print(f'  {s["name"]:30s} {s["desc"]:30s} {s["size_kb"]:>5d}KB {s["processed"]:>3d}/{s["pages"]:<d} {s["avg_time"]:>6.1f}s')
print(f'{"="*80}')
print(f'  Results saved to ocr_results/')
print(f'  VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')
