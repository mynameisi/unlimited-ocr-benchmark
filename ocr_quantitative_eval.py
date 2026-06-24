#!/usr/bin/env python3
"""Quantitative OCR quality evaluation using edit distance / CER / WER"""

import torch, os, time, fitz, json, re
from transformers import AutoModel, AutoTokenizer
from PIL import Image

device = torch.device('cuda')
print('Loading model...', flush=True)
t0 = time.time()
model = AutoModel.from_pretrained('baidu/Unlimited-OCR', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

BEST_NGRAM = 50
BEST_WINDOW = 128
BEST_TEMP = 0.0
MAX_LEN = 2048

def capture_ocr(model, tokenizer, img):
    """Run OCR and capture the generated text output"""
    out_dir = './ocr_output'
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f'_tmp_{time.time()}.png')
    img.save(tmp)
    
    # Capture stdout during model.infer
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    
    try:
        t1 = time.time()
        model.infer(tokenizer=tokenizer, image_file=tmp, prompt='document parsing.',
                     output_path=out_dir, temperature=BEST_TEMP, max_length=MAX_LEN,
                     no_repeat_ngram_size=BEST_NGRAM, ngram_window=BEST_WINDOW)
        elapsed = time.time() - t1
    finally:
        sys.stdout = old_stdout
        if os.path.exists(tmp):
            os.remove(tmp)
    
    text = captured.getvalue()
    return text, elapsed

def normalize(s):
    """Normalize text for comparison: lowercase, collapse whitespace"""
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_cer(hypothesis, reference):
    """Character Error Rate (edit distance / ref length)"""
    h, r = normalize(hypothesis), normalize(reference)
    m, n = len(h), len(r)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if h[i-1] == r[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    edit_dist = dp[m][n]
    cer = edit_dist / max(n, 1)
    return cer, edit_dist, len(r)

# ===== GROUND TRUTH DEFINITIONS =====
# These are the exact texts we wrote when generating the challenge PDFs

ground_truths = {
    "c1_chinese_scanned.pdf": (
        "本报讯（记者 张华）2024年6月15日，全国人工智能发展大会在北京隆重召开。"
        "本次大会汇聚了来自全国各地的顶尖学者、行业专家和企业代表，共同探讨人工智能技术的最新进展和未来发展方向。"
        "会上，多位知名专家发表了精彩演讲，就大语言模型、多模态学习、智能机器人等前沿领域进行了深入交流。"
        "会议期间还举办了多场专题论坛，涵盖了AI在医疗、教育、金融等领域的应用实践。"
        "与会代表一致认为，人工智能技术正在深刻改变着各行各业的发展模式，未来将在更多领域发挥重要作用。"
        "本次大会的成功举办，为我国人工智能技术的发展注入了新的动力。"
    ),
    "c2_photo_doc.pdf": (
        "APPLICATION FOR EMPLOYMENT\n\nPersonal Information:\n"
        "Name: John A. Smith\nDate of Birth: March 15, 1985\n"
        "Phone: (555) 123-4567\nEmail: john.smith@email.com\n\n"
        "Education:\nBachelor of Science in Computer Science\n"
        "University of Technology, 2008\n\nWork Experience:\n"
        "Senior Software Engineer\nTech Corp Inc., 2015 - Present\n"
        "- Led development of cloud-based microservices\n"
        "- Managed team of 5 engineers\n"
        "- Reduced system latency by 40%\n\nPrevious Employment:\n"
        "Software Developer\nStartUp Labs, 2012 - 2015\n"
        "- Developed web applications using Python/Django\n"
        "- Implemented CI/CD pipeline"
    ),
    "c3_handwriting.pdf": (
        "Dear Sir/Madam,\n\nI am writing to apply for the position of Research "
        "Assistant at your laboratory. I recently completed my PhD in Computer "
        "Vision at Stanford University, where my research focused on deep learning "
        "methods for document understanding.\n\nDuring my doctoral studies, I have "
        "published 8 papers in top-tier conferences including CVPR and ICCV. I have "
        "also developed several open-source tools for OCR and document parsing that "
        "have gained significant traction in the community.\n\nI believe my background "
        "and skills would be a great fit for your research group. I look forward to "
        "the opportunity to discuss my qualifications further.\n\nSincerely,\nDr. Jane Researcher"
    ),
    "c4_poor_quality.pdf": (
        "This is a poor quality scan of an old document.\n"
        "The paper has yellowed over time, and the text has faded.\n"
        "This tests the OCR model's ability to handle low contrast\n"
        "scenarios that are common in real-world document digitization.\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n"
        "0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>?"
    ),
    "c5_newspaper.pdf": (
        "BREAKING NEWS\n\nThe city council voted 7-2 last night to approve the "
        "new downtown development project. The plan includes three office towers, "
        "a public plaza, and underground parking for 2,000 vehicles. Construction "
        "is expected to begin next spring and take approximately 3 years to complete. "
        "Opponents of the project cited concerns about increased traffic and the "
        "displacement of existing small businesses.\n\n"
        "WEATHER\nToday: Sunny, High 72F\nTonight: Clear, Low 55F\nTomorrow: Partly Cloudy\n"
        "High 68F, Low 52F\nThe warm weather pattern is expected to continue through "
        "the weekend with temperatures reaching the mid-70s by Sunday."
    ),
    "c6_stamped_doc.pdf": (
        "CONTRACT AGREEMENT\n\n"
        "This agreement is entered into on June 1, 2024 between:\n\n"
        "Party A: ABC Corporation\n123 Business Ave, Suite 400\nNew York, NY 10001\n\n"
        "Party B: XYZ Solutions Ltd.\n456 Commerce Dr.\nSan Francisco, CA 94105\n\n"
        "Terms and Conditions:\n"
        "1. Party A agrees to provide consulting services to Party B\n"
        "2. The term of this agreement shall be 12 months\n"
        "3. Payment terms: Net 30 days from invoice date\n"
        "4. Either party may terminate with 30 days written notice\n\n"
        "IN WITNESS WHEREOF, the parties have executed this agreement\n"
        "on the date first written above.\n\n"
        "______________________             ______________________\n"
        "For ABC Corporation                For XYZ Solutions Ltd."
    ),
}

challenges = [
    "c1_chinese_scanned.pdf",
    "c2_photo_doc.pdf",
    "c3_handwriting.pdf",
    "c4_poor_quality.pdf",
    "c5_newspaper.pdf",
    "c6_stamped_doc.pdf",
]

os.makedirs('./ocr_results_quant', exist_ok=True)

print(f'\n{"="*100}')
print(f'  QUANTITATIVE OCR EVALUATION')
print(f'  Params: ngram={BEST_NGRAM}, window={BEST_WINDOW}, temp={BEST_TEMP}')
print(f'{"="*100}')
print(f'  {"File":30s} {"CER":>8s} {"Edits":>8s} {"RefLen":>8s} {"Time":>8s} {"Verdict":>12s}')
print(f'  {"─"*30} {"─"*8} {"─"*8} {"─"*8} {"─"*8} {"─"*12}')

all_results = []

for fname in challenges:
    path = f'test_pdfs/{fname}'
    gt = ground_truths[fname]
    
    # Render PDF to image
    doc = fitz.open(path)
    page = doc[0]
    mat = fitz.Matrix(200/72, 200/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    doc.close()
    
    # OCR with output capture
    ocr_output, elapsed = capture_ocr(model, tokenizer, img)
    
    # Extract just the text content from OCR output (strip <|det|> tags)
    ocr_text = re.sub(r'<\|det\|>\w+\s*\[?[^\]]*\]?\s*\|?', '', ocr_output)
    ocr_text = re.sub(r'<\|/det\|>', ' ', ocr_text)
    # Actually the OCR text is in the <|det|>tag [x,y,w,h]</det|>... format
    # Let me extract the actual content after </det|>
    content_parts = re.findall(r'<\|/det\|>(.+?)(?=<|\n|$)', ocr_output, re.DOTALL)
    ocr_clean = ' '.join(p.strip() for p in content_parts if p.strip())
    
    if not ocr_clean:
        # Fallback: use the full text without tags
        ocr_clean = re.sub(r'<[^>]+>', ' ', ocr_output)
    
    # Compute CER
    cer, edits, ref_len = compute_cer(ocr_clean, gt)
    
    # Determine verdict
    if cer < 0.10:
        verdict = '⭐EXCELLENT'
    elif cer < 0.30:
        verdict = '✅ GOOD'
    elif cer < 0.50:
        verdict = '⚠️ FAIR'
    else:
        verdict = '❌ POOR'
    
    print(f'  {fname:30s} {cer:>7.1%} {edits:>8d} {ref_len:>8d} {elapsed:>6.1f}s {verdict:>12s}')
    
    all_results.append({
        'file': fname,
        'cer': round(cer, 4),
        'edit_distance': edits,
        'ref_chars': ref_len,
        'time_s': round(elapsed, 1),
        'verdict': verdict.strip()
    })

print(f'{"="*100}')
avg_cer = sum(r['cer'] for r in all_results) / len(all_results)
avg_time = sum(r['time_s'] for r in all_results) / len(all_results)
print(f'  AVERAGE: CER={avg_cer:.1%}, Time={avg_time:.1f}s')

with open('ocr_results_quant/_quantitative_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\nResults saved to ocr_results_quant/')
