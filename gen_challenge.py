"""Generate real-world OCR challenge test images"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, random

test_dir = "test_pdfs"
os.makedirs(test_dir, exist_ok=True)

IMG_W, IMG_H = 1600, 2200

def add_noise(img, amount=0.05):
    """Add random noise to simulate scan artifacts"""
    pixels = img.load()
    w, h = img.size
    for _ in range(int(w * h * amount)):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        pixels[x, y] = tuple(min(255, max(0, c + random.randint(-40, 40))) for c in pixels[x, y])
    return img

def add_skew(img, max_angle=3):
    """Apply slight rotation to simulate skewed scan"""
    return img.rotate(random.uniform(-max_angle, max_angle), 
                      resample=Image.BICUBIC, expand=False, fillcolor=255)

def add_blur(img, radius=0.5):
    """Apply Gaussian blur to simulate out-of-focus scan"""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_stamp(img, text="DRAFT"):
    """Add a simulated stamp/seal overlay"""
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
    except:
        font = ImageFont.load_default()
    # Red stamp with rotation
    stamp = Image.new("RGBA", img.size, (0,0,0,0))
    sdraw = ImageDraw.Draw(stamp)
    sdraw.text((100, 100), text, fill=(255,50,50,180), font=font)
    stamp = stamp.rotate(random.uniform(-20, -10), expand=False)
    img.paste(stamp, (0, 0), stamp)
    return img

def add_fold_shadow(img):
    """Simulate document fold shadow"""
    draw = ImageDraw.Draw(img, "RGBA")
    x = random.randint(img.width//3, 2*img.width//3)
    for i in range(20):
        alpha = int(40 * (1 - i/20))
        draw.line([(x+i, 0), (x+i, img.height)], fill=(100, 100, 100, alpha), width=1)
    return img

def render_text(draw, text, x, y, font, max_width, color=0):
    """Render text with word wrap"""
    words = text.split()
    line = ""
    lines = []
    for w in words:
        test = line + " " + w if line else w
        bbox = draw.textbbox((0,0), test, font=font)
        if bbox[2] - bbox[0] > max_width and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    for l in lines:
        draw.text((x, y), l, fill=color, font=font)
        y += (draw.textbbox((0,0), "A", font=font)[3] - draw.textbbox((0,0), "A", font=font)[1]) + 5
    return y

# ===== 1. SCANNED CHINESE TEXT =====
print("1. Scanned Chinese text...")
img = Image.new("RGB", (IMG_W, IMG_H), "white")
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 28)
except:
    font = ImageFont.load_default()
chinese_text = """本报讯（记者 张华）2024年6月15日，全国人工智能发展大会在北京隆重召开。
本次大会汇聚了来自全国各地的顶尖学者、行业专家和企业代表，共同探讨人工智能技术的
最新进展和未来发展方向。会上，多位知名专家发表了精彩演讲，就大语言模型、多模态
学习、智能机器人等前沿领域进行了深入交流。会议期间还举办了多场专题论坛，涵盖了
AI在医疗、教育、金融等领域的应用实践。与会代表一致认为，人工智能技术正在深刻
改变着各行各业的发展模式，未来将在更多领域发挥重要作用。本次大会的成功举办，
为我国人工智能技术的发展注入了新的动力。"""
y = 50
for line in chinese_text.strip().split("\n"):
    draw.text((50, y), line, fill=0, font=font)
    y += 45
# Apply scan artifacts: slight noise + skew
img = add_noise(img, 0.02)
img = add_skew(img, 2)
img.save(os.path.join(test_dir, "c1_chinese_scanned.pdf"), "PDF", resolution=200)
print("  c1_chinese_scanned.pdf: OK")

# ===== 2. PHOTO OF DOCUMENT =====
print("2. Photo of document with shadows...")
img = Image.new("RGB", (IMG_W, IMG_H), "white")
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 32)
except:
    font = ImageFont.load_default()
text = """APPLICATION FOR EMPLOYMENT

Personal Information:
Name: John A. Smith
Date of Birth: March 15, 1985
Phone: (555) 123-4567
Email: john.smith@email.com

Education:
Bachelor of Science in Computer Science
University of Technology, 2008

Work Experience:
Senior Software Engineer
Tech Corp Inc., 2015 - Present
- Led development of cloud-based microservices
- Managed team of 5 engineers
- Reduced system latency by 40%

Previous Employment:
Software Developer
StartUp Labs, 2012 - 2015
- Developed web applications using Python/Django
- Implemented CI/CD pipeline"""
render_text(draw, text, 50, 50, font, IMG_W - 100)
# Photo effects: vignette (darken edges) + noise
img = add_noise(img, 0.03)
img = add_fold_shadow(img)
img = add_skew(img, 1.5)
img.save(os.path.join(test_dir, "c2_photo_doc.pdf"), "PDF", resolution=200)
print("  c2_photo_doc.pdf: OK")

# ===== 3. HANDWRITING MIMIC =====
print("3. Simulated handwriting...")
img = Image.new("RGB", (IMG_W, IMG_H), "white")
draw = ImageDraw.Draw(img)
# Use a handwriting-like font if available
try:
    hfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
except:
    hfont = ImageFont.load_default()
hw_text = """Dear Sir/Madam,

I am writing to apply for the position of Research Assistant
at your laboratory. I recently completed my PhD in Computer
Vision at Stanford University, where my research focused on
deep learning methods for document understanding.

During my doctoral studies, I have published 8 papers in
top-tier conferences including CVPR and ICCV. I have also
developed several open-source tools for OCR and document
parsing that have gained significant traction in the community.

I believe my background and skills would be a great fit for
your research group. I look forward to the opportunity to
discuss my qualifications further.

Sincerely,
Dr. Jane Researcher"""
render_text(draw, hw_text, 50, 50, hfont, IMG_W - 100)
img = add_skew(img, 1)
img = add_noise(img, 0.01)
img.save(os.path.join(test_dir, "c3_handwriting.pdf"), "PDF", resolution=200)
print("  c3_handwriting.pdf: OK")

# ===== 4. POOR QUALITY SCAN =====
print("4. Poor quality scan (low contrast + blur)...")
img = Image.new("RGB", (IMG_W, IMG_H), "#ccbbaa")  # yellowish paper
draw = ImageDraw.Draw(img)
try:
    pfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except:
    pfont = ImageFont.load_default()
# Low contrast text
render_text(draw, """This is a poor quality scan of an old document.
The paper has yellowed over time, and the text has faded.
This tests the OCR model's ability to handle low contrast
scenarios that are common in real-world document digitization.
The quick brown fox jumps over the lazy dog.
ABCDEFGHIJKLMNOPQRSTUVWXYZ
abcdefghijklmnopqrstuvwxyz
0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>?""",
            50, 50, pfont, IMG_W - 100, color=80)  # dark gray instead of black
img = add_blur(img, 1.0)
img = add_noise(img, 0.04)
img.save(os.path.join(test_dir, "c4_poor_quality.pdf"), "PDF", resolution=200)
print("  c4_poor_quality.pdf: OK")

# ===== 5. NEWSPAPER WITH MULTI-COLUMN =====
print("5. Multi-column newspaper layout...")
img = Image.new("RGB", (IMG_W, IMG_H), "#e8dcc8")
draw = ImageDraw.Draw(img)
try:
    nfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 22)
    tfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 36)
except:
    nfont = tfont = ImageFont.load_default()
# Left column
draw.text((30, 30), "BREAKING NEWS", fill=0, font=tfont)
left_text = """The city council voted 7-2
last night to approve the
new downtown development
project. The plan includes
three office towers, a
public plaza, and under-
ground parking for 2,000
vehicles. Construction is
expected to begin next
spring and take approxi-
mately 3 years to complete.
Opponents of the project
cited concerns about in-
creased traffic and the
displacement of existing
small businesses."""
render_text(draw, left_text, 30, 100, nfont, 700)

# Right column
draw.text((800, 30), "WEATHER", fill=80, font=tfont)
right_text = """Today: Sunny, High 72F
Tonight: Clear, Low 55F
Tomorrow: Partly Cloudy
High 68F, Low 52F

The warm weather pattern
is expected to continue
through the weekend with
temperatures reaching
the mid-70s by Sunday."""
render_text(draw, right_text, 800, 100, nfont, 700, color=80)

# Horizontal rule
draw.line([(30, 85), (img.width-30, 85)], fill=0, width=2)
# Vertical divider between columns
draw.line([(770, 30), (770, img.height-30)], fill=100, width=1)

img = add_noise(img, 0.015)
img = add_skew(img, 0.5)
img.save(os.path.join(test_dir, "c5_newspaper.pdf"), "PDF", resolution=200)
print("  c5_newspaper.pdf: OK")

# ===== 6. STAMPED/SEALED DOCUMENT =====
print("6. Document with stamp/seal overlay...")
img = Image.new("RGB", (IMG_W, IMG_H), "white")
draw = ImageDraw.Draw(img)
try:
    sfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 28)
except:
    sfont = ImageFont.load_default()
contract_text = """CONTRACT AGREEMENT

This agreement is entered into on June 1, 2024 between:

Party A: ABC Corporation
123 Business Ave, Suite 400
New York, NY 10001

Party B: XYZ Solutions Ltd.
456 Commerce Dr.
San Francisco, CA 94105

Terms and Conditions:
1. Party A agrees to provide consulting services to Party B
2. The term of this agreement shall be 12 months
3. Payment terms: Net 30 days from invoice date
4. Either party may terminate with 30 days written notice

IN WITNESS WHEREOF, the parties have executed this agreement
on the date first written above.

______________________             ______________________
For ABC Corporation                For XYZ Solutions Ltd."""
render_text(draw, contract_text, 50, 50, sfont, IMG_W - 100)
img = add_stamp(img, "APPROVED")
img = add_stamp(img, "CONFIDENTIAL")
img = add_noise(img, 0.01)
img.save(os.path.join(test_dir, "c6_stamped_doc.pdf"), "PDF", resolution=200)
print("  c6_stamped_doc.pdf: OK")

print(f"\nDone! Generated {6} real-world OCR challenge PDFs:")
for f in sorted(os.listdir(test_dir)):
    if f.startswith("c"):
        sz = os.path.getsize(os.path.join(test_dir, f)) // 1024
        print(f"  {f}: {sz}KB")
