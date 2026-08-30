#!/usr/bin/env python3
"""Generate the repo social card (1200x630 PNG) — no external assets needed."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (13, 17, 23)          # GitHub dark
GREEN = (118, 185, 0)      # NVIDIA green
FG = (230, 237, 243)
DIM = (139, 148, 158)
CHIP_BG = (22, 27, 34)
CHIP_EDGE = (48, 54, 61)

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# subtle green glow band on the left
for i in range(14):
    a = 1 - i / 14
    d.rectangle([0, 0, int(6 + i * 10), H], fill=(int(BG[0] + (GREEN[0]-BG[0])*a*0.14),
                                                   int(BG[1] + (GREEN[1]-BG[1])*a*0.14),
                                                   int(BG[2] + (GREEN[2]-BG[2])*a*0.14)))
d.rectangle([0, 0, 6, H], fill=GREEN)

f_title = ImageFont.truetype(FB, 72)
f_sub = ImageFont.truetype(FR, 34)
f_stat = ImageFont.truetype(FM, 28)
f_label = ImageFont.truetype(FR, 20)
f_url = ImageFont.truetype(FM, 24)

x = 80
d.text((x, 90), "Qwen3.8-Flash-Next", font=f_title, fill=FG)
d.text((x, 190), "176B MoE on a single NVIDIA DGX Spark", font=f_sub, fill=GREEN)
d.text((x, 245), "vLLM · NVFP4 + fp8 hybrid · MTP speculative decoding", font=f_sub, fill=DIM)

stats = [("125B+51B", "params · 6B active"),
         ("21.6 tok/s", "decode · single stream"),
         ("262k", "native context"),
         ("128 GB", "unified GB10 memory")]
cx, cy = x, 350
for value, label in stats:
    tw = d.textlength(value, font=f_stat)
    lw = d.textlength(label, font=f_label)
    w = int(max(tw, lw)) + 48
    d.rounded_rectangle([cx, cy, cx + w, cy + 112], radius=14, fill=CHIP_BG, outline=CHIP_EDGE, width=2)
    d.text((cx + 24, cy + 22), value, font=f_stat, fill=FG)
    d.text((cx + 24, cy + 66), label, font=f_label, fill=DIM)
    cx += w + 20

d.text((x, 545), "github.com/madeye/qwen38-flash-next-on-dgx-spark", font=f_url, fill=DIM)

img.save("social-card.png", optimize=True)
print("wrote social-card.png", img.size)
