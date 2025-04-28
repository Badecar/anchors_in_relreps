# -----------------------------------------------------------
# 0.  install deps   (uncomment if you haven’t)
# -----------------------------------------------------------
# !pip install torch torchvision sentence-transformers pandas pillow requests tqdm

# -----------------------------------------------------------
# 1.  encoders: ResNet‑50  &  MiniLM‑L6‑v2
# -----------------------------------------------------------
import torch, torchvision.models as tvm
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- image encoder -----------------------------------------
img_enc = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
img_enc.fc = torch.nn.Identity()          # keep 2048‑D global‑avg features
img_enc = img_enc.to(device).eval()

# --- text encoder ------------------------------------------
txt_enc = SentenceTransformer('paraphrase-MiniLM-L6-v2',
                              device=device)        # 384‑D sentence vectors
txt_enc.eval()

# (both encoders are frozen -> no gradients needed)
for p in img_enc.parameters(): p.requires_grad_(False)
for p in txt_enc.parameters(): p.requires_grad_(False)

# -----------------------------------------------------------
# 2.  CC3M loader  (download TSV once, then stream)
#     * each row:  url \t caption
# -----------------------------------------------------------
import pandas as pd, requests, io, os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

CC3M_TSV = "cc3m_train.tsv"          # path to local TSV file
IMG_ROOT = "./cc3m_cache"            # folder to cache downloaded images
os.makedirs(IMG_ROOT, exist_ok=True)

df = pd.read_csv(CC3M_TSV, sep='\t', names=['img_url', 'caption'])

# torchvision preprocessing (resize -> centre‑crop -> tensor -> normalise)
preproc = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

def stream_cc3m(pairs=df.itertuples(index=False), max_errs=100):
    """Yields (img_tensor, caption_string)"""
    err = 0
    for row in tqdm(pairs, total=len(df)):
        url, cap = row
        fname = os.path.join(IMG_ROOT, os.path.basename(url))
        try:
            if not os.path.exists(fname):
                r = requests.get(url, timeout=5)
                r.raise_for_status()
                open(fname, 'wb').write(r.content)
            img = Image.open(fname).convert('RGB')
            yield preproc(img).unsqueeze(0).to(device), cap
        except Exception:
            err += 1
            if err > max_errs: break
            continue

# -----------------------------------------------------------
# 3.  One‑pass demo: encode first 5 pairs
# -----------------------------------------------------------
with torch.no_grad():
    for n, (img_t, cap) in enumerate(stream_cc3m()):
        if n == 5: break
        z_img = img_enc(img_t)                     # [1,2048]
        z_txt = txt_enc.encode(cap, convert_to_tensor=True)  # [384]
        print(f"[{n}]  img‑feat shape {z_img.shape}   |   txt‑feat shape {z_txt.shape}")
