# ================================================================
# 0)  Imports & helpers
# ================================================================
import torch, torchvision as tv, pandas as pd, requests, io, os
from PIL import Image
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# tiny helper to fetch images by URL (streaming, no disk)
def fetch_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        return None      # caller decides what to do on failure

# ================================================================
# 1)  Download SBU Captions (1 M pairs, ~6 GB JPEGs when fetched)
#     => we only pull the TSV index now; images fetched on‑the‑fly
# ================================================================
SBU_URL = "https://cs.stanford.edu/people/karpathy/sbucaptions/sbucaptions.tsv.gz"
tsv_path = "sbu_captions.tsv.gz"

if not os.path.exists(tsv_path):
    print("Downloading SBU captions index …")
    r = requests.get(SBU_URL, stream=True)
    with open(tsv_path, "wb") as fh:
        for chunk in tqdm(r.iter_content(chunk_size=1<<20)):
            fh.write(chunk)

sbu_df = pd.read_csv(tsv_path, sep="\t", names=["url", "caption"])
print("SBU index loaded:", len(sbu_df), "rows")

# ================================================================
# 2)  Image encoder — ResNet‑50 (ImageNet‑1K pretrained)
# ================================================================
resnet50 = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
resnet50.fc = torch.nn.Identity()                 # get 2048‑D penultimate layer
resnet50 = resnet50.to(device).eval()

# standard 224×224 preprocessing
preproc = tv.transforms.Compose([
    tv.transforms.Resize(256, interpolation=tv.transforms.InterpolationMode.BICUBIC),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225]),
])

def encode_image_pil(img_pil):
    with torch.no_grad():
        x = preproc(img_pil).unsqueeze(0).to(device)
        return torch.nn.functional.normalize(resnet50(x), dim=-1).cpu()  # [1,2048]

# ================================================================
# 3)  Text encoder — MiniLM‑L6‑v2 (Sentence‑BERT)
# ================================================================
text_enc = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
text_enc.eval()                    # 384‑D normalised sentence embeddings

def encode_text(sent):
    with torch.no_grad():
        return torch.tensor(text_enc.encode(sent, convert_to_numpy=True)).unsqueeze(0)  # [1,384]

# ================================================================
# 4)  Quick sanity test on N = 8 samples
# ================================================================
examples = sbu_df.sample(8, random_state=0).to_dict("records")

for row in examples:
    img = fetch_image(row["url"])
    if img is None:
        continue
    img_vec = encode_image_pil(img)
    txt_vec = encode_text(row["caption"])
    cos = torch.nn.functional.cosine_similarity(img_vec, txt_vec).item()
    print(f"{cos: .3f}  |  {row['caption'][:60]}")

# ------------------------------------------------
# You now have:
#   encode_image_pil(...)  -> 2048‑D torch tensor
#   encode_text(...)       -> 384‑D torch tensor
# ready to feed into your Relative‑Representation layer.
# ------------------------------------------------
