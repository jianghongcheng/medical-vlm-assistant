import sys, os, torch, numpy as np
import SimpleITK as sitk
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, '/media/max/b/LLM/Med3DVLM')
from src.model.CLIP import DEC_CLIP, DEC_CLIPConfig

CLIP_PATH = '/media/max/b/LLM/Med3DVLM/models/DCFormer_SigLIP'
DEMO_ROOT = '/media/max/b/LLM/Med3DVLM/data/demo'
COLLECTION = 'ct_image_rag'
device = torch.device('cuda')

print("Loading DCFormer-SigLIP...")
AutoConfig.register('dec_clip', DEC_CLIPConfig)
AutoModel.register(DEC_CLIPConfig, DEC_CLIP)
tokenizer = AutoTokenizer.from_pretrained(CLIP_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float32, trust_remote_code=True).to(device)
model.eval()
print("Model loaded!")

client = QdrantClient(host='localhost', port=6333)

with torch.no_grad():
    test_img = torch.zeros(1, 1, 10, 64, 64).float().to(device)
    test_emb = model.encode_image(test_img)
    emb_dim = test_emb.shape[-1]
print(f"Embedding dim: {emb_dim}")

try:
    client.delete_collection(COLLECTION)
except:
    pass

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE)
)
print(f"Collection created")

cases = []
for case_id in sorted(os.listdir(DEMO_ROOT)):
    case_dir = os.path.join(DEMO_ROOT, case_id)
    if not os.path.isdir(case_dir): continue
    nii_files = [f for f in os.listdir(case_dir) if f.endswith('.nii.gz')]
    txt_files = [f for f in os.listdir(case_dir) if f.endswith('.txt')]
    if nii_files and txt_files:
        cases.append({
            'case_id': case_id,
            'image': os.path.join(case_dir, nii_files[0]),
            'text': os.path.join(case_dir, txt_files[0])
        })

print(f"Found {len(cases)} cases")

points = []
for i, case in enumerate(cases):
    img = sitk.GetArrayFromImage(sitk.ReadImage(case['image']))
    img_np = np.expand_dims(img, axis=0).astype(np.float32)
    img_pt = torch.from_numpy(img_np).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = model.encode_image(img_pt)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    embedding = emb.cpu().numpy()[0].tolist()
    with open(case['text']) as f:
        report_text = f.read()
    points.append(PointStruct(
        id=i,
        vector=embedding,
        payload={
            'case_id': case['case_id'],
            'image_path': case['image'],
            'finding_summary': report_text[:200],
            'diagnosis': report_text[:80]
        }
    ))
    print(f"  [{i+1}/{len(cases)}] {case['case_id']}")

client.upsert(collection_name=COLLECTION, points=points)
count = client.count(collection_name=COLLECTION)
print(f"Done! {count.count} cases in Qdrant")
