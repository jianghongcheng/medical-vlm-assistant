import torch
import warnings
warnings.filterwarnings('ignore')
from merlin.data import download_sample_data, DataLoader
from merlin import Merlin

device = 'cpu'
print(f'Device: {device}')

model = Merlin(ImageEmbedding=True)
model.eval()
# 不调用 model.cuda()，直接用CPU

data_dir = 'abct_data'
cache_dir = 'abct_data_cache'

datalist = [{
    'image': download_sample_data(data_dir),
    'text': 'Normal CT scan.'
}]

dataloader = DataLoader(
    datalist=datalist,
    cache_dir=cache_dir,
    batchsize=1,
    shuffle=False,
    num_workers=0
)

for batch in dataloader:
    outputs = model(batch['image'].to(device))
    print(f'Image embedding shape: {outputs[0].shape}')
    print('SUCCESS: Merlin ImageEmbedding working on CPU!')
