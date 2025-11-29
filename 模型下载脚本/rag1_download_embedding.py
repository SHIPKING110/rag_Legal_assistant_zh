#模型下载
from modelscope.hub.snapshot_download import snapshot_download
 
model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='./model/embedding_model')
print("模型已下载至：", model_dir)