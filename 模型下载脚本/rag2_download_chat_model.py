#模型下载
from modelscope.hub.snapshot_download import snapshot_download
 
model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='./model')
print("模型已下载至：", model_dir)