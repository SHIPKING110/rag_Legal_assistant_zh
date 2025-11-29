#模型下载
from modelscope import snapshot_download
#这个BAAI重排序模型有6.74GB，用于高精度场景。
#model_dir = snapshot_download('BAAI/bge-reranker-large',cache="./rag_falv_ch/model/rank")
#在多个垂直领域（如金融、法律）进行高性能、低成本的重排。
#model_dir = snapshot_download('zeroentropy/zerank-1-small',cache_dir="./model/rank")

model_dir = snapshot_download('Qwen/Qwen3-Reranker-0.6B',cache_dir="./model/rank")
