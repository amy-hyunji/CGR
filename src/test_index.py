import faiss
import torch 
from time import time 
import numpy as np
import resource
import sys

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard)) ## /2 makes limit of hald

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def main():
    index_path = "/data/project/rw/contextualized_GENRE/dataset/bi.t5_large.hotpot.epoch8/tokId_emb.ES.False.PQ.False"
    start_time = time()
    print("loading index")
    index = faiss.read_index(index_path, faiss.IO_FLAG_ONDISK_SAME_DIR)

    print(f"index loading takes {time()-start_time}s")
    # reconst_fn = faiss.downcast_index(index.index).reconstruct
    # R = torch.FloatTensor(faiss.vector_to_array(faiss.downcast_VectorTransform(index.chain.at(0)).A).reshape(index.d, index.d))

    assert torch.cuda.is_available(), f"Cuda availability {torch.cuda.is_available()}"
    device = torch.device('cuda')
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 256
    quantizer = index_ivf.quantizer
    quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer)
    index_ivf.quantizer = quantizer_gpu
    # R = R.to(device)
    start_time = time()
    dummy_input = np.random.rand(1024).astype("float32").reshape(1,-1)
    b_scores,I = index.search(dummy_input, 3)
    print(f"searching takes {time()-start_time}s")
    print(b_scores)

if __name__ == '__main__':
    print("starting limit")
    memory_limit() # Limitates maximun memory usage to half
    print("limiting done")
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)