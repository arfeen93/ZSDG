CUDA_VISIBLE_DEVICES=0 python3 main.py -hd sketch -tr 0 -log 400 -bs 60 -wfeat 2 -beta 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -hd infograph -tr 0 -log 400 -bs 60 -wfeat 2 -beta 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -hd quickdraw -tr 0 -log 400 -bs 60 -wfeat 2 -beta 1