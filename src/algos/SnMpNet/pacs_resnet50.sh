# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd art_painting -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd cartoon -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd sketch -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd photo -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1

# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd art_painting -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd cartoon -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd sketch -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd photo -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1

# CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet50 -hd art_painting -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 0.5 -alpha 2 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd cartoon -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 0.5 -alpha 1 -dg 1
# CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet50 -hd sketch -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 0.5 -alpha 2 -dg 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd photo -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 0.5 -alpha 2 -dg 1

CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd art_painting -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 0 -wrat 0 -alpha 0 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd cartoon -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 0 -wrat 0 -alpha 0 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd sketch -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 0 -wrat 0 -alpha 0 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd photo -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 0 -wrat 0 -alpha 0 -dg 1