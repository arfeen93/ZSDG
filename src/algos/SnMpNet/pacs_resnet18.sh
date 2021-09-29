CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd art_painting -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1
CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd cartoon -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1
CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd sketch -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1
CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd photo -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 1 -dg 1

CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd art_painting -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd cartoon -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd sketch -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=1 python3 main.py -arch resnet18 -hd photo -lr 1e-3 -log 10 -data PACS -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1