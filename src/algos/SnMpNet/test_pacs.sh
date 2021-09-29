CUDA_VISIBLE_DEVICES=0 python3 test.py -hd cartoon -log 10 -dg 1 -data PACS -arch resnet50 -bs 60
CUDA_VISIBLE_DEVICES=0 python3 test.py -hd cartoon -log 10 -dg 1 -data PACS -arch resnet18 -bs 60

CUDA_VISIBLE_DEVICES=0 python3 test.py -hd art_painting -log 10 -dg 1 -data PACS -arch resnet50 -bs 60
CUDA_VISIBLE_DEVICES=0 python3 test.py -hd art_painting -log 10 -dg 1 -data PACS -arch resnet18 -bs 60

CUDA_VISIBLE_DEVICES=0 python3 test.py -hd sketch -log 10 -dg 1 -data PACS -arch resnet50 -bs 60
CUDA_VISIBLE_DEVICES=0 python3 test.py -hd sketch -log 10 -dg 1 -data PACS -arch resnet18 -bs 60

CUDA_VISIBLE_DEVICES=0 python3 test.py -hd photo -log 10 -dg 1 -data PACS -arch resnet50 -bs 60
CUDA_VISIBLE_DEVICES=0 python3 test.py -hd photo -log 10 -dg 1 -data PACS -arch resnet18 -bs 60