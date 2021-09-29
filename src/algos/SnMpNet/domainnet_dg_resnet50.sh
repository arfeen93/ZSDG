CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd real -lr 1e-3 -log 100 -data DomainNet -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd painting -lr 1e-3 -log 100 -data DomainNet -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd sketch -lr 1e-3 -log 100 -data DomainNet -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd quickdraw -lr 1e-3 -log 100 -data DomainNet -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd clipart -lr 1e-3 -log 100 -data DomainNet -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -arch resnet50 -hd infograph -lr 1e-3 -log 100 -data DomainNet -tr 0 -wcce 1 -wmse 1 -wrat 1 -alpha 2 -dg 1