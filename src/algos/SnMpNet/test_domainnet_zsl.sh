CUDA_VISIBLE_DEVICES=0 python3 test.py -data DomainNet -log 10 -dg 0 -arch resnet50 -bs 240 -tv 0 -hd painting
CUDA_VISIBLE_DEVICES=0 python3 test.py -data DomainNet -log 10 -dg 0 -arch resnet50 -bs 240 -tv 0 -hd real
CUDA_VISIBLE_DEVICES=0 python3 test.py -data DomainNet -log 10 -dg 0 -arch resnet50 -bs 240 -tv 0 -hd sketch
CUDA_VISIBLE_DEVICES=0 python3 test.py -data DomainNet -log 10 -dg 0 -arch resnet50 -bs 240 -tv 0 -hd quickdraw
CUDA_VISIBLE_DEVICES=0 python3 test.py -data DomainNet -log 10 -dg 0 -arch resnet50 -bs 240 -tv 0 -hd clipart
CUDA_VISIBLE_DEVICES=0 python3 test.py -data DomainNet -log 10 -dg 0 -arch resnet50 -bs 240 -tv 0 -hd infograph