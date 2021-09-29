CUDA_VISIBLE_DEVICES=1 python3 trainer_tent.py -hd clipart -bs 60 -lrc 1e-6 -tstep 1 -log 40
CUDA_VISIBLE_DEVICES=1 python3 trainer_tent.py -hd infograph -bs 60 -lrc 1e-6 -tstep 1 -log 40
CUDA_VISIBLE_DEVICES=1 python3 trainer_tent.py -hd sketch -bs 60 -lrc 1e-6 -tstep 1 -log 40
CUDA_VISIBLE_DEVICES=1 python3 trainer_tent.py -hd quickdraw -bs 60 -lrc 1e-6 -tstep 1 -log 40
CUDA_VISIBLE_DEVICES=1 python3 trainer_tent.py -hd painting -bs 60 -lrc 1e-6 -tstep 1 -log 40