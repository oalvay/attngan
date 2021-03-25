.ONESHELL:
.PHONY: oalvay
oalvay:
	. ~/miniconda3/etc/profile.d/conda.sh
	conda activate clip
	python main.py --cfg cfg/$(name).yml --gpu 0

.PHONY: oweii
oweii:
	. ~/miniconda3/etc/profile.d/conda.sh
	conda activate clip
	python main_clip.py --cfg cfg/$(name).yml --gpu 0

.PHONY: damsm
damsm:
	. ~/miniconda3/etc/profile.d/conda.sh
	conda activate clip
	python pretrain_DAMSM.py --cfg cfg/DAMSM/face.yml --gpu 0 > log1.txt

.PHONY: eval
eval:
	. ~/miniconda3/etc/profile.d/conda.sh
	conda activate clip
	python $(name).py --cfg cfg/eval_face.yml
