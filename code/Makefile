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
	python main_clip.py --cfg cfg/eval_replace_loss_text.yml --gpu 0 --manualSeed $(seed)
	python main_clip.py --cfg cfg/eval_replace_text_only.yml --gpu 0 --manualSeed $(seed)
	python main_clip.py --cfg cfg/eval_replace_loss_only.yml --gpu 0 --manualSeed $(seed)
	python main_clip.py --cfg cfg/eval_mixed_loss_only.yml --gpu 0 --manualSeed $(seed)
	python main_clip.py --cfg cfg/eval_mixed_loss_text.yml --gpu 0 --manualSeed $(seed)
	python main_clip.py --cfg cfg/eval_original.yml --gpu 0 --manualSeed $(seed)
