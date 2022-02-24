# CLIP-finetune

根据CLIP原文以及source code的issue写的训练过程。


# conda environment from OpanAI CLIP
https://github.com/openai/CLIP

#train/finetune：
python finetune/modified_train_float32.py

#test：
cd evaluation
python evluation.py

讨论：
1.遇到过灾难性遗忘(catastrophic forgetting)的问题，即用小规模的细粒度的图文对finetune之后，在原来的任务上效果下降了。
目前未解决。
