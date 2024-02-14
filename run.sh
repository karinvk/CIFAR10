python ./datasets/dataloader.py \
--dataset CIFAR10 \
--batch_size 50

python ./models/model.py

python ./train.py \
--configs.yaml
