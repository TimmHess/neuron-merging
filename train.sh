# Model Training
#python3 main.py --arch VGG --dataset cifar10 --batch-size 64 --epochs 20 --model-type original --pruning-ratio 0.0 --lr 0.01 

# Merging Test
#python3 main.py --arch VGG --dataset cifar10 --batch-size 64 --epochs 20 --model-type merge --retrain --evaluate --pretrained saved_models/VGG.cifar10.original.pth.tar --pruning-ratio 0.2 --threshold -1.0 --lamda 0.8 --criterion l1-norm --no_bn

# Debugging
python3 main.py --arch VGG --dataset cifar10 --pretrained saved_models/VGG.cifar10.original.pth.tar --model-type original