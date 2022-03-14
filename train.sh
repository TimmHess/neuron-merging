# Model Training
#python3 main.py --arch LeNet_300_100 --dataset MNIST --batch-size 32 --epochs 5 --model-type original --pruning-ratio 0.0 --lr 0.01 --log-interval 500

# Merging Test
python3 main.py --arch LeNet_300_100 --dataset MNIST --batch-size 32 --epochs 10 --model-type merge --retrain --evaluate --pretrained saved_models/LeNet_300_100.MNIST.original.pth.tar --pruning-ratio 0.4 --threshold -1.0 --lamda 0.8 --criterion l1-norm --target ip

# Debugging
#python3 main.py --arch VGG --dataset cifar10 --pretrained saved_models/VGG.cifar10.original.pth.tar --model-type original --num_classes 10
