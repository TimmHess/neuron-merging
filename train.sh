# Model Training
#python3 main.py --arch LeNet_300_100 --dataset MNIST --batch-size 32 --epochs 5 --model-type original --pruning-ratio 0.0 --lr 0.01 --log-interval 500

# Merging Test
#python3 main.py --arch LeNet_300_100 --dataset MNIST --batch-size 32 --epochs 10 --model-type merge --retrain --evaluate --pretrained saved_models/LeNet_300_100.MNIST.original.pth.tar --pruning-ratio 0.4 --threshold -1.0 --lamda 0.8 --criterion l1-norm --target ip

# Debugging
#python3 main.py --arch VGG --dataset cifar10 --pretrained saved_models/VGG.cifar10.original.pth.tar --model-type original --num_classes 10



##########################################
# SimpleCNN Train MNIST
#python3 main.py --arch SimpleCNN --dataset MNIST --batch-size 32 --epochs 5 --model-type original --pruning-ratio 0.0 --lr 0.01 --log-interval 500

# SimpleCNN Merge Test
#python3 main.py --arch SimpleCNN --dataset MNIST --batch-size 32 --epochs 1 --model-type merge --retrain --evaluate --pretrained saved_models/SimpleCNN.MNIST.original.pth.tar --pruning-ratio 0.5 --threshold -1.0 --lamda 0.8 --criterion l1-norm

# Simple CL Test
#python3 main_cl_mnist.py --arch SimpleCNN --dataset MNIST --batch-size 32 --log-interval 500 --retrain --pretrained saved_models/SimpleCNN.MNIST.original.pth.tar --threshold 1.0 --criterion l1-norm --lr 0.01 --seed 42 --pruning-ratio 1 --epochs 3

##########################################
# PreTrain SimpleCNN 
#python3 main_cl.py --arch SimpleCNN --dataset DecrasingLighting2 --batch-size 32 --epochs 10 --model-type original --pruning-ratio 0.0 --lr 0.001 --log-interval 200 --num_classes 5 --seq_order 4

# CL Tests
python3 main_cl.py --arch SimpleCNN --dataset DecrasingLighting --batch-size 32 --epochs 5 --model-type original --pruning-ratio 0.0 --lr 0.001 --log-interval 200 --num_classes 5 --evaluate --pretrained saved_models/SimpleCNN.DecrasingLighting2.original.pth.tar --seq_order 0 1 2 3 4