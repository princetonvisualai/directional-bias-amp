for i in 7 39
do 
    for j in 0 1 2 3 4 # number of runs per model
    do
        # training models for model comparison
        python3 classifier.py --save-model --model_type vgg_orig --model_name VGG16/$i\_$j --epochs 10 --batch-size 64 --attribute_set -$i --training_version 0 --lr .00014
        python3 classifier.py --save-model --model_type alexnet --model_name AlexNet/$i\_$j --epochs 10 --batch-size 64 --attribute_set -$i --training_version 0 --lr .0003
        python3 classifier.py --save-model --model_type resnet18 --model_name ResNet18/$i\_$j --epochs 10 --batch-size 64 --attribute_set -$i --training_version 0 --lr .0001

        # training resnet18 with different ratios for ratio comparison
        python3 classifier.py --save-model --model_type resnet18 --model_name resnet_a/$i\_$j --epochs 10 --criterion 0 --batch-size 64 --attribute_set -$i --training_version 0 --lr .00014 --weighting_version 2 --weighting_ratio 1.5
        python3 classifier.py --save-model --model_type resnet18 --model_name resnet_b/$i\_$j --epochs 10 --criterion 0 --batch-size 64 --attribute_set -$i --training_version 0 --lr .00014 --weighting_version 2 --weighting_ratio 1.75
        python3 classifier.py --save-model --model_type resnet18 --model_name resnet_c/$i\_$j --epochs 10 --criterion 0 --batch-size 64 --attribute_set -$i --training_version 0 --lr .00014 --weighting_version 2 --weighting_ratio 2.0
        python3 classifier.py --save-model --model_type resnet18 --model_name resnet_d/$i\_$j --epochs 10 --criterion 0 --batch-size 64 --attribute_set -$i --training_version 0 --lr .00014 --weighting_version 2 --weighting_ratio 2.25
        python3 classifier.py --save-model --model_type resnet18 --model_name resnet_e/$i\_$j --epochs 10 --criterion 0 --batch-size 64 --attribute_set -$i --training_version 0 --lr .00014 --weighting_version 2 --weighting_ratio 2.5

    done
done

# to run analysis on model outputs 
python3 make_graphs.py --attribute 7
python3 make_graphs.py --attribute 39

# to run analysis on ratio outputs
python3 make_graphs.py --attribute 7 --ratios_not_models
python3 make_graphs.py --attribute 39 --ratios_not_models
