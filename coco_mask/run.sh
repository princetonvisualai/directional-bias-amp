for i in {0..4}
do 
    for j in 0 9 7
    do 
        python3 classifier.py --model_type vgg16 --save-model --model_name mask$j\_$i --mask_person $j --lr .05 --epochs 12
    done
done

python3 interpret_coco.py

