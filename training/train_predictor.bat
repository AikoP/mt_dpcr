python "./training/train_predictor.py"^
 --train_data=multi_faces^
 --model=cnet --model_cap=normal^
 --epochs=10 --batch_size=2 --lr=0.005^
 --safe_descent --dynamic_lr --diff_features_only^
