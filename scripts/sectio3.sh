echo "00001 00001 00001 00001"
python src/section3/main.py     --env MountainCarContinuous-v0     --exp_name mountaincar_exp     --gamma 0.99     --learning_rate_actor 0.00001     --learning_rate_critic 0.0001     --episodes 2000     --max_steps 2000         --models_dir assets/section_3/actor_critic_00001_00001_00001_00001/mountain_car/models --h11_w1 0.00001 --h11_w2 0.00001 --h12_w1 0.00001 --h12_w2 0.00001 --early_exit 90

python src/section3/main.py     --env MountainCarContinuous-v0     --exp_name mountaincar_exp     --gamma 0.99     --learning_rate_actor 0.00001     --learning_rate_critic 0.0001     --episodes 2000     --max_steps 2000         --models_dir assets/section_3/actor_critic_001_001_001_001/mountain_car/models --h11_w1 0.001 --h11_w2 0.001 --h12_w1 0.001 --h12_w2 0.001 --early_exit 90
echo "001 001 001 001"
python src/section3/main.py     --env MountainCarContinuous-v0     --exp_name mountaincar_exp     --gamma 0.99     --learning_rate_actor 0.00001     --learning_rate_critic 0.0001     --episodes 2000     --max_steps 2000         --models_dir assets/section_3/actor_critic_0001_0001_0001_0001/mountain_car/models --h11_w1 0.0001 --h11_w2 0.0001 --h12_w1 0.0001 --h12_w2 0.0001 --early_exit 90
echo "01 01 01 01"
python src/section3/main.py     --env MountainCarContinuous-v0     --exp_name mountaincar_exp     --gamma 0.99     --learning_rate_actor 0.00001     --learning_rate_critic 0.0001     --episodes 2000     --max_steps 2000         --models_dir assets/section_3/actor_critic_01_01_01_01/mountain_car/models --h11_w1 0.01 --h11_w2 0.01 --h12_w1 0.01 --h12_w2 0.01 --early_exit 90

