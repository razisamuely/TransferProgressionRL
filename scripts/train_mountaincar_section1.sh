# python src/section1/train_actor_critic_agent.py \
#     --env MountainCarContinuous-v0 \
#     --exp_name mountaincar_exp \
#     --gamma 0.99 \
#     --learning_rate_actor 0.00001 \
#     --learning_rate_critic 0.0001 \
#     --episodes 2000 \
#     --max_steps 2000 \
#     --models_dir assets/section_1/actor_critic/mountain_car/models



python src/section1/train_actor_critic_agent.py \
--env MountainCarContinuous-v0 \
--exp_name mountaincar_exp \
--models_dir assets/section_1/actor_critic/mountain_car/models




# python visualize_agent.py \
# --env MountainCarContinuous-v0 \
# --models_dir assets/section_1/actor_critic/mountain_car/models \
# --episode 1000 \
# --num_episodes 5
