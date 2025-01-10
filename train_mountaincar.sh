export PYTHONPATH=$(pwd)
python src/section1/train_actor_critic_agent.py \
--env MountainCarContinuous-v0 \
--early_exit 90 \
--models_dir assets/section_1x/actor_critic/mountain_car/models
