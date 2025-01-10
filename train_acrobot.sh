export PYTHONPATH=$(pwd)
python src/section1/train_actor_critic_agent.py \
--env Acrobot-v1 \
--early_exit -90 \
--models_dir assets/section_1/actor_critic/acrobot/models
