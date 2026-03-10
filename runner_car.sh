#TODO, set path to directory containing pretrained flow, e.g.
flow_path="trained_flows/MyCarRacing-v1-2026-02-12_14-25-43"

for seed in 1 2 3; do
    # standard SAC
    python3 online-sac.py --seed $seed --config configs/car_racing.yaml

    # SAC with imitation learning regularization
    python3 online-sac.py --seed $seed --config configs/car_racing.yaml --il_coef 2.0

    # SAC with imitation learning regularization and Q-filter
    python3 online-sac.py --seed $seed --config configs/car_racing.yaml --il_coef 2.0 --il_use_q_filter

    # PARROT
    python3 online-apc.py --seed $seed --config configs/car_racing.yaml --parrot --pretrained_flow_paths $flow_path  

    # APC
    python3 online-apc.py --seed $seed --config configs/car_racing.yaml --pretrained_flow_paths $flow_path
done
