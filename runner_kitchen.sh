# TODO, set paths to directories containing per-task pre-trained flows, e.g.
flow_path_list=(
    "trained_flows/FrankaKitchen-v1-2026-02-18_09-49-04-microwave"
    "trained_flows/FrankaKitchen-v1-2026-02-18_09-51-46-kettle"
    "trained_flows/FrankaKitchen-v1-2026-02-18_09-52-51-light_switch"
    "trained_flows/FrankaKitchen-v1-2026-02-18_09-56-41-slide_cabinet"
    "trained_flows/FrankaKitchen-v1-2026-02-18_09-58-21-hinge_cabinet"
    "trained_flows/FrankaKitchen-v1-2026-02-18_10-03-38-top_burner"
    "trained_flows/FrankaKitchen-v1-2026-02-18_10-08-45-bottom_burner"
);

for task in "microwave" "kettle" "light_switch" "slide_cabinet" "hinge_cabinet" "top_burner" "bottom_burner"; do
    echo "Running franka kitchen task: $task"

    for seed in 1 2 3; do
        # standard SAC
        python3 online-sac.py --seed $seed --config configs/kitchen.yaml --env_task $task --tag "task:${task}-seed${seed}"

        for il_data_path in \
            "data/kitchen/microwave.pkl" \
            "data/kitchen/kettle.pkl" \
            "data/kitchen/light_switch.pkl" \
            "data/kitchen/slide_cabinet.pkl" \
            "data/kitchen/hinge_cabinet.pkl" \
            "data/kitchen/top_burner.pkl" \
            "data/kitchen/bottom_burner.pkl"; do
        
            # SAC with imitation learning regularization
            python3 online-sac.py --seed $seed --config configs/kitchen.yaml --env_task $task --il_coef 2.0 --il_data_paths $il_data_path --tag "task:${task}_ildata:$(basename $il_data_path .pkl)-seed${seed}"

            # SAC with imitation learning regularization and Q-filter
            python3 online-sac.py --seed $seed --config configs/kitchen.yaml --env_task $task --il_coef 2.0 --il_use_q_filter --il_data_paths $il_data_path --tag "task:${task}_ildata:$(basename $il_data_path .pkl)-seed${seed}"

        done

        for flow_path in "${flow_path_list[@]}"; do
            # PARROT
            python3 online-apc.py --seed $seed --config configs/kitchen.yaml --env_task $task --pretrained_flow_paths $flow_path --parrot --tag "task:${task}_flow:$(basename $flow_path)-seed${seed}" 

            # APC
            python3 online-apc.py --seed $seed --config configs/kitchen.yaml --env_task $task --pretrained_flow_paths $flow_path --tag "task:${task}_flow:$(basename $flow_path)-seed${seed}"

        done

    done

done

