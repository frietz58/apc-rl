# TODO, set paths to directories containing per-task pre-trained flows, e.g.
flow_path_list=(
    "trained_flows/MyPointMaze-v1-2026-02-16_09-04-24-bottom_left"
    "trained_flows/MyPointMaze-v1-2026-02-16_09-30-59-bottom_right"
    "trained_flows/MyPointMaze-v1-2026-02-16_09-42-57-top_left"
    "trained_flows/MyPointMaze-v1-2026-02-16_09-54-13-top_right"
);

for maze_task in "gtl" "gtr" "gbl" "gbr"; do
    echo "Running maze task: $maze_task"

    for seed in 1 2 3; do
        # standard SAC
        python3 online-sac.py --seed $seed --config configs/maze.yaml --env_task $maze_task --tag "task:${maze_task}-seed${seed}"

        for il_data_path in \
            "data/maze/goal_bottom_left.pkl" \
            "data/maze/goal_bottom_right.pkl" \
            "data/maze/goal_top_left.pkl" \
            "data/maze/goal_top_right.pkl"; do
        
            # SAC with imitation learning regularization
            python3 online-sac.py --seed $seed --config configs/maze.yaml --env_task $maze_task --il_coef 0.5 --il_data_paths $il_data_path --tag "task:${maze_task}_ildata:$(basename $il_data_path .pkl)-seed${seed}"

            # SAC with imitation learning regularization and Q-filter
            python3 online-sac.py --seed $seed --config configs/maze.yaml --env_task $maze_task --il_coef 0.5 --il_use_q_filter --il_data_paths $il_data_path --tag "task:${maze_task}_ildata:$(basename $il_data_path .pkl)-seed${seed}"

        done

        for flow_path in "${flow_path_list[@]}"; do

            # PARROT
            python3 online-apc.py --seed $seed --config configs/maze.yaml --env_task $maze_task --pretrained_flow_paths $flow_path --parrot --tag "task:${maze_task}_flow:$(basename $flow_path)-seed${seed}" 

            # APC
            python3 online-apc.py --seed $seed --config configs/maze.yaml --env_task $maze_task --pretrained_flow_paths $flow_path --tag "task:${maze_task}_flow:$(basename $flow_path)-seed${seed}"

        done

    done

done

