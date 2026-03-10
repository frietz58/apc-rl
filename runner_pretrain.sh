python3 pretrain_nflow.py --config configs/car_racing.yaml

python3 pretrain_nflow.py --config configs/maze.yaml --override_data_path "data/maze/goal_bottom_left.pkl" --tag "bottom_left"
python3 pretrain_nflow.py --config configs/maze.yaml --override_data_path "data/maze/goal_bottom_right.pkl" --tag "bottom_right"
python3 pretrain_nflow.py --config configs/maze.yaml --override_data_path "data/maze/goal_top_left.pkl" --tag "top_left"
python3 pretrain_nflow.py --config configs/maze.yaml --override_data_path "data/maze/goal_top_right.pkl" --tag "top_right"

python3 pretrain_nflow.py --config configs/kitchen.yaml --override_data_path "data/kitchen/microwave.pkl" --tag "microwave"
python3 pretrain_nflow.py --config configs/kitchen.yaml --override_data_path "data/kitchen/kettle.pkl" --tag "kettle"
python3 pretrain_nflow.py --config configs/kitchen.yaml --override_data_path "data/kitchen/light_switch.pkl" --tag "light_switch"
python3 pretrain_nflow.py --config configs/kitchen.yaml --override_data_path "data/kitchen/slide_cabinet.pkl" --tag "slide_cabinet"
python3 pretrain_nflow.py --config configs/kitchen.yaml --override_data_path "data/kitchen/hinge_cabinet.pkl" --tag "hinge_cabinet"
python3 pretrain_nflow.py --config configs/kitchen.yaml --override_data_path "data/kitchen/top_burner.pkl" --tag "top_burner"
python3 pretrain_nflow.py --config configs/kitchen.yaml --override_data_path "data/kitchen/bottom_burner.pkl" --tag "bottom_burner"


