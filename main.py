import sys
from model.trainer import trainer
from model.utils import setup_training, get_config_path

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "CartPole"
    config_path = get_config_path(config_file)

    try:
        args = setup_training(config_path)
        trainer(args)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}.yaml' not found in 'config/' directory.")