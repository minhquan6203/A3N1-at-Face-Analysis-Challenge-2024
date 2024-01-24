import yaml
import argparse
from typing import Text
from task.training import Classify_Task
from task.infer import Predict

def main(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    if config['phase']=='train':
        print("Training started...")
        task_train=Classify_Task(config)
        task_train.training()
        print("Training complete")
    else:
        print('Now evaluate on test data...')
        task_infer=Predict(config)
        task_infer.predict_submission()
        print('Task done!!!')
    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True, default='Face-Analysis-Challenge/config/trans_config.yaml')
    args = args_parser.parse_args()
    
    main(args.config)