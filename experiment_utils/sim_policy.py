import joblib
import tensorflow as tf
import argparse
import os.path as osp
from learning_to_adapt.samplers.utils import rollout
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str, help='Directory with the pkl and json file')
    parser.add_argument('--max_path_length', '-l', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', '-n', type=int, default=10,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    parser.add_argument('--ignore_done', action='store_true',
                        help='Whether stop animation when environment done or continue anyway')
    args = parser.parse_args()

    with tf.Session() as sess:
        pkl_path = osp.join(args.param, 'params.pkl')
        json_path = osp.join(args.param, 'params.json')
        print("Testing policy %s" % pkl_path)
        json_params = json.load(open(json_path, 'r'))
        data = joblib.load(pkl_path)
        policy = data['policy']
        env = data['env']
        for _ in range(args.num_rollouts):
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, ignore_done=args.ignore_done,
                           adapt_batch_size=json_params.get('adapt_batch_size', None))
            # print(sum(path['rewards']))
