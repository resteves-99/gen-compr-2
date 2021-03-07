import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)

    parser.add_argument('--type', type=str, choices='ae_base, ae_exp, ae_small', default='ae_base')
    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--prog-grow', action='store_true')
    # parser.add_argument('--use-linear', action='')

    parser.add_argument('--load_dir', type=str, default=None)

    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()
    return args
