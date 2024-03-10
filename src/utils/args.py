import argparse


def finetune_parser():
    parser = argparse.ArgumentParser(description='Fine tune the pretrained models')
    parser.add_argument('-w', '--workers', default=20, type=int, 
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-e', '--epochs', default=10, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N')
    parser.add_argument('-w', '--weight_path', type=str, default=None,
                        help='initial weight path')
    parser.add_argument('-o', '--optimizer', default=None, type=str,
                        help="choose loss function from ['SAM'] optimization method")
    parser.add_argument('-m', '--model', default=None, type=str,
                        help='choose the source model')
    parser.add_argument('-l', '--learning_rate', default=None, type=float,
                        help='choose learning rate for fine_tune') 
    args = parser.parse_args()
    return args


def SASD_parser():
    parser = argparse.ArgumentParser(description='Single teacher distillation.')
    parser.add_argument('-m', '--model', default="resnet50", type=str, 
                        help='choose the source model.')
    parser.add_argument('-b', '--batch_size', default=72, type=int, 
                        help='choose the batch size.')
    parser.add_argument('-e', '--epochs', default=10, type=int, 
                        help='choose the epochs.')
    parser.add_argument('-t', '--temperature', default=1, type=int, 
                        help='choose the temperature.')
    parser.add_argument('-w', '--workers', default=40, type=int, 
                        help='choose the number of workers.')
    parser.add_argument('-s', '--sharpness_aware', action="store_true",
                        help='choose whether to use SAM.')
    parser.add_argument('-o', '--offset', default=1, type=int, 
                        help='choose the offset.')
    parser.add_argument('-l', '--learning_rate', default=0.05, type=float, 
                        help='choose the distillation learning rate')
    args = parser.parse_args()
    return args


def eval_parser():
    parser = argparse.ArgumentParser(description='Ensemble Baseline')
    parser.add_argument('-w', '--workers', default=16, type=int, 
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch_size', default=10, type=int, 
                        help='mini-batch size')
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('-r', '--scaling_ratio', default=1, type=float, 
                        help="choose the scale percentage, 1 for not scale")
    parser.add_argument('-s', '--save_path', default=None, type=str, 
                        help="choose the path to save the adversarial examples")
    parser.add_argument('--lossfunc', default="ce", type=str, 
                        help="choose the loss function, ce or logit")
    parser.add_argument('--learning_rate', default=2/255, type=float,
                        help='choose the step size of perturbation generation')
    parser.add_argument('--epsilon', default=16, type=float,
                        help='choose the L_inf norm bound of perturbation')
    parser.add_argument('--src_type', default='pretrain', type=str, choices=['SASD', 'pretrain', 'GN', 'LGV'],
                        help="choose the type of source models, pretrain, SASD, LGV or GN")
    parser.add_argument('-i', '--max_iterations', default=200, type=int, 
                        help="choose the max iterations")
    parser.add_argument('--seed', default=42, type=int,
                        help='choose the random seed')
    parser.add_argument('--DI', action="store_true",
                        help='Diversify input images')
    parser.add_argument('--TI', action="store_true",
                        help='Conduct translation invariant transformation')
    parser.add_argument('--MI', action="store_true",
                        help='Conduct momentum iteration')
    parser.add_argument('--Admix', action="store_true",
                        help='Conduct multimodal inputs')
    args = parser.parse_args()
    return args


def eval_single_parser():
    parser = argparse.ArgumentParser(description='Single Baseline')
    parser.add_argument('-w', '--workers', default=16, type=int, 
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch_size', default=10, type=int, 
                        help='mini-batch size')
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('-r', '--scaling_ratio', default=1, type=float, 
                        help="choose the scale percentage, 1 for not scale")
    parser.add_argument('--lossfunc', default="ce", type=str, 
                        help="choose the loss function, ce or logit")
    parser.add_argument('--learning_rate', default=2/255, type=float,
                        help='choose the step size of perturbation generation')
    parser.add_argument('--epsilon', default=16, type=float,
                        help='choose the L_inf norm bound of perturbation')
    parser.add_argument('--src_type', default='pretrain', type=str, choices=['SASD', 'pretrain', 'GN', 'LGV'],
                        help="choose the type of source models, pretrain, SASD, LGV or GN")
    parser.add_argument('--src_name', default='resnet50', type=str, choices=['resnet50', 'densenet121', 'vgg16_bn', 'inception_v3'],
                        help="choose the source model")
    parser.add_argument('-i', '--max_iterations', default=400, type=int, 
                        help="choose the max iterations")
    parser.add_argument('--seed', default=42, type=int,
                        help='choose the random seed')
    parser.add_argument('--DI', action="store_true",
                        help='Diversify input images')
    parser.add_argument('--TI', action="store_true",
                        help='Conduct translation invariant transformation')
    parser.add_argument('--MI', action="store_true",
                        help='Conduct momentum iteration')
    parser.add_argument('--Admix', action="store_true",
                        help='Conduct multimodal inputs')
    args = parser.parse_args()
    return args


def rap_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model', type=str, default='resnet50', choices=['resnet50', 'inception-v3', 'densenet121', 'vgg16bn', 'ens6'])
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_iterations', type=int, default=400)
    parser.add_argument('--loss_function', type=str, default='CE', choices=['CE','MaxLogit'])
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('-w', '--workers', type=int, default=20)
    parser.add_argument('--m1', type=int, default=1, help='number of randomly sampled images')
    parser.add_argument('--m2', type=int, default=1, help='num of copies')
    parser.add_argument('--strength', type=float, default=0)
    parser.add_argument('--adv_perturbation', action='store_true')
    parser.add_argument('--adv_loss_function', type=str, default='CE', choices=['CE', 'MaxLogit'])
    parser.add_argument('--adv_epsilon', type=eval, default=16/255)
    parser.add_argument('--adv_steps', type=int, default=8)
    parser.add_argument('--transpoint', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--MI', action='store_true')
    parser.add_argument('--DI', action='store_true')
    parser.add_argument('--TI', action='store_true')
    parser.add_argument('--SI', action='store_true')
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--file_path', type=str, default=None)
    args = parser.parse_args()
    return args