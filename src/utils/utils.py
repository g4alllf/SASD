import os
import scipy.io as io


def auto_file_name(path, file_name):
    if not os.path.isdir(path):
        os.makedirs(path)
    basename, ext = os.path.splitext(file_name)
    i = 1
    new_name = file_name
    while os.path.exists(os.path.join(path, new_name)):
        new_name = f"{basename}({i}){ext}"
        i += 1
    return os.path.join(path, new_name)


def log_to_file(message, filename):
    with open(filename, 'a') as f:
        f.write(message + "\n")


def logging(args, s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.file_path, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')


def get_idx2label():
    synset = io.loadmat(os.path.abspath('./imagenet/meta.mat'))['synsets'][:,0]
    idx2label = [(synset[i][1][0], synset[i][2][0]) for i in range(1000)]
    idx2label.sort(key=lambda x: x[0]); idx2label = [e[1] for e in idx2label]
    return idx2label


def rap_initialize(args):
    args.adv_alpha = args.adv_epsilon / args.adv_steps
    exp_name = args.source_model + '_' + args.loss_function + '_'

    if args.targeted:
        exp_name += 'Targeted_'
    if args.MI:
        exp_name += 'MI_'
    if args.DI:
        exp_name += 'DI_'
    if args.TI:
        exp_name += 'TI_'
    if args.m1 != 1:
        exp_name += f'm1_{args.m1}_'
    if args.m2 != 1:
        exp_name += f'm2_{args.m2}_'
    if args.strength != 0:
        exp_name += 'Admix_'

    exp_name += str(args.transpoint)

    # for targeted attack, we need to conduct the untargeted attack during the inner loop.
    # for untargeted attack, we need to conduct the targeted attack (the true label) during the inner loop. 
    if not args.targeted:
        args.adv_targeted = 1
    else:
        args.adv_targeted = 0

    logging(args, exp_name.format())
    logging(args, 'Hyper-parameters: {}\n'.format(args.__dict__))
    return exp_name