import os
from opts import Options
import time
import shutil

if __name__ == '__main__':
    parser = Options()
    (configuration, args) = parser.parse_args()
    #display = configuration.root + configuration.name + '/display.txt'
    configuration.root = configuration.root + configuration.name
    configuration.cp_dest = configuration.root + configuration.cp_dest
    configuration.im_dest = configuration.root + configuration.im_dest
    configuration.tb_dest = configuration.root + configuration.tb_dest

    if not os.path.exists(configuration.root):
        os.makedirs(configuration.root)

    if not os.path.exists(configuration.cp_dest):
        os.makedirs(configuration.cp_dest)

    if not os.path.exists(configuration.im_dest):
        os.makedirs(configuration.im_dest)

    if not os.path.exists(configuration.tb_dest):
        os.makedirs(configuration.tb_dest)

    if os.path.isfile(configuration.root + ('/opts_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/opts_%s.py' % configuration.name))
    shutil.copy2('opts.py', configuration.root + ('/opts_%s.py' % configuration.name))
    if configuration.prenet:
        if os.path.isfile(configuration.root + ('/opts2_%s.py' % configuration.name)):
            os.remove(configuration.root + ('/opts2_%s.py' % configuration.name))
        shutil.copy2('opts2.py', configuration.root + ('/opts2_%s.py' % configuration.name))
    if os.path.isfile(configuration.root + ('/train_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/train_%s.py' % configuration.name))
    shutil.copy2('train.py', configuration.root + ('/train_%s.py' % configuration.name))
    train = 'train_%s.py' % configuration.name
    if configuration.preprocess_filename != 'None' and os.path.isfile(configuration.preprocess_filename):
        print('2D_slices was copied')
        shutil.copy2(configuration.preprocess_filename, configuration.cp_dest + '2D_slices.npz')
    #if os.path.exists(configuration.root + '/models'):
    #    shutil.rmtree(configuration.root + '/models')
    #shutil.copytree('models', configuration.root + '/models')
    # --constraint="12g" -p undergrad --qos=short"pascal
    os.system('sbatch -p turing --gres=gpu:%d train_shell.sh %s %s' % (configuration.gpu_devices, configuration.root, train))