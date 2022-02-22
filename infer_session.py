import os
from opts import Options
from opts2 import Options as Options2
import time
import shutil

if __name__ == '__main__':
    parser = Options()
    (configuration, args) = parser.parse_args()
    display = configuration.root + configuration.name + '/display.txt'
    configuration.root = configuration.root + configuration.name
    configuration.cp_dest = configuration.root + configuration.cp_dest
    configuration.im_dest = configuration.root + configuration.im_dest
    configuration.tb_dest = configuration.root + configuration.tb_dest

    if not os.path.exists(configuration.root):
        os.makedirs(configuration.root)

    if os.path.isfile(configuration.root + ('/opts_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/opts_%s.py' % configuration.name))
    shutil.copy2('opts.py', configuration.root + ('/opts_%s.py' % configuration.name))

    if os.path.isfile(configuration.root + ('/dataloader_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/dataloader_%s.py' % configuration.name))
    shutil.copy2('utils/dataloader.py', configuration.root + ('/dataloader_%s.py' % configuration.name))

    if os.path.isfile(configuration.root + ('/infer_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/infer_%s.py' % configuration.name))
    shutil.copy2('inferer.py', configuration.root + ('/infer_%s.py' % configuration.name))

    train = 'infer_%s.py' % configuration.name

    #if os.path.exists(configuration.root + '/models'):
    #    shutil.rmtree(configuration.root + '/models')
    #shutil.copytree('models', configuration.root + '/models')
    # --constraint="12g" -p undergrad --qos=short"
    #os.system('. ~/.anaconda')
    #time.sleep(10)
    # os.system('qsub -hold_jid depe -q all.q -P i13 -l gpu=%d -l gpu_arch=Volta -e %s train_shell.sh %s %s %s %s' % (configuration.gpu_devices, display, configuration.root, train, display, configuration.gpu_devices_sp))
    # --dependency=afterany:432265
    os.system('sbatch -c %d --mem %dG --gres=gpu:%d --output=%s --error=%s train_shell.sh %s %s %s' % (configuration.cores, configuration.memory, configuration.gpu_devices, display, display, configuration.root, train, display))
    while not os.path.isfile(display):
        time.sleep(1)
    os.system('tail -f ' + display)
