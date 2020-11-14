import os
from opts import Options
from opts2 import Options as Options2
import time
import shutil

if __name__ == '__main__':
    parser = Options()
    (configuration, args) = parser.parse_args()
    display = configuration.root + configuration.name + '/display_recalc2.txt'
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

    if os.path.isfile(configuration.root + ('/optsRe2_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/optsRe2_%s.py' % configuration.name))
    shutil.copy2('opts.py', configuration.root + ('/optsRe2_%s.py' % configuration.name))

    if os.path.isfile(configuration.root + ('/dataloaderRe2_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/dataloaderRe2_%s.py' % configuration.name))
    shutil.copy2('utils/dataloader.py', configuration.root + ('/dataloaderRe2_%s.py' % configuration.name))

    if os.path.isfile(configuration.root + ('/valrecalc2_%s.py' % configuration.name)):
        os.remove(configuration.root + ('/valrecalc2_%s.py' % configuration.name))
    shutil.copy2('valrecalc.py', configuration.root + ('/valrecalc2_%s.py' % configuration.name))

    recalc = 'valrecalc2_%s.py' % configuration.name

    if configuration.preprocess_filename != 'None' and os.path.isfile(configuration.preprocess_filename):
        print('2D_slices was copied')
        shutil.copy2(configuration.preprocess_filename, configuration.cp_dest + '2D_slices_Real.npz')

    #if os.path.exists(configuration.root + '/models'):
    #    shutil.rmtree(configuration.root + '/models')
    #shutil.copytree('models', configuration.root + '/models')
    # --constraint="12g" -p undergrad --qos=short"
    #os.system('. ~/.anaconda')
    #time.sleep(10)
    # os.system('qsub -hold_jid depe -q all.q -P i13 -l gpu=%d -l gpu_arch=Volta -e %s train_shell.sh %s %s %s %s' % (configuration.gpu_devices, display, configuration.root, train, display, configuration.gpu_devices_sp))
    # --dependency=afterany:432265
    os.system('sbatch -c %d --mem %dG --gres=gpu:%d --constraint="pascal" --output=%s --error=%s train_shell.sh %s %s %s' % (configuration.cores, configuration.memory, configuration.gpu_devices, display, display, configuration.root, recalc, display))
    while not os.path.isfile(display):
        time.sleep(1)
    os.system('tail -f ' + display)
