from optparse import OptionParser
from optparse import IndentedHelpFormatter

def Options():
    parser = OptionParser(formatter=IndentedHelpFormatter(max_help_position=70,width=200))
    #---------------------------------------------General options---------------------------------------------------------
    parser.add_option('--manual-seed', dest='manual_seed', default=1993, type='int', help='Set Seed Manually')
    parser.add_option('--gpu', action='store_false', dest='gpu', default=True, help='Use CUDA (True/False)')
    parser.add_option('--gpu-devices', dest='gpu_devices', default='0,1,2,3', type='string', help='Select CUDA_VISIBLE_DEVICES (default:0)')
    #----------------------------------------------Model options----------------------------------------------------------
    parser.add_option('--num-classes', dest='n_classes', default=4, type='int', help='Set Number of Classes (default:3)')
    parser.add_option('--num-layers', dest='n_layers', default=100, type='int', help='Set Number of Layers in the 3D Unet (default:4)')
    parser.add_option('--growth-rate', dest='gr', default=1, type='int', help='Set Growth Rate (default:32)')
    parser.add_option('--downsampling', dest='down_samp', default='Convolution', type='choice', choices=['MaxPool', 'Convolution'], help='Select Downsampling Method (default:Convolution)')
    parser.add_option('--upsampling', dest='up_samp', default='UpConvolution', type='choice', choices=['Trilinear', 'UpConvolution'], help='Select Upsampling Method (default:UpConvolution)')
    #---------------------------------------------Training options--------------------------------------------------------
    #parser.add_option('--epochs', dest='epochs', default=5000, type='int', help='Number of Epochs (default:5000)')
    parser.add_option('--learning-rate', dest='lr', default=0.1, type='float', help='Learning Rate (default:0.1)')
    parser.add_option('--momentum', dest='momentum', default=0.1, type='float', help='Learning Rate (default:0.1)')
    parser.add_option('--batch-size', dest='batchsize', default=7, type='int', help='Batch Size (default:8)')
    parser.add_option('--opt-method', dest='optmethod', default='adam', type='choice', choices=['rmsprop', 'sgd', 'adam'], help='Select Optimizer (default:adam)')
    parser.add_option('--load', dest='load', default='0', type='string', help='Load Saved Model') #TODO
    parser.add_option('--weighted', action='store_false', dest='weighted', default=True, help='Use weights in criterion (True/False)')
    parser.add_option('--reduce', action='store_false', dest='reduce', default=True, help='Use reduce dimensions in criterion (True/False)')
    #-----------------------------------------------Data options----------------------------------------------------------
    parser.add_option('--input-filename', dest='input_filename', default='/db3/psxdb3/data.h5', type='string', help='Select Input Filename')
    parser.add_option('--input-filename2', dest='input_filename2', default='/db3/psxdb3/data.h5', type='string', help='Select Input Filename')
    parser.add_option('--annotations-filename', dest='annotations_filename', default='/db3/psxdb3/SuRVoS_All/annotations/annotations.h5', type='string', help='Select Annotations Filename')
    parser.add_option('--input-size', dest='input_size', default='1,2560,2560', type='string', help='Input Size (default: 5,512,512 )')
    parser.add_option('--input-area', dest='input_area', default='1186,0,0,2160,2560,2560', type='string', help='Input Area (default: 1186,0,0,2160,2560,2560 )')
    parser.add_option('--output-area', dest='output_area', default='0,0,0,1,2560,2560', type='string', help='Output Area in relation with the Input (default: 2,0,0,1,512,512 )')
    parser.add_option('--input-channels', dest='input_channels', default=1, type='int', help='Input Channels (default:1)')
    parser.add_option('--input-stride', dest='input_stride', default='1,2560,2560', type='string', help='Input stride (default:1,512,512)')
    parser.add_option('--input-pad', dest='input_pad', default=0, type='int', help='Input Pad (default:0)')
    parser.add_option('--clean-run', action='store_true', dest='clean_go', default=False, help='Redo all the data preprocessing (True/False)')
    parser.add_option('--normalize', action='store_false', dest='normalize', default=True, help='Use normalize inputs (True/False)')
    parser.add_option('--mask-class', dest='mask_class', default=-100, type='int', help='Use normalize inputs (True/False)')
    #--------------------------------------------Validation options-------------------------------------------------------
    parser.add_option('--val-perc', dest='val_precentage', default=0.05, type='float', help='Validation Precentage (default:0.1)')
    parser.add_option('--test-perc', dest='test_precentage', default=0.05, type='float', help='Testing Precentage (default:0.1)')
    parser.add_option('--val-freq', dest='val_freq', default=1, type='int', help='Validation Frequency every n epochs (default:1)')
    parser.add_option('--cp-dest', dest='cp_dest', default='/db3/psxdb3/SemSegCT/Checkpoints/', type='string', help='Checkpoints Save Directory (default:0.1)')
    return parser
