import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataloader import Data
from opts import Options
import matplotlib.pyplot as plt
import numpy as np


# plt.switch_backend('WXAgg')

def save_img(filename, x, y, y_pred, options):
    options.CamVid=False
    if options.dice != 'MSE':
        y_pred = F.softmax(y_pred, 1)
        _, y_pred = y_pred.max(1)
    else:
        y = y[:, int(y.shape[1] / 2), :, :].squeeze()
        y_pred = y_pred[:, int(y_pred.shape[1] / 2), :, :].squeeze()
    if options.gpu:
        x = x.cpu()
        y = y.cpu()
        y_pred = y_pred.cpu()
    f, axarr = plt.subplots(3, x.shape[0], figsize=(4, 3), dpi=512)
    if options.dice == 'MSE':
        if options.data_3D is False:
            if x.shape[0] != 1:
                for i in range(0, x.shape[0]):
                    x_out = x[i,
                            options.output_area[0, 0],
                            options.output_area[0, 1]:options.output_area[0, 1] + options.output_area[1, 1],
                            options.output_area[0, 2]:options.output_area[0, 2] + options.output_area[1, 2]].detach().numpy()
                    y_out = y[i,
                            options.output_area[0, 1]:options.output_area[1, 1],
                            options.output_area[0, 2]:options.output_area[1, 2]].detach().numpy()
                    y_pred_out = y_pred[i,
                                 options.output_area[0,1]:options.output_area[1,1],
                                 options.output_area[0,2]:options.output_area[1,2]].detach().numpy()

                    axarr[0, i].imshow(x_out,
                                vmin=x_out.min(),
                                vmax=x_out.max(),
                                cmap='gray')
                    # axarr[0, i].set_title("input")
                    axarr[0, i].axis('off')

                    axarr[1, i].imshow(y_out,
                                vmin=y_out.min(),
                                vmax=y_out.max(),
                                cmap='gray')
                    # axarr[1, i].set_title("annotations")
                    axarr[1, i].axis('off')

                    axarr[2, i].imshow(y_pred_out,
                                vmin=y_out.min(),
                                vmax=y_out.max(),
                                cmap='gray')
                    # axarr[1, i].set_title("annotations")
                    axarr[2, i].axis('off')
            else:
                x_out = x[0,
                          options.output_area[0, 0],
                          options.output_area[0, 1]:options.output_area[0, 1] + options.output_area[1, 1],
                          options.output_area[0, 2]:options.output_area[0, 2] + options.output_area[1, 2]].detach().numpy()
                y_out = y[0,
                          options.output_area[0, 1]:options.output_area[1, 1],
                          options.output_area[0, 2]:options.output_area[1, 2]].detach().numpy()
                y_pred_out = y_pred[0,
                                    options.output_area[0, 1]:options.output_area[1, 1],
                                    options.output_area[0, 2]:options.output_area[1, 2]].detach().numpy()

                axarr[0].imshow(x_out,
                                   vmin=x_out.min(),
                                   vmax=x_out.max(),
                                   cmap='gray')
                # axarr[0, i].set_title("input")
                axarr[0].axis('off')

                axarr[1].imshow(y_out,
                                   vmin=y_out.min(),
                                   vmax=y_out.max(),
                                   cmap='gray')
                # axarr[1, i].set_title("annotations")
                axarr[1].axis('off')

                axarr[2].imshow(y_pred_out,
                                   vmin=y_out.min(),
                                   vmax=y_out.max(),
                                   cmap='gray')
                # axarr[1, i].set_title("annotations")
                axarr[2].axis('off')
        else:
            if x.shape[0] != 1:
                for i in range(0, x.shape[0]):
                    x_out = x[i,
                              0,
                              int(x.shape[2] / 2),
                              options.output_area[0, 1]:options.output_area[0, 1] + options.output_area[1, 1],
                              options.output_area[0, 2]:options.output_area[0, 2] + options.output_area[1, 2]].detach().numpy()
                    y_out = y[i, int(y.shape[1] / 2), :, :].detach().numpy()
                    y_pred_out = y_pred[i, int(y_pred.shape[1] / 2), :, :].detach().numpy()

                    axarr[0, i].imshow(x_out,
                                vmin=x_out.min(),
                                vmax=x_out.max(),
                                cmap='gray')
                    # axarr[0, i].set_title("input")
                    axarr[0, i].axis('off')

                    axarr[1, i].imshow(y_out,
                                vmin=y_out.min(),
                                vmax=y_out.max(),
                                cmap='gray')
                    # axarr[1, i].set_title("annotations")
                    axarr[1, i].axis('off')

                    axarr[2, i].imshow(y_pred_out,
                                vmin=y_out.min(),
                                vmax=y_out.max(),
                                cmap='gray')
                    # axarr[1, i].set_title("annotations")
                    axarr[2, i].axis('off')
            else:
                x_out = x[0,
                          0,
                          int(x.shape[2] / 2),
                          options.output_area[0,1]:options.output_area[0,1]+options.output_area[1,1],
                          options.output_area[0,2]:options.output_area[0,2]+options.output_area[1,2]].detach().numpy()
                y_out = y[0, int(y.shape[1] / 2), :, :].detach().numpy()
                y_pred_out = y_pred[0, int(y_pred.shape[1] / 2), :, :].detach().numpy()
                axarr[0].imshow(x_out,
                                vmin=x_out.min(),
                                vmax=x_out.max(),
                                cmap='gray')
                # axarr[0, i].set_title("input")
                axarr[0].axis('off')

                axarr[1].imshow(y_out,
                                vmin=y_out.min(),
                                vmax=y_out.max(),
                                cmap='gray')
                # axarr[1, i].set_title("annotations")
                axarr[1].axis('off')

                axarr[2].imshow(y_pred_out,
                                vmin=y_out.min(),
                                vmax=y_out.max(),
                                cmap='gray')
                # axarr[1, i].set_title("annotations")
                axarr[2].axis('off')
    else:
        if options.data_3D is False:
            if x.shape[0] != 1:
                for i in range(0, x.shape[0]):
                    x_out = x[i, int(x.shape[1] / 2), options.output_area[0, 1]:options.output_area[0, 1] + options.output_area[1, 1],
                            options.output_area[0, 2]:options.output_area[0, 2] + options.output_area[1, 2]].detach().numpy()
                    axarr[0, i].imshow(x_out,
                                vmin=x_out.min(),
                                vmax=x_out.max(),
                                cmap='gray')
                    # axarr[0, i].set_title("input")
                    axarr[0, i].axis('off')

                    axarr[1, i].imshow(y[i, options.output_area[0,1]:options.output_area[1,1], options.output_area[0,2]:options.output_area[1,2]].detach().numpy(),
                                       vmin=0, vmax=options.n_classes - 1, cmap='gray')
                    # axarr[1, i].set_title("annotations")
                    axarr[1, i].axis('off')

                    axarr[2, i].imshow(y_pred[i, options.output_area[0,1]:options.output_area[1,1], options.output_area[0,2]:options.output_area[1,2]].detach().numpy(),
                                       vmin=0, vmax=options.n_classes - 1, cmap='gray')
                    # axarr[1, i].set_title("annotations")
                    axarr[2, i].axis('off')
            else:
                x_out = x[0, int(x.shape[1] / 2), options.output_area[0, 1]:options.output_area[0, 1] + options.output_area[1, 1],
                        options.output_area[0, 2]:options.output_area[0, 2] + options.output_area[1, 2]].detach().numpy()
                axarr[0].imshow(x_out,
                                vmin=x_out.min(),
                                vmax=x_out.max(),
                                cmap='gray')
                axarr[0].axis('off')

                axarr[1].imshow(y[0, options.output_area[0,1]:options.output_area[1,1], options.output_area[0,2]:options.output_area[1,2]].detach().numpy(),
                                vmin=0, vmax=options.n_classes - 1, cmap='hot')
                # axarr[1, i].set_title("annotations")
                axarr[1].axis('off')

                axarr[2].imshow(y_pred[0, options.output_area[0,1]:options.output_area[1,1], options.output_area[0,2]:options.output_area[1,2]].detach().numpy(),
                                vmin=0, vmax=options.n_classes - 1, cmap='hot')
                # axarr[1, i].set_title("annotations")
                axarr[2].axis('off')
        else:
            if x.shape[0] != 1:
                for i in range(0, x.shape[0]):
                    x_out = x[i, 0, int(x.shape[2] / 2), options.output_area[0, 1]:options.output_area[0, 1] + options.output_area[1, 1],
                            options.output_area[0, 2]:options.output_area[0, 2] + options.output_area[1, 2]].detach().numpy()
                    axarr[0, i].imshow(x_out,
                                vmin=x_out.min(),
                                vmax=x_out.max(),
                                cmap='gray')
                    # axarr[0, i].set_title("input")
                    axarr[0, i].axis('off')

                    axarr[1, i].imshow(y[i, int(y.shape[1] / 2), :, :].detach().numpy(),
                                       vmin=0, vmax=options.n_classes - 1, cmap='hot')
                    # axarr[1, i].set_title("annotations")
                    axarr[1, i].axis('off')

                    axarr[2, i].imshow(y_pred[i, int(y_pred.shape[1] / 2), :, :].detach().numpy(),
                                       vmin=0, vmax=options.n_classes - 1, cmap='hot')
                    # axarr[1, i].set_title("annotations")
                    axarr[2, i].axis('off')
            else:
                x_out = x[0, 0, int(x.shape[2] / 2), options.output_area[0,1]:options.output_area[0,1]+options.output_area[1,1], options.output_area[0,2]:options.output_area[0,2]+options.output_area[1,2]].detach().numpy()
                axarr[0].imshow(x_out,
                                vmin=x_out.min(),
                                vmax=x_out.max(),
                                cmap='gray')
                # axarr[0, i].set_title("input")
                axarr[0].axis('off')

                axarr[1].imshow(y[0, int(y.shape[1] / 2), :, :].detach().numpy(),
                                vmin=0, vmax=options.n_classes - 1, cmap='hot')
                # axarr[1, i].set_title("annotations")
                axarr[1].axis('off')

                axarr[2].imshow(y_pred[0, int(y_pred.shape[1] / 2), :, :].detach().numpy(),
                                vmin=0, vmax=options.n_classes - 1, cmap='hot')
                # axarr[1, i].set_title("annotations")
                axarr[2].axis('off')

    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    # plt.setp(axarr, xticks=[], yticks=[])
    # plt.show(dpi=512, bbox_inches='tight', pad_inches=0)
    # f.canvas.draw()
    # img = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = img.reshape(f.canvas.get_width_height()[::-1] + (3,))
    # img = img.transpose(2,0,1)
    f.savefig(filename, dpi=512, bbox_inches='tight', pad_inches=0)
    return  # img


if __name__ == '__main__':
    parser = Options()
    (options, args) = parser.parse_args()

    torch.manual_seed(options.manual_seed)

    tmp = options.input_size.split(",")
    options.input_size = [int(x.strip()) for x in tmp]
    options.input_size = np.array(options.input_size)

    tmp = options.input_stride.split(",")
    options.input_stride = [int(x.strip()) for x in tmp]
    options.input_stride = np.array(options.input_stride)

    tmp = options.input_area.split(",")
    options.input_area = [int(x.strip()) for x in tmp]
    options.input_area = np.array([options.input_area[0:3], options.input_area[3:6]])

    tmp = options.output_area.split(",")
    options.output_area = [int(x.strip()) for x in tmp]
    options.output_area = np.array([options.output_area[0:3], options.output_area[3:6]])

    data = Data(options, 'train')
    trainDataloader = iter(DataLoader(dataset=data, batch_size=options.batchsize, num_workers=8))
    x, y = next(trainDataloader)

    f, axarr = plt.subplots(2, x.shape[0], figsize=(8, 4))
    for i in range(0, x.shape[0]):
        axarr[0, i].imshow(x[i, 2, :, :].detach().numpy(), vmin=x[i, 2, :, :].detach().numpy().min(), vmax=x[i, 2, :, :].detach().numpy().max(), cmap='gray')
        # axarr[0, i].set_title("input")
        axarr[0, i].axis('off')

        axarr[1, i].imshow(y[i, :, :].detach().numpy(), vmin=y[i, :, :].detach().numpy().min(), vmax=y[i, :, :].detach().numpy().max(), cmap='hot')
        # axarr[1, i].set_title("annotations")
        axarr[1, i].axis('off')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    plt.setp(axarr, xticks=[], yticks=[])
    plt.show(dpi=512, bbox_inches='tight', pad_inches=0)
