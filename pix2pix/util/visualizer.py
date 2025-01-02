import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import xarray as xr


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def compute_fss(forecast, observation, threshold, neighborhood_size):
    """
    Compute the Fractions Skill Score (FSS) for precipitation data using a threshold and neighborhood averaging.

    Parameters:
    forecast (2D array): Forecasted precipitation values.
    observation (2D array): Observed precipitation values.
    threshold (float): Precipitation threshold to define an event (e.g., 1.0 mm).
    neighborhood_size (int): Size of the neighborhood (moving window) for spatial averaging.
    OG: 2, 2

    Returns:
    float: The Fractions Skill Score (FSS), which ranges from 0 (no skill) to 1 (perfect forecast).
    """
    # Ensure the input arrays have the same shape
    if forecast.shape != observation.shape:
        raise ValueError("Forecast and observation arrays must have the same shape")

    # Binarize the forecast and observation based on the threshold
    forecast_binary = (forecast >= threshold).astype(float)
    observation_binary = (observation >= threshold).astype(float)

    # Compute fractions over neighborhoods (moving window smoothing)
    forecast_fraction = uniform_filter(forecast_binary, size=neighborhood_size, mode='constant')
    observation_fraction = uniform_filter(observation_binary, size=neighborhood_size, mode='constant')
    
    numerator = np.sum((forecast_fraction - observation_fraction) ** 2)
    denominator = np.sum(forecast_fraction ** 2) + np.sum(observation_fraction ** 2)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    fss = 1 - numerator / denominator

    return fss


def compute_metrics(pred, target):
    """
    Compute various metrics to compare forecast with target.
    
    Parameters:
        pred (numpy.ndarray)   -- predicted precipitation (2D)
        target (numpy.ndarray) -- observed precipitation (2D)
    
    Returns:
        dict -- a dictionary of computed metrics (MSE, MAE, PCC, SSIM)
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    pcc = 0.0
    ssim_value = 0.0
    if np.std(pred) * np.std(target) != 0:
        pcc = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
        ssim_value = ssim(pred, target, data_range=target.max() - target.min())
    fss_25_5 = compute_fss(pred, target, 0.25, 5)
    fss_1_5 = compute_fss(pred, target, 1, 5)
    fss_2_5 = compute_fss(pred, target, 2, 5)
    fss_25_15 = compute_fss(pred, target, 0.25, 15)
    fss_1_15 = compute_fss(pred, target, 1, 15)
    fss_2_15 = compute_fss(pred, target, 2, 15)
    
    return {
            'MSE': mse, 'MAE': mae, 'PCC': pcc, 'SSIM': ssim_value,
            'FSS_0.25_5': fss_25_5, 'FSS_1_5': fss_1_5, 'FSS_2_5': fss_2_5,
            'FSS_0.25_15': fss_25_15, 'FSS_1_15': fss_1_15, 'FSS_2_15': fss_2_15
            }


def plot_and_save(image_array, filename):
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.imshow(image_array, cmap='viridis')  # Choose a colormap
    plt.colorbar()  # Add a colorbar
    plt.axis('off')  # Turn off axis
    plt.savefig(filename, bbox_inches='tight')  # Save the figure as a PNG file
    plt.close()
    # plot_and_save(hrrr_forecast_im, './results/hrrr.png')
    # plot_and_save(model_output_im, './results/model.png')
    # plot_and_save(target_im, './results/target.png')


def save_images(phase, dataset, webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)

    hrrr = visuals.get('real_A').cpu().float().numpy().squeeze(0)[0]
    pred = np.squeeze(visuals.get('fake_B').cpu().float().numpy())
    target = np.squeeze(visuals.get('real_B').cpu().float().numpy())
    
    
    # NetCDF saving for special cases
    if phase not in ["train", "val", "test"]:
        # Metadata for the grid
        lat_min = 0.0 # Bottom-left latitude
        lon_min = 0.0 # Bottom-left longitude
        
        if (dataset == "erie"):
            lat_min = 41.0 - 0.03
            lon_min = -83.5 + 0.88
        elif (dataset == "michigan"):
            lat_min = 41.0 + 0.88
            lon_min = -87.0 - 0.03
        elif (dataset == "ontario"):
            lat_min = 42.5 - 0.03
            lon_min = -80.0 + 0.88
        elif (dataset == "superior"):
            lat_min = 46.0 - 0.03
            lon_min = -91.0 + 0.88
        
        lat_step = 0.01  # Step size in degrees for latitude
        lon_step = 0.01  # Step size in degrees for longitude
        
        # Dimensions of the grid
        height, width = 256, 512
        if (dataset == "michigan"):
            height, width = 512, 256
            hrrr = np.transpose(hrrr, (1, 0))
            pred = np.transpose(pred, (1, 0))
            target = np.transpose(target, (1, 0))
            hrrr = np.flip(hrrr, axis=1)
            pred = np.flip(pred, axis=1)
            target = np.flip(target, axis=1)
            hrrr = np.flip(hrrr, axis=0)
            pred = np.flip(pred, axis=0)
            target = np.flip(target, axis=0)
        
        # Create 1D coordinate arrays
        lats = lat_min + np.arange(height) * lat_step
        lons = lon_min + np.arange(width) * lon_step
        
        # Flip data along latitude axis
        hrrr = np.flip(hrrr, axis=0)
        pred = np.flip(pred, axis=0)
        target = np.flip(target, axis=0)
        
        
        
        hrrr = np.transpose(hrrr, (1, 0))  # Swap lat/lon
        pred = np.transpose(pred, (1, 0))
        target = np.transpose(target, (1, 0))
    
        # Create the dataset
        ds = xr.Dataset(
            {
                "HRRR": (["lon", "lat"], hrrr),
                "Prediction": (["lon", "lat"], pred),
                "Target": (["lon", "lat"], target),
            },
            coords={
                "lon": lons,
                "lat": lats,
            },
        )
            
        
        # # Create the dataset
        # ds = xr.Dataset(
        #     {
        #         "HRRR": (["lat", "lon"], hrrr),
        #         "Prediction": (["lat", "lon"], pred),
        #         "Target": (["lat", "lon"], target),
        #     },
        #     coords={
        #         "lat": lats,
        #         "lon": lons,
        #     },
        # )
        
        # Add variable attributes and save
        ds["HRRR"].attrs["description"] = "HRRR input data"
        ds["Prediction"].attrs["description"] = "Model predictions"
        ds["Target"].attrs["description"] = "Target data from MRMS"
        ds.attrs["title"] = f"{name}"
        ds.to_netcdf(os.path.join(os.path.dirname(image_dir), (name[:-6] + ".nc")))
    
    
    model_metrics = compute_metrics(pred, target)
    hrrr_metrics = compute_metrics(hrrr, target)

    return model_metrics, hrrr_metrics



class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        if self.use_wandb:
            self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
