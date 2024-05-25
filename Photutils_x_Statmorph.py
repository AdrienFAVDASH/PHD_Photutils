#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.stats import SigmaClip
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
#from astropy.coordinates import SkyCoord
from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS, BkgZoomInterpolator
from photutils.segmentation import detect_sources, deblend_sources, detect_threshold
import statmorph
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import warnings

#------------------------------------------------------------------------------
# QOL
#------------------------------------------------------------------------------

warnings.simplefilter("ignore")

#------------------------------------------------------------------------------
# Main procedures
#------------------------------------------------------------------------------

def process_images(detection_image, filter_images, nsigma, npixels, nlevels=32, connectivity=8, contrast=0.001, mode='exponential', smooth_data=True, kernel_name='Tophat', smooth_fwhm=2, kernel_size=5, progress_bar=True, show_plots=True, properties=['flag', 'concentration', 'asymmetry', 'smoothness'], output_hdf5_filename='Photutils.hdf5'):
    """
    Creates a hdf5 file with the desired photometric information of all sources in the provided images.

    Parameters :
    - detection_image : detection image to be provided for photometric analysis. The image will also be used to compute the detection catalog for the filter images, see initialize_photometry
    - filter_images : filter images to be provided for photometric analysis.
    - nsigma : see detect_sources
    - npixels : see detect_sources
    - nlevels : see detect_sources
    - connectivity : see detect_sources
    - contrast : see detect_sources
    - mode : see detect_sources
    - smooth_data : see detect_sources
    - kernel_name : see detect_sources and smooth_data
    - smooth_fwhm : see detect_sources and smooth_data
    - kernel_size : see detect_sources and smooth_data
    - progress_bar : see detect_sources
    - show_plots : see detect_sources and smooth_data

    - output_hdf5_filename : name of the output hdf5 file. Default : 'Photutils.hdf5'
        
    Returns :
    - Saves the hdf5 file under the desired name. File structure is as follows : Photo / Detection_Image or Filter_Images / Image name / dataset
    """
    
    #hdf5 file creation 

    with h5py.File(output_hdf5_filename, 'w') as f:

        # Detection image handling
        print('Detection Image processing')
        
        photo = f.create_group('Photo')
        detection_img = photo.create_group('Detection_Image')
        detection_img_name = detection_img.create_group(detection_image.img_name)
        detection_image.detect_sources(nsigma, npixels, nlevels, connectivity, contrast, mode, smooth_data, kernel_name, smooth_fwhm, kernel_size, None, progress_bar, show_plots)    
        detection_image.morphology()
        
        for i in properties:
            column_name = detection_img_name.create_dataset(i, data=[getattr(detection_image.morphologies[j], i) for j in range(len(detection_image.morphologies))])
        
        # Filter images handling
        if filter_images is None:
            pass
        else:
            filter_images_group = photo.create_group('Filter_Images')
        
            if isinstance(filter_images, tuple):
                filter_images = list(filter_images)
            elif not isinstance(filter_images, list):
                filter_images = [filter_images]
                   
            for i, filter_image in enumerate(filter_images, start=1):
                print('Filter Image', i, 'processing')
            
                filter_image_name = filter_images_group.create_group(filter_image.img_name)
            
                filter_image.morphology()
            
                for i in properties:
                    column_name = detection_img_name.create_dataset(i, data = [getattr(detection_image.morphologies[j], i) for j in range(len(detection_image.morphologies))])
            
def make_cutout(image, x=None, y=None, width=None, height=None, xmin=None, xmax=None, ymin=None, ymax=None, extensions = ['sci', 'err', 'wht', 'bkg', 'bkg_rms'], cutout_img_name='cutout_image'):
        return image.cutout(x, y, width, height, xmin, xmax, ymin, ymax, extensions, cutout_img_name)
    
def make_img_panel(image, vmin=None, vmax=None, scaling=False, cmap=cm.magma):
        fig, ax = plt.subplots()
        image.img_panel(ax, image.sci, vmin, vmax, scaling, cmap)
        plt.show()
        
def make_significance_panel(image, threshold = 2.5, background_substracted = True):
        fig, ax = plt.subplots()
        image.significance_panel(ax, threshold, background_substracted)
        plt.show()
    
#------------------------------------------------------------------------------
# Supporting functions
#------------------------------------------------------------------------------

class Image:
        
    def detect_sources(self, nsigma, npixels, nlevels=32, connectivity=8, contrast=0.001, mode='exponential', smooth_data=True, kernel_name='Tophat', smooth_fwhm=2, kernel_size=5, segm_map=None, progress_bar=True, show_plots=True):
        """
        Creates a deblended segmentation image.

        Parameters :
        - nsigma : The threshold value in units of standard deviation for detecting sources. Sources are pixels whose values are above nsigma times the local background standard deviation.
        - npixels : The minimum number of connected pixels above the threshold that an object must have to be deblended.
        - nlevels : The number of multi-thresholding levels to use for deblending. Each source will be re-thresholded at nlevels levels spaced between its minimum and maximum values (non-inclusive). Default : 32
        - connectivity : The type of pixel connectivity used in determining how pixels are grouped into a detected source. Available options are 8 and 4 where 8-connected pixels touch along their edges or corners while 4-connected pixels touch along their edges. Default : 8
        - contrast : The fraction of the total source flux that a local peak must have (at any one of the multi-thresholds) to be deblended as a separate object. Default : 0.0001
        - mode : The mode used in defining the spacing between the multi-thresholding levels during deblending. Available options are 'exponential', 'linear' and 'sinh'. Default : 'exponential'
        - smooth_data : Bolean, whether to convolve the data with a smoothing kernel before source detection.
        - kernel_name : Name of the kernel used to convolve the data. Only Tophat and Gaussian are supported at the moment. Default : Tophat
        - smooth_fwhm : The width of the kernel. Default : 2
        - kernel_size : The size of the kernel. Default : 5
        - progress_bar : Bolean, whether to display a progress bar while the deblending is taking place
        - show_plots : Bolean, whether to display plots of both the segmented and deblended images
        
        Returns :
        - Saves the deblended segmentation image in the self parameter.
        
        Documentation :
        - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html
        - https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html
        """
        
        detection_threshold = detect_threshold(self.sci, nsigma, background=self.bkg, error = self.err)
            
        #Before detection, convolve data
        if smooth_data is True:
            self.sci = self.smooth_data(data=self.sci, kernel_name=kernel_name, smooth_fwhm=smooth_fwhm, kernel_size=kernel_size)
        elif smooth_data is False:
            pass
        if segm_map is None:
            # Detect sources with npixels connected pixels at/above threshold in data smoothed by kernel
            # https://photutils.readthedocs.io/en/stable/segmentation.html
            self.segm_detect = detect_sources(data=self.sci, threshold=detection_threshold, npixels=npixels, connectivity=connectivity)
        
        else:
            self.segm_detect = segm_map
        
        # Deblend: separate connected/overlapping sources
        # https://photutils.readthedocs.io/en/stable/segmentation.html#source-deblending
        self.segm_deblend = deblend_sources(data=self.sci, segment_img=self.segm_detect, npixels=npixels, nlevels=nlevels, contrast=contrast, mode=mode, connectivity=connectivity, progress_bar=progress_bar)
        
        if show_plots is True:
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.imshow(self.segm_detect, origin='lower', cmap=self.segm_detect.cmap, interpolation='nearest')
            ax1.set_title('Segmentation Image')
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.imshow(self.segm_deblend, origin='lower', cmap=self.segm_deblend.cmap, interpolation='nearest')
            ax2.set_title('Deblended Segmentation Image')
            plt.show()

    def smooth_data(self, data, kernel_name='Tophat', smooth_fwhm=2, kernel_size=5):
        """
        Convolves the data using a kernel.

        Parameters :
        - kernel_name : Name of the kernel used to convolve the data. Only Tophat and Gaussian are supported at the moment. Default : Tophat
        - smooth_fwhm : The width of the kernel. Default : 2
        - kernel_size : The size of the kernel. Default : 5

        Returns :
        - Saves the convolved data to the self parameter.
        
        Documentation :
        - https://docs.astropy.org/en/stable/convolution/index.html
        - https://docs.astropy.org/en/stable/api/astropy.convolution.Gaussian2DKernel.html
        - https://docs.astropy.org/en/stable/api/astropy.convolution.Gaussian2DKernel.html
        """
        
        if kernel_name == 'Gaussian':
            smooth_sigma = smooth_fwhm * gaussian_fwhm_to_sigma
            smooth_kernel = Gaussian2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        elif kernel_name == 'Tophat':
            smooth_sigma = smooth_fwhm / np.sqrt(2)
            smooth_kernel = Tophat2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        else :
            raise ValueError('Kernel not supported: {}'.format(kernel_name))
        
        smooth_kernel.normalize()
        data = convolve(data, smooth_kernel)
        print('Smoothing kernel used : ' + kernel_name)
        return data            
            
    def background(self, box_size=50, filter_size=(3, 3), mask=None, coverage_mask=None, fill_value=0.0, exclude_percentile=10.0, filter_threshold=None, edge_method='pad', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), bkg_estimator=SExtractorBackground(sigma_clip=None), bkgrms_estimator=StdBackgroundRMS(sigma_clip=None), interpolator=BkgZoomInterpolator(order=3, mode='reflect', cval=0.0, grid_mode=True, clip=True)):
        """Calculates background and background rms for a given image"""
        
        bkg = Background2D(data=self.sci, box_size=box_size, mask=mask, coverage_mask=coverage_mask, fill_value=fill_value, exclude_percentile=exclude_percentile, filter_size=filter_size, filter_threshold=filter_threshold, edge_method=edge_method, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, bkgrms_estimator=bkgrms_estimator, interpolator=interpolator)
        background = bkg.background
        background_rms = bkg.background_rms
        return background, background_rms
    
    def morphology(self):
        
        morphology_list = statmorph.source_morphology(self.sci, self.segm_deblend, weightmap=self.wht)
        self.morphologies = morphology_list

    
    def cutout(self, x=None, y=None, width=None, height=None, xmin=None, xmax=None, ymin=None, ymax=None, extensions=['sci', 'err', 'wht', 'bkg', 'bkg_rms'], img_name='cutout_image'):
        """Extract cut out"""

        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            width = xmax - xmin
            height = ymax - ymin
        elif x is not None and y is not None and width is not None and height is not None:
            xmin = x - width // 2
            xmax = x + width // 2
            ymin = y - height // 2
            ymax = y + height // 2
        else:
            raise ValueError("Invalid arguments provided")

        if 'err' in extensions: 
            err = np.zeros((width, height))
        if 'sci' in extensions: 
            data = np.zeros((width, height))
        if 'wht' in extensions: 
            wht = np.zeros((width, height))
        if 'bkg' in extensions: 
            bkg = np.zeros((width, height))
        if 'bkg_rms' in extensions: 
            bkg_rms = np.zeros((width, height))

        xmin = int(np.round(xmin, 0))
        ymin = int(np.round(ymin, 0))

        xstart = 0
        ystart = 0
        xend = width
        yend = height

        if xmin < 0:
            xstart = -xmin
            xmin = 0
            print('Cutout xmin below 0')
        if ymin < 0:
            ystart = -ymin
            ymin = 0
            print('Cutout ymin below 0')
        if xmax > self.sci.shape[0]:
            xend -= xmax - self.sci.shape[0]
            xmax = self.sci.shape[0]
            print('Cutout xmax above image boundary')
        if ymax > self.sci.shape[1]:
            yend -= ymax - self.sci.shape[1]
            ymax = self.sci.shape[1]
            print('Cutout ymax above image boundary')

        data[xstart:xend, ystart:yend] = self.sci[xmin:xmax, ymin:ymax]
        if 'err' in extensions: 
            err[xstart:xend, ystart:yend] = self.err[xmin:xmax, ymin:ymax]
        if 'wht' in extensions: 
            wht[xstart:xend, ystart:yend] = self.wht[xmin:xmax, ymin:ymax]
        if 'bkg' in extensions: 
            bkg[xstart:xend, ystart:yend] = self.bkg[xmin:xmax, ymin:ymax]
        if 'bkg_rms' in extensions: 
            bkg_rms[xstart:xend, ystart:yend] = self.bkg_rms[xmin:xmax, ymin:ymax]

        return ImageFromArrays(data, img_name, err=err, wht=wht, bkg=bkg, bkg_rms=bkg_rms)


    def img_panel(self, ax, im, vmin=None, vmax=None, scaling=False, cmap=cm.magma):
        
        if vmin is None:
            vmin = np.min(im)
        if vmax is None:
            vmax = np.max(im)

        if scaling:
            im = scaling(im)

        ax.axis('off')
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')  # choose better scaling
        
        return ax
    
    def significance_panel(self, ax, threshold = 2.5, background_substracted = True):
    
        if background_substracted is True:
            sig = (self.sci)/self.err
            
        if background_substracted is False:
            sig = (self.sci-self.bkg)/self.bkg_rms

        ax.imshow(sig, cmap = cm.Greys, vmin = -threshold*2, vmax = threshold*2, origin = 'lower', interpolation = 'none')
        ax.imshow(np.ma.masked_where(sig <= threshold, sig), cmap = cm.plasma, vmin = threshold, vmax = 100, origin = 'lower', interpolation = 'none')
        ax.set_axis_off()

        return ax

#------------------------------------------------------------------------------
# Image initializing
#------------------------------------------------------------------------------

class ImageFromMultiFITS(Image):

    def __init__(self, img_filename, img_name, idata = {'sci': 1, 'err': 2, 'wht': 4}, mask = None, mask_edge_thickness=10, background_substracted = True, box_size=50, filter_size=(3, 3), bkg_mask=None, bkg_coverage_mask=None, fill_value=0.0, exclude_percentile=10.0, filter_threshold=None, edge_method='pad', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), bkg_estimator=SExtractorBackground(sigma_clip=None), bkgrms_estimator=StdBackgroundRMS(sigma_clip=None), interpolator=BkgZoomInterpolator(order=3, mode='reflect', cval=0.0, grid_mode=True, clip=True)):

        """generate instance of image class from file"""

        self.hdu = fits.open(img_filename)
        self.sci = self.hdu[idata['sci']].data
        self.sci_initial = self.sci.copy()
        self.err = self.hdu[idata['err']].data
        self.wht = self.hdu[idata['wht']].data
        self.img_name = img_name
        
        if self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        
        self.mask = np.isnan(self.err)
        self.mask = ndimage.binary_dilation(self.mask, iterations=mask_edge_thickness)
        
        self.header = self.hdu[0].header
        self.imwcs = wcs.WCS(self.hdu[idata['sci']].header, self.hdu)
                
        if background_substracted is True:
            self.bkg = np.empty(self.sci.shape)
            self.bkg_rms = np.empty(self.sci.shape)
        if background_substracted is False:
            self.bkg, self.bkg_rms = self.measure_background_map(box_size, filter_size, bkg_mask, bkg_coverage_mask, fill_value, exclude_percentile, filter_threshold, edge_method, sigma_clip, bkg_estimator, bkgrms_estimator, interpolator)        

class ImageFromArrays(Image):

    def __init__(self, data, img_name, err = None, wht = None, bkg = None, bkg_rms = None, background_substracted = True, box_size=50, filter_size=(3, 3), mask=None, coverage_mask=None, fill_value=0.0, exclude_percentile=10.0, filter_threshold=None, edge_method='pad', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), bkg_estimator=SExtractorBackground(sigma_clip=None), bkgrms_estimator=StdBackgroundRMS(sigma_clip=None), interpolator=BkgZoomInterpolator(order=3, mode='reflect', cval=0.0, grid_mode=True, clip=True)):

        """generate instance of image class from cutout"""

        self.sci = data
        self.sci_initial = self.sci.copy()
        self.err = err
        self.wht = wht
        self.bkg = bkg
        self.bkg_rms = bkg_rms
        self.img_name = img_name
        
        if self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        
        if self.bkg is None:
            if background_substracted is True:
                self.bkg = np.empty(self.sci.shape)
                self.bkg_rms = np.empty(self.sci.shape)
            if background_substracted is False:
                self.bkg, self.bkg_rms = self.measure_background_map(box_size, filter_size, mask, coverage_mask, fill_value, exclude_percentile, filter_threshold, edge_method, sigma_clip, bkg_estimator, bkgrms_estimator, interpolator)


class ImageFromDifferentSources(Image):
    
    def __init__(self, data_file, img_name, err_file = None, wht_file = None, bkg_file = None, bkg_rms_file = None, background_substracted = True, box_size=50, filter_size=(3, 3), mask=None, coverage_mask=None, fill_value=0.0, exclude_percentile=10.0, filter_threshold=None, edge_method='pad', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), bkg_estimator=SExtractorBackground(sigma_clip=None), bkgrms_estimator=StdBackgroundRMS(sigma_clip=None), interpolator=BkgZoomInterpolator(order=3, mode='reflect', cval=0.0, grid_mode=True, clip=True)):
        
        """generate instance of image class from different files"""
        
        self.sci = fits.open(data_file)[0].data
        self.sci_initial = self.sci.copy()
        self.err = fits.open(err_file)[0].data if err_file else None
        self.wht = fits.open(wht_file)[0].data if wht_file else None
        self.bkg = fits.open(bkg_file)[0].data if bkg_file else None
        self.bkg_rms = fits.open(bkg_rms_file)[0].data if bkg_rms_file else None
        self.img_name = img_name
        
        if self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        
        if self.bkg is None:
            if background_substracted is True:
                self.bkg = np.empty(self.sci.shape)
                self.bkg_rms = np.empty(self.sci.shape)
            if background_substracted is False:
                self.bkg, self.bkg_rms = self.measure_background_map(box_size, filter_size, mask, coverage_mask, fill_value, exclude_percentile, filter_threshold, edge_method, sigma_clip, bkg_estimator, bkgrms_estimator, interpolator)