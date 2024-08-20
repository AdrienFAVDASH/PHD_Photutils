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
from photutils.background import SExtractorBackground, StdBackgroundRMS, BkgZoomInterpolator
from photutils.segmentation import detect_sources, deblend_sources, detect_threshold, SourceCatalog
import statmorph
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#------------------------------------------------------------------------------
# Main procedures
#------------------------------------------------------------------------------

def process_images_morphology(image, data, nsigma, npixels, nlevels, contrast, weightmap, bkg_error, gain=None, background=0.0, labels=None, connectivity=8, mode='exponential', mask='None', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, relabel=True, nproc=1, progress_bar=True):
    
    segment_img = image.source_detection(data, nsigma, npixels, connectivity, mask, background, bkg_error, sigma_clip, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = image.source_deblending(data, segment_img, nlevels, contrast, npixels, labels, mode, connectivity, relabel, nproc, progress_bar)
    morphologies = image.morphology(image, deblend_img, weightmap, gain)
    
def process_images_photometry(image, data, nsigma, npixels, nlevels, contrast, kron_params, bkg_error, tot_error, background=0.0, presub_background='None', labels=None, connectivity=8, mode='exponential', mask='None', local_bkg_width=0, apermask_method='correct', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, relabel=True, wcs='None', nproc=1, progress_bar=True):
    
    segment_img = image.source_detection(data, nsigma, npixels, connectivity, mask, background, bkg_error, sigma_clip, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = image.source_deblending(data, segment_img, nlevels, contrast, npixels, labels, mode, connectivity, relabel, nproc, progress_bar)
    
    if smooth_data==True:
        convolved_data = image.convolved_data
    elif smooth_data==False:
        convolved_data=None
    else:
        raise ValueError('invalid value given to "smooth_data"')

    detection_cat = image.photometry(data, deblend_img, convolved_data, tot_error, mask, presub_background, wcs, local_bkg_width, apermask_method, kron_params, 'None', progress_bar)
    photometry_cat = image.photometry(data, deblend_img, convolved_data, tot_error, mask, presub_background, wcs, local_bkg_width, apermask_method, kron_params, detection_cat, progress_bar)

def process_images_number_counts(image, data, nsigma, npixels, nlevels, contrast, weightmap, bkg_error, gain=None, background=0.0, labels=None, connectivity=8, mode='exponential', mask='None', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), smooth_data=True, kernel_name='Tophat', smooth_fwhm=3, kernel_size=5, relabel=True, nproc=1, progress_bar=True):
    
    segment_img = image.source_detection(data, nsigma, npixels, connectivity, mask, background, bkg_error, sigma_clip, smooth_data, kernel_name, smooth_fwhm, kernel_size)
    deblend_img = image.source_deblending(data, segment_img, nlevels, contrast, npixels, labels, mode, connectivity, relabel, nproc, progress_bar)

    return len(deblend_img.labels)
    
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

    def source_detection(self, data, nsigma, npixels, connectivity, mask, background, bkg_error, sigma_clip, smooth_data, kernel_name, smooth_fwhm, kernel_size):

        threshold = detect_threshold(data, nsigma, background, bkg_error, mask, sigma_clip)

        if smooth_data==True:
            convolved_data = smooth_data(data, kernel_name, smooth_fwhm, kernel_size)
            segmentation_image = detect_sources(convolved_data, threshold, npixels, connectivity, mask)
        elif smooth_data==False:
            segmentation_image = detect_sources(data, threshold, npixels, connectivity, mask)
        else:
            raise ValueError('invalid value given to "smooth_data"')

        return segmentation_image

    def source_deblending(self, data, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar):

        deblended_image = deblend_sources(data, segment_img, npixels, labels, nlevels, contrast, mode, connectivity, relabel, nproc, progress_bar)

        return deblended_image

    def morphology(self):
        """Computes all the morphological properties available in Statmorph of all sources in the segmentation map.
            Statmorph documentation : https://statmorph.readthedocs.io/en/latest/"""

        morphology_list = statmorph.source_morphology(self.sci, self.segm_deblend, weightmap=self.wht)
        self.morphologies = morphology_list
        return morphology_list

    def photometry(self, data, segment_img, convolved_data, error, mask, background, wcs, localbkg_width, apermask_method, kron_params, detection_cat, progress_bar):
        
        photometry = SourceCatalog(data, segment_img, convolved_data, error, mask, background, wcs, localbkg_width, apermask_method, kron_params, detection_cat, progress_bar)
        self.photometry = photometry
        return photometry

    def smooth_data(self, data, kernel_name, smooth_fwhm, kernel_size):

        if kernel_name == 'Gaussian':
            smooth_sigma = smooth_fwhm * gaussian_fwhm_to_sigma
            smooth_kernel = Gaussian2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        elif kernel_name == 'Tophat':
            smooth_sigma = smooth_fwhm / np.sqrt(2)
            smooth_kernel = Tophat2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        else :
            raise ValueError('Kernel not supported: {}'.format(kernel_name))

        smooth_kernel.normalize()
        convolved_data = convolve(data, smooth_kernel)
        self.convolved = convolved_data
        return convolved_data       

    def cutout(self, x=None, y=None, width=None, height=None, xmin=None, xmax=None, ymin=None, ymax=None, extensions=['sci', 'err', 'wht', 'bkg', 'bkg_rms'], img_name='cutout_image'):
        """Returns an image cutout"""

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
        """Returns an image representation"""
        
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
        """Returns a pixel significance plot"""
        
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