import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.visualization import make_lupton_rgb, SqrtStretch, LogStretch, hist, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord
from photutils.background import Background2D, MedianBackground
from photutils.utils import calc_total_error
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.psf.matching import resize_psf, create_matching_kernel, CosineBellWindow
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Tophat2DKernel
from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from scipy import ndimage
from inspect import signature
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from photutils.morphology import data_properties

class Image:
    
    def measure_background_map(self, bkg_size=50, filter_size=3):
        # Calculate sigma-clipped background in cells of 50x50 pixels, then median filter over 3x3 cells
        # For best results, the image should span an integer number of cells in both dimensions (e.g., 1000=20x50 pixels)
        # https://photutils.readthedocs.io/en/stable/background.html
        self.background_map = Background2D(self.sci, bkg_size, filter_size=filter_size)
        self.bkg = self.background_map.background
        self.bkg_rms = self.background_map.background_rms
    
    def smooth_data(self, kernel_name = 'Tophat', smooth_fwhm=2, kernel_size=5):
        # convolve data with Gaussian
        # convolved_data used for source detection and to calculate source centroids and morphological properties
        smooth_sigma = smooth_fwhm * gaussian_fwhm_to_sigma
        
        if kernel_name == 'Gaussian':
            self.smooth_kernel = Gaussian2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        elif kernel_name == 'Tophat':
            self.smooth_kernel = Tophat2DKernel(smooth_sigma, x_size=kernel_size, y_size=kernel_size)
        else :
            raise ValueError('Kernel not supported: {}'.format(kernel_name))
        
        print(kernel_name)
        self.smooth_kernel.normalize()
        self.sci = convolve(self.sci_initial, self.smooth_kernel)
    
    def make_cutout(self, x, y, width, extensions = ['sci', 'err', 'wht', 'bkg', 'bkg_rms']):
        
        """extract cut out"""

        if 'err' in extensions: err = np.zeros((width, width))
        if 'sci' in extensions: data = np.zeros((width, width))
        if 'wht' in extensions: wht = np.zeros((width, width))
        if 'bkg' in extensions: bkg = np.zeros((width, width))
        if 'bkg_rms' in extensions: bkg_rms = np.zeros((width, width))

        x = int(np.round(x, 0))
        y = int(np.round(y, 0))

        xmin = x - width // 2
        xmax = x + width // 2
        ymin = y - width // 2
        ymax = y + width // 2

        xstart = 0
        ystart = 0
        xend = width
        yend = width

        if xmin < 0:
            xstart = -xmin
            xmin = 0
        if ymin < 0:
            ystart = -ymin
            ymin = 0
        if xmax > self.sci.shape[0]:
            xend -= xmax - self.sci.shape[0]
            xmax = self.sci.shape[0]
        if ymax > self.sci.shape[1]:
            yend -= ymax - self.sci.shape[1]
            ymax = self.sci.shape[1]

        if (width % 2) != 0:
            xmax += 1
            ymax += 1

        data[xstart:xend,ystart:yend] = self.sci_initial[xmin:xmax,ymin:ymax]
        if 'err' in extensions: err[xstart:xend,ystart:yend] = self.err[xmin:xmax,ymin:ymax]
        if 'wht' in extensions: wht[xstart:xend,ystart:yend] = self.wht[xmin:xmax,ymin:ymax]
        if 'bkg' in extensions: bkg[xstart:xend,ystart:yend] = self.bkg[xmin:xmax,ymin:ymax]
        if 'bkg_rms' in extensions: bkg_rms[xstart:xend,ystart:yend] = self.bkg_rms[xmin:xmax,ymin:ymax]
        
        return ImageFromArrays(data, err = err, wht = wht, bkg = bkg, bkg_rms = bkg_rms)

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
#turn off smoothing in imshow
        return ax
    
    def pixel_hist(self, bins=1000):
        
        data = self.sci.flatten()
        
        plt.figure()
        plt.hist(data, bins)
        plt.xlabel('pixel value')
        plt.ylabel('pixel count')
        plt.title('Pixel value histogram')
        plt.show()

        negative_values = data[data < 0]
        positive_values = -negative_values
        fit_data = np.concatenate((positive_values, negative_values))

        hist1, bins1 = np.histogram(data, bins=np.linspace(min(data), max(data), bins))
        hist2, bins2 = np.histogram(fit_data, bins=np.linspace(min(data), max(data), bins))

        x = (bins1[:-1] + bins1[1:]) / 2
        subtracted_y = hist1 - hist2
        
        plt.figure()
        plt.hist(x, weights=subtracted_y, bins=bins)
        plt.xlabel('pixel value')
        plt.ylabel('pixel count')
        plt.title('Noise-subtracted pixel value histogram')
        plt.show()
    
    def detect_sources(self, nsigma, npixels, kernel_name='Tophat', smooth_fwhm=2, kernel_size=5, deblend_levels=32, deblend_contrast=0.0001):
        
        # Set detection threshold map as nsigma times RMS above background pedestal
        #detection_threshold = (nsigma * self.bkg_rms) + self.bkg
        #detection_threshold = (nsigma * self.background_map.background_rms)/self.wht + self.background_map.background
        #detection_threshold = nsigma * np.sqrt(np.mean(self.bkg ** 2)) + self.bkg
        if self.background_substracted is True:
            detection_threshold = nsigma * self.err
        if self.background_substracted is False:
            detection_threshold = (nsigma * self.err) + self.bkg
            
        # Before detection, convolve data
        self.smooth_data(kernel_name, smooth_fwhm, kernel_size)

        # Detect sources with npixels connected pixels at/above threshold in data smoothed by kernel
        # https://photutils.readthedocs.io/en/stable/segmentation.html
        self.segm_detect = detect_sources(self.sci, detection_threshold, npixels=npixels)

        # Deblend: separate connected/overlapping sources
        # https://photutils.readthedocs.io/en/stable/segmentation.html#source-deblending
        self.segm_deblend = deblend_sources(self.sci, self.segm_detect, npixels=npixels, nlevels=deblend_levels, contrast=deblend_contrast)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.imshow(self.segm_detect, origin='lower', cmap=self.segm_detect.cmap, interpolation='nearest')
        ax1.set_title('Segmentation Image')
        ax2.imshow(self.segm_deblend, origin='lower', cmap=self.segm_deblend.cmap, interpolation='nearest')
        ax2.set_title('Deblended Segmentation Image')
        plt.tight_layout()
    
    def initialize_photometry(self, kron_params = [2.5, 1]):
        
        self.detection_cat = SourceCatalog(self.sci, self.segm_deblend, convolved_data = self.sci, error = self.err, background = self.bkg, apermask_method = 'correct', kron_params = kron_params, progress_bar=True)
        
        self.photometry_cat = SourceCatalog(self.sci, self.segm_deblend, convolved_data = self.sci, error = self.err, background = self.bkg, apermask_method = 'correct', kron_params = kron_params, detection_cat = self.detection_cat, progress_bar=True)
        return self.photometry_cat
    
    def perform_circular_aperture_photometry(self, radius = 3.0, name = 'aperture_phot'):
        self.photometry_cat.circular_photometry(radius, name)
        
    def perform_kron_photometry(self, kron_params = [2.5, 1], name = 'kron_phot'):
        self.photometry_cat.kron_photometry(kron_params, name)
        
    def to_table(self, columns = ['label', 'xcentroid','ycentroid','bbox_xmin','bbox_xmax','bbox_ymin','bbox_ymax','area',
                 'semimajor_sigma','semiminor_sigma', 'orientation', 'eccentricity', 'segment_flux', 'segment_fluxerr', 
                 'aperture_phot_flux', 'aperture_phot_fluxerr', 'kron_phot_flux', 'kron_phot_fluxerr']):
        self.photometry_table = self.photometry_cat.to_table(columns)
        #self.photometry_table['sky_centroid'] = self.photometry_table['sky_centroid'].astype(str)
        self.photometry_table.write('Photometry_table.h5', path = 'data', format='hdf5', overwrite=True)
        return self.photometry_table
    
class ImageFromMultiFITS(Image):

    def __init__(self, filename, idata = {'sci': 1, 'err': 2, 'wht': 4}, mask = None, mask_edge_thickness=10, background_substracted = True):

        """generate instance of image class from file"""

        self.filename = filename
        self.hdu = fits.open(filename)
        self.sci = self.hdu[idata['sci']].data
        self.sci_initial = self.sci.copy()
        self.err = self.hdu[idata['err']].data
        self.wht = self.hdu[idata['wht']].data
        
        if self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        
        self.mask = np.isnan(self.err)
        self.mask = ndimage.binary_dilation(self.mask, iterations=mask_edge_thickness)
        
        self.header = self.hdu[0].header
        self.imwcs = wcs.WCS(self.hdu[idata['sci']].header, self.hdu)
        
        self.background_substracted  = background_substracted
        
        if self.background_substracted is True:
            self.bkg = np.empty(self.sci.shape)
            self.bkg_rms = np.empty(self.sci.shape)
        if self.background_substracted is False:
            self.measure_background_map()        

class ImageFromArrays(Image):

    def __init__(self, data, err = None, wht = None, bkg = None, bkg_rms = None, background_substracted = True):

        """generate instance of image class from cutout"""

        self.sci = data
        self.sci_initial = self.sci.copy()
        self.err = err
        self.wht = wht
        self.bkg = bkg
        self.bkg_rms = bkg_rms
        self.background_substracted  = background_substracted
        
        if self.bkg is None:
            if self.background_substracted is True:
                self.bkg = np.empty(self.sci.shape)
                self.bkg_rms = np.empty(self.sci.shape)
            if self.background_substracted is False:
                self.measure_background_map()


class ImageFromDifferentSources(Image):
    
    def __init__(self, data_file, err_file = None, wht_file = None, bkg_file = None, bkg_rms_file = None, background_substracted = True):
        
        """generate instance of image class from different files"""
        
        self.sci = fits.open(data_file)[0].data
        self.sci_initial = self.sci.copy()
        self.err = fits.open(err_file)[0].data if err_file else None
        self.wht = fits.open(wht_file)[0].data if wht_file else None
        self.bkg = fits.open(bkg_file)[0].data if bkg_file else None
        self.bkg_rms = fits.open(bkg_rms_file)[0].data if bkg_rms_file else None
        self.background_substracted = background_substracted
        
        if self.err is None:
            self.err = 1 / np.sqrt(self.wht)
        
        if self.bkg is None:
            if self.background_substracted is True:
                self.bkg = np.empty(self.sci.shape)
                self.bkg_rms = np.empty(self.sci.shape)
            if self.background_substracted is False:
                self.measure_background_map()