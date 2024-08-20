#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Main procedures
#------------------------------------------------------------------------------

def process_images_morphology(image, nsigma, npixels, nlevels, contrast, weightmap, bkg_error, gain=None, background=0.0, labels=None, connectivity=8, mode='exponential', mask='None', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), relabel=True, nproc=1, progress_bar=True):
    
    segment_img = detect_sources(image, nsigma, npixels, connectivity, mask, background, bkg_error, sigma_clip)
    deblend_img = deblend_sources(image, segment_img, nlevels, contrast, npixels, labels, mode, connectivity, relabel, nproc, progress_bar)
    morphologies = morphology(image, deblend_img, weightmap, gain)
    
def process_images_photometry(image, nsigma, npixels, nlevels, contrast, bkg_error, tot_error, background=0.0, presub_background='None', labels=None, connectivity=8, mode='exponential', mask='None', local_bkg_width=0, apermask_method='correct', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), relabel=True, wcs='None', nproc=1, progress_bar=True):
    
    segment_img = detect_sources(image, nsigma, npixels, connectivity, mask, background, bkg_error, sigma_clip)
    deblend_img = deblend_sources(image, segment_img, nlevels, contrast, npixels, labels, mode, connectivity, relabel, nproc, progress_bar)
    detection_cat = photometry(self.sci, deblend_img, self.convolved_data, tot_error, mask, presub_background, wcs, local_bkg_width, apermask_method, kron_params, 'None', progress_bar)
    photometry_cat = photometry(self.sci, deblend_img, self.convolved_data, tot_error, mask, presub_background, wcs, local_bkg_width, apermask_method, kron_params, detection_cat, progress_bar)

def process_images_number_counts(image, nsigma, npixels, nlevels, contrast, weightmap, bkg_error, gain=None, background=0.0, labels=None, connectivity=8, mode='exponential', mask='None', sigma_clip=SigmaClip(sigma=3.0, sigma_lower=3.0, sigma_upper=3.0, maxiters=10, cenfunc='median', stdfunc='std', grow=False), relabel=True, nproc=1, progress_bar=True):
    
    segment_img = detect_sources(image, nsigma, npixels, connectivity, mask, background, bkg_error, sigma_clip)
    deblend_img = deblend_sources(image, segment_img, nlevels, contrast, npixels, labels, mode, connectivity, relabel, nproc, progress_bar)

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

def detect_sources():
    q

def deblend_sources():
    q

def morphology():
    q

def photometry():
    q

def cutout():
    q

def img_panel:():
    q

def significance_panel():
    q

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