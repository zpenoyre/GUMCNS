import sys, os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import tqdm, h5py, astropy, scipy, numpy as np
import astropy.units as u
from multiprocessing import Pool
from functools import partial

np.seterr(all='ignore')

# requires your own path to asstromet - if pip installed this is easy
# though some functions only present in the github version may be needed
#sys.path.append('../../astrometpy/')
import dev.astromet.astromet as astromet

# Run fits
def fit_object(isource, return_dict=False, component='unresolved'):

    _params=astromet.params()

    for key in ['ra','dec','pmdec','parallax']:
        setattr(_params, key, gums[key][isource])
    _params.pmrac = gums['pmra'][isource]
    if gums['binary'][isource]:
        # Binaries
        for key in ['period','l','q','a','e',
                  'vtheta','vphi','vomega','tperi']:
            setattr(_params, key, gums[key][isource])
        if component=='primary':
            _params.l = 0.
        if component=='secondary':
            _params.l = 0.
            _params.q = 1/_params.q
            _params.vomega=_params.vomega+np.pi
    else:
        # Single sources - no binary motion
        setattr(_params, 'a', 0)

    if _params.e==0:
        _params.e+=1e-10

    c = Source(float(_params.ra),float(_params.dec),unit='deg',frame='icrs')
    sl=dr3_sl(c, return_times=True, return_angles=True)
    # sl = {'times': [np.array([2016,2016.5,2017]), np.array([2016,2016.5,2017])],
    #       'angles': [np.array([10.,50.,70.]), np.array([10.,50.,70.])]}

    ts=2010+np.hstack(sl['times']).flatten()/365.25
    sort=np.argsort(ts)
    ts=ts[sort].astype(float)
    phis=np.hstack(sl['angles']).flatten()[sort].astype(float)

    # i have a feeling the earth_barycenter call was to save resources....
    trueRacs,trueDecs=astromet.track(ts,_params)#,earth_barycenter=pos_earth_interp)

    # previously used gums['phot_g_mean_mag'] but this doesn't appear in data?
    al_err = astromet.sigma_ast(gums['primary_mag_g'][isource])
    t_obs,x_obs,phi_obs,rac_obs,dec_obs=astromet.mock_obs(ts,phis,trueRacs,trueDecs,err=al_err)

    # EDR3
    gaiaedr3_output=astromet.gaia_fit(t_obs,x_obs,phi_obs,al_err,_params.ra,_params.dec)#, earth_barycenter=pos_earth_interp)

    # DR2
    subset = t_obs<2016.391467761807
    gaiadr2_output=astromet.gaia_fit(t_obs[subset],x_obs[subset],phi_obs[subset],al_err,_params.ra,_params.dec)#, earth_barycenter=pos_earth_interp)
    gaia_keys = ['system_id', 'astrometric_matched_transits_dr2', 'astrometric_matched_transits_edr3', 'visibility_periods_used_dr2', 'visibility_periods_used_edr3', 'astrometric_n_obs_al_dr2', 'astrometric_n_obs_al_edr3', 'astrometric_params_solved_dr2', 'astrometric_params_solved_edr3', 'ra_dr2', 'ra_edr3', 'ra_error_dr2', 'ra_error_edr3', 'dec_dr2', 'dec_edr3', 'dec_error_dr2', 'dec_error_edr3', 'ra_dec_corr_dr2', 'ra_dec_corr_edr3', 'parallax_dr2', 'parallax_edr3', 'parallax_error_dr2', 'parallax_error_edr3', 'ra_parallax_corr_dr2', 'ra_parallax_corr_edr3', 'dec_parallax_corr_dr2', 'dec_parallax_corr_edr3', 'pmra_dr2', 'pmra_edr3', 'pmra_error_dr2', 'pmra_error_edr3', 'ra_pmra_corr_dr2', 'ra_pmra_corr_edr3', 'dec_pmra_corr_dr2', 'dec_pmra_corr_edr3', 'parallax_pmra_corr_dr2', 'parallax_pmra_corr_edr3', 'pmdec_dr2', 'pmdec_edr3', 'pmdec_error_dr2', 'pmdec_error_edr3', 'ra_pmdec_corr_dr2', 'ra_pmdec_corr_edr3', 'dec_pmdec_corr_dr2', 'dec_pmdec_corr_edr3', 'parallax_pmdec_corr_dr2', 'parallax_pmdec_corr_edr3', 'pmra_pmdec_corr_dr2', 'pmra_pmdec_corr_edr3', 'astrometric_excess_noise_dr2', 'astrometric_excess_noise_edr3', 'astrometric_chi2_al_dr2', 'astrometric_chi2_al_edr3', 'astrometric_n_good_obs_al_dr2', 'astrometric_n_good_obs_al_edr3', 'UWE_dr2', 'UWE_edr3']

    gaia_output = {}
    gaia_output['system_id'] = gums['system_id'][isource]
    for key in gaiadr2_output.keys():
        gaia_output[key+'_dr2'] = gaiadr2_output[key]
        gaia_output[key+'_edr3'] = gaiaedr3_output[key]

    if component=='primary': gaia_output['system_id']+=b'A'
    elif component=='secondary': gaia_output['system_id']+=b'B'

    rvs=astromet.radial_velocity(ts,_params,source='p')
    seps=astromet.seperation(ts,_params,phis=phis)

    tsubset = ts<2016.391467761807
    gaia_output['max_rv_dr2']=np.max(rvs[tsubset])
    gaia_output['max_rv_edr3']=np.max(rvs)
    gaia_output['min_rv_dr2']=np.min(rvs[tsubset])
    gaia_output['min_rv_edr3']=np.min(rvs)
    gaia_output['max_seperation_dr2']=np.max(seps[tsubset])
    gaia_output['max_seperation_edr3']=np.max(seps)
    gaia_output['min_seperation_dr2']=np.min(seps[tsubset])
    gaia_output['min_seperation_edr3']=np.min(seps)

    # global results
    # for key in gaia_output.keys():
    #     try: results[key].append(gaia_output[key])
    #     except KeyError: results[key] = [gaia_output[key]]

    if return_dict: return gaia_output

    output = []
    for key in gaia_keys:
        output.append(gaia_output[key])

    return output

gums = {}
# Load scanning law
import scanninglaw.times
from scanninglaw.source import Source
from scanninglaw.config import config
#config['data_dir'] = '/data/asfe2/Projects/testscanninglaw'
# used tto be scanninglaw.times.Times but i think I have an old version
dr3_sl=scanninglaw.times.dr2_sl(version='dr3_nominal')

def fit_gums(max_dist=50):
    print(f'mag_dist: {max_dist}')

    # Load data
    gums_file = f'data/gums_sample_reparameterised_{max_dist}pc.h'
    with h5py.File(gums_file, 'r') as f:
        for key in f.keys():
            gums[key] = f[key][...]


    # Get Earth barycenter
    times = np.linspace(2014.6, 2017.5,100000)
    print('Getting earth barycenter coordinates...')
    # ZP - I removed optional argument ', ephemeris="de430"' - hope it wasn't important!
    pos_earth = astropy.coordinates.get_body_barycentric('earth', astropy.time.Time(times, format='jyear'))
    pos_earth =  np.array([pos_earth.x.to(u.AU) / u.AU,
                         pos_earth.y.to(u.AU) / u.AU,
                         pos_earth.z.to(u.AU) / u.AU]).T
    pos_earth_interp = scipy.interpolate.interp1d(times, pos_earth.T, bounds_error=False, fill_value=0.)

    results = {}
    gaia_keys = fit_object(0, return_dict=True).keys()
    print(gaia_keys)
    for key in gaia_keys: results[key] = []

    # ZP - I'm dumb and don't know how to multithread, but wouldn't run for me
    ncores=40

    # Resolved binaries - Primary component
    isources = np.argwhere(~gums['unresolved']&gums['binary'])[:,0]
    print('resolved primary')
    for i in tqdm.tqdm(range(isources.size)):
        gaia_output=fit_object(isources[i],component='primary',return_dict=True)
        #print(gaia_output)
        for j, key in enumerate(gaia_keys):
            results[key].append(gaia_output[key])
    '''with Pool(ncores) as pool:
        pool_output = list(tqdm.tqdm(pool.imap(partial(fit_object, component='primary'), isources), total=len(isources)))
    for gaia_output in pool_output:
        for i, key in enumerate(gaia_keys):
            results[key].append(gaia_output[i])'''

    # Resolved binaries - Secondary component
    isources = np.argwhere(~gums['unresolved']&gums['binary'])[:,0]
    print('resolved secondary')
    for i in tqdm.tqdm(range(isources.size)):
        gaia_output=fit_object(isources[i],component='secondary',return_dict=True)
        for j, key in enumerate(gaia_keys):
            results[key].append(gaia_output[key])
    '''with Pool(ncores) as pool:
        pool_output = list(tqdm.tqdm(pool.imap(partial(fit_object, component='secondary'), isources), total=len(isources)))
    for gaia_output in pool_output:
        for i, key in enumerate(gaia_keys):
            results[key].append(gaia_output[i])'''

    # Unresolved binaries
    isources = np.argwhere(gums['unresolved']|~gums['binary'])[:,0]
    print('everything else')
    for i in tqdm.tqdm(range(isources.size)):
        gaia_output=fit_object(isources[i],component='secondary',return_dict=True)
        for j, key in enumerate(gaia_keys):
            results[key].append(gaia_output[key])
    '''with Pool(ncores) as pool:
        pool_output = list(tqdm.tqdm(pool.imap(fit_object, isources), total=len(isources)))
    for gaia_output in pool_output:
        for i, key in enumerate(gaia_keys):
            results[key].append(gaia_output[i])'''

    # Save results
    save_file = f'data/gums_fits_dr2&3_{max_dist}pc.h'
    with h5py.File(save_file, 'w') as f:
        for key in results.keys():
            f.create_dataset(key, data=results[key])
