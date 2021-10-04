import sys
sys.path.append('../../astrometpy')
import h5py, numpy as np, astropy, scipy
# requires your own path to asstromet - if pip installed this is easy
# though some functions only present in the github version may be needed
#sys.path.append('../../astrometpy/')
import dev.astromet.astromet as astromet

def get_hr_class(uamags, ucols):

    # WD
    wdsel=np.flatnonzero((ucols<1.8) & (uamags-3*ucols > 9.5))
    # Red/Brown Dwarf
    dwarfsel=np.flatnonzero((ucols>1.8) & (uamags > 13.4)
                  & (uamags+(4/3)*ucols >17.4))
    # Giant
    giantsel=np.flatnonzero((uamags < 3.8) & (ucols>0.9))
    # Young Main Sequence
    ymssel=np.flatnonzero((uamags < 3.8) & (ucols<0.9))
    # Main Sequence
    mssel=np.flatnonzero((uamags > 3.8) & (uamags < 13.4)
               & (uamags-3.2*ucols < 3.8))
    # Sub-Main Sequence
    smssel=np.flatnonzero((uamags+(4/3)*ucols <17.4) & (uamags-3.2*ucols > 3.8)
               & (uamags-3*ucols < 9.5) & (uamags>3.8))

    hr_class = np.zeros(len(uamags), dtype=np.int64)-1
    hr_class[wdsel]=0
    hr_class[dwarfsel]=1
    hr_class[giantsel]=2
    hr_class[ymssel]=3
    hr_class[mssel]=4
    hr_class[smssel]=5

    return hr_class

def JC_Gaia(V, V_I, output='G'):
    # Riello + colour-colour transformations: Table C2
    if output=='G': coeffs = [-0.01597, -0.02809, -0.2483, 0.03656, -0.002939]
    elif output=='BP': coeffs = [-0.0143, 0.3564, -0.1332, 0.01212, 0.]
    elif output=='RP': coeffs = [0.01868, -0.9028, -0.005321, -0.004186, 0.]

    mag = V + np.sum(np.array(coeffs)[None,:] * V_I[:,None]**(np.arange(5)), axis=1)
    return mag

def change_gums_parameters(max_dist=50):

    print(f'mag_dist: {max_dist}')

    gums_file = f'data/gums_sample_{max_dist}pc.h'
    with h5py.File(gums_file, 'r') as f:
        gums = {}
        for key in f.keys():
            gums[key] = f[key][...]

    # Colour-colour transformations
    gums['primary_mag_g'] = JC_Gaia(gums['primary_mean_absolute_v'], gums['primary_v_i'], output='G')
    gums['primary_mag_bp'] = JC_Gaia(gums['primary_mean_absolute_v'], gums['primary_v_i'], output='BP')
    gums['primary_mag_rp'] = JC_Gaia(gums['primary_mean_absolute_v'], gums['primary_v_i'], output='RP')
    gums['secondary_mag_g'] = JC_Gaia(gums['secondary_mean_absolute_v'], gums['secondary_v_i'], output='G')
    gums['secondary_mag_bp'] = JC_Gaia(gums['secondary_mean_absolute_v'], gums['secondary_v_i'], output='BP')
    gums['secondary_mag_rp'] = JC_Gaia(gums['secondary_mean_absolute_v'], gums['secondary_v_i'], output='RP')

    # Transform variables
    gums['parallax'] = 1e3/gums['barycentric_distance']
    gums['period'] = gums.pop('orbit_period')/astromet.T # years
    gums['l'] = 10**(0.4*(gums['primary_mag_g']-gums['secondary_mag_g']))
    gums['q'] = gums['secondary_mass']/gums['primary_mass']
    gums['a'] = gums.pop('semimajor_axis')
    gums['e'] = gums.pop('eccentricity')


    # gums['vtheta'] = np.arccos(-1+2*np.random.rand(gums['system_id'].size))#gums['periastron_argument']
    # gums['vphi'] = 2*np.pi*np.random.rand(gums['system_id'].size)#gums['longitude_ascending_node']
    # gums['vomega'] = 2*np.pi*np.random.rand(gums['system_id'].size)#gums['inclination']
    gums['vtheta'], gums['vphi'], gums['vomega'] = astromet.viewing_angles(gums['longitude_ascending_node'], gums['inclination'], gums['periastron_argument'])
    gums['tperi'] = gums.pop('periastron_date')

    max_proj_sep = gums['a']*gums['parallax']*(1+gums['e'])*np.cos(gums['vtheta'])
    gums['unresolved']=(gums['binary']==True) & (max_proj_sep<180) # unresolved binaries

    flipped = np.flatnonzero(gums['l']>1)
    gums['l'][flipped] = 1/gums['l'][flipped]
    gums['q'][flipped] = 1/gums['q'][flipped]

    tempg=1.*gums['primary_mag_g'][flipped]
    tempbp=1.*gums['primary_mag_bp'][flipped]
    temprp=1.*gums['primary_mag_rp'][flipped]

    gums['primary_mag_g'][flipped]=1.*gums['secondary_mag_g'][flipped]
    gums['primary_mag_bp'][flipped]=1.*gums['secondary_mag_bp'][flipped]
    gums['primary_mag_rp'][flipped]=1.*gums['secondary_mag_rp'][flipped]

    gums['secondary_mag_g'][flipped]=1.*tempg
    gums['secondary_mag_bp'][flipped]=1.*tempbp
    gums['secondary_mag_rp'][flipped]=1.*temprp

    gums['primary_hr_class'] = get_hr_class(gums['primary_mag_g'], gums['primary_mag_bp']-gums['primary_mag_rp'])
    gums['secondary_hr_class'] = get_hr_class(gums['secondary_mag_g'], gums['secondary_mag_bp']-gums['secondary_mag_rp'])


    # gums['phot_g_mean_mag'] = np.where(gums['binary'], -2.5*np.log10(10**(-gums['primary_mag_g']/2.5) + 10**(-gums['secondary_mag_g']/2.5)),
    #                                                    gums['primary_mag_g'])

    # Save
    new_file = f'data/gums_sample_reparameterised_{max_dist}pc.h'
    with h5py.File(new_file, 'w') as hf:
        for key in gums.keys():
            print(key)
            hf.create_dataset(key, data=gums[key], compression = 'lzf', chunks = True)
