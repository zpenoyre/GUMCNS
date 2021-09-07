from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt
import tqdm, h5py, time


# Download from Gaia archive
max_dist=50
Gaia.ROW_LIMIT=-1

tstart = time.time()
job = Gaia.launch_job_async(f"select * from gaiaedr3.gaia_universe_model where barycentric_distance < {max_dist}")
gums = job.get_results()
gums.sort('source_extended_id')
gums['system_id'] = np.array(gums['source_extended_id'], dtype='S17')
gums['source_extended_id'] = gums['source_extended_id'].astype('S23')
gums.add_index('source_extended_id')
print(f"Query time: {time.time()-tstart:.0f}s")
print('N sources: ', len(gums['source_extended_id']))

# Get multiplicity of sources
print(gums['source_extended_id'].dtype)
systems = np.array([sei[:17] for sei in gums['source_extended_id'] if '+' not in sei])
multiplicity = {sys_id:multiplicity for sys_id,multiplicity  in zip(*np.unique(systems, return_counts=True))}

# Get binaries
binaries = {'system_id':[]}
# 'mag_g','mag_bp','mag_rp','vsini','spectral_type'
star_keys = ['mass','primary_mean_absolute_v', 'v_i','mag_rvs','teff','logg','radius','spectral_type']
system_keys = ['ra','dec','barycentric_distance','pmra','pmdec','radial_velocity','population','age','feh','alphafe']
orbit_keys = ['semimajor_axis','eccentricity','inclination','longitude_ascending_node','orbit_period','periastron_date','periastron_argument']

for key in star_keys:
  binaries['primary_'+key] = []
  binaries['secondary_'+key] = []
for key in system_keys+orbit_keys:
  binaries[key] = []

for i in tqdm.tqdm(range(len(gums))):
    if multiplicity[gums['system_id'][i]] == 2 and gums['source_extended_id'][i]==gums['system_id'][i]+'+     ':

        node = gums[i]
        primary = gums[i+1]
        secondary = gums[i+2]

        if not (primary['source_extended_id'] == gums['system_id'][i]+'A     ') | (primary['source_extended_id'] == gums['system_id'][i]+'AV    '):
            print('primary ', primary['source_extended_id'], gums['system_id'][i])
            print('Distance (no primary): ', gums['barycentric_distance'][i])
        if not (secondary['source_extended_id'] == gums['system_id'][i]+'B     ') | (secondary['source_extended_id'] == gums['system_id'][i]+'BV    '):
            print('secondary ', secondary['source_extended_id'], gums['system_id'][i])
            print('Distance (no secondary): ', gums['barycentric_distance'][i])

        # assert (primary['source_extended_id'] == gums['system_id'][i]+'A     ') | (primary['source_extended_id'] == gums['system_id'][i]+'AV    ')
        # assert (secondary['source_extended_id'] == gums['system_id'][i]+'B     ') | (secondary['source_extended_id'] == gums['system_id'][i]+'BV    ')

        binaries['system_id'].append(node['system_id'])

        for key in star_keys:
          binaries['primary_'+key].append(primary[key])
          binaries['secondary_'+key].append(secondary[key])

        for key in system_keys:
          binaries[key].append(node[key])

        for key in orbit_keys:
          binaries[key].append(secondary[key])


# Get single sources
sys_id, multiplicity = np.unique(systems, return_counts=True)
single_source = np.intersect1d(np.array(gums['source_extended_id'], dtype='S17'),
                               sys_id[multiplicity==1], return_indices=True)[1]

singles = {'system_id':np.array(gums['source_extended_id'], dtype='S17')[single_source]}

for key in star_keys:
  singles['primary_'+key] = gums[key][single_source]
  singles['secondary_'+key] = np.zeros(len(single_source))+np.nan

for key in system_keys:
  singles[key] = gums[key][single_source]

for key in orbit_keys:
  singles[key] = np.zeros(len(single_source))+np.nan

# Convert types
dtypes = {'system_id':'S17',
          'primary_variability_type':'S32',
          'secondary_variability_type':'S32',
          'primary_spectral_type':'S32',
          'secondary_spectral_type':'S32'}
data = {}
for key in singles.keys():
    data[key] = np.hstack((singles[key], binaries[key]))
for key in dtypes:
    data[key] = data[key].astype(dtypes[key])
data['binary'] = np.hstack((np.zeros(len(singles['system_id']), dtype=bool),
                            np.ones(len(binaries['system_id']), dtype=bool)))

# Save
with h5py.File(f'/data/asfe2/Projects/binaries/GCNS_mock/gums_sample_{max_dist}pc.h', 'w') as hf:
    for key in data.keys():
        #print(key)
        hf.create_dataset(key, data=data[key])
