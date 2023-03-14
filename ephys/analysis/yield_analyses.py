from ephys_alf_summary import *

# 1. Load all data
sessions = ephys_ephys_dataset(len(LASER_ONLY))
for i, ses in enumerate(LASER_ONLY):
        print(ses)
        ses_data = alf(ses, ephys=True)
        ses_data.mouse = Path(ses).parent.parent.name
        ses_data.date = Path(ses).parent.name
        ses_data.ses = Path(ses).name
        sessions[i] = ses_data

# Load at unique regions
loc = [] 
for i in np.arange(len(LASER_ONLY)):
    ses = sessions[i]
    for j in np.arange(4):
        try:
            loc.append(np.unique(ses.probe[j].cluster_locations.astype(str)))
        except:
            continue
unique_regions = np.unique(np.concatenate(loc))
unique_regions = unique_regions[np.where(unique_regions!='nan')]

# Look for any unreferenced regions
groups = pd.read_csv('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
groups = groups.iloc[:,1:3]
groups = groups.set_index('original')
group_dict = groups.to_dict()['group']
current_regions = groups.index.unique()
[group_dict[r] for r in current_regions] # This will error if dictionary is not complete

# Stats by regions
yields = pd.DataFrame()
for i in np.arange(len(LASER_ONLY)):
    ses = sessions[i]
    for j in np.arange(4):
        try:
            prob = pd.DataFrame()
            good_units = ses.probe[j].cluster_selection
            prob[['regions','count']] = pd.Series(ses.probe[j].cluster_locations[good_units]).map(group_dict).value_counts().reset_index()
            prob['mouse'] = ses.mouse
            prob['date'] = ses.date
            prob['ses'] = ses.ses
            prob['probe'] = j 
            prob['id'] =  ses.mouse + '_' + ses.date + '_' + ses.ses + '_' + str(j) 
            prob['model_accuracy'] = ses.faccuracy
            prob['bias'] = abs(np.mean(ses.choice))
            yields = pd.concat([yields,prob])
        except:
            continue

yield_by_region(yields)

# Make dictionary dividing M1 and M2
motor_regions = np.array(list(group_dict.keys()))[np.where(np.array(list(group_dict.values()))=='MO')]
motor_p_s= np.array(['M1', 'M1', 'M1', 'M1', 'M1', 'M2', 'M2', 'M2', 'M2', 'M2'])
for i,r in enumerate(motor_regions):
  group_dict[r] = motor_p_s[i]

yields = pd.DataFrame()
for i in np.arange(len(LASER_ONLY)):
    ses = sessions[i]
    for j in np.arange(4):
        try:
            prob = pd.DataFrame()
            good_units = ses.probe[j].cluster_selection
            prob[['regions','count']] = pd.Series(ses.probe[j].cluster_locations[good_units]).map(group_dict).value_counts().reset_index()
            prob['mouse'] = ses.mouse
            prob['date'] = ses.date
            prob['ses'] = ses.ses
            prob['probe'] = j 
            prob['id'] =  ses.mouse + '_' + ses.date + '_' + ses.ses + '_' + str(j) 
            prob['model_accuracy'] = ses.faccuracy
            prob['bias'] = abs(np.mean(ses.choice))
            yields = pd.concat([yields,prob])
        except:
            continue

yield_by_region(yields)

####

yields = pd.DataFrame()
for i in np.arange(len(LASER_ONLY)):
    ses = sessions[i]
    for j in np.arange(4):
        try:
            prob = pd.DataFrame()
            good_units = ses.probe[j].cluster_selection
            prob['regions'] = np.unique(ses.probe[j].cluster_locations[good_units], return_counts=True)[0]
            prob['count'] = np.unique(ses.probe[j].cluster_locations[good_units], return_counts=True)[1]
            prob['mouse'] = ses.mouse
            prob['date'] = ses.date
            prob['ses'] = ses.ses
            prob['probe'] = j 
            prob['id'] =  ses.mouse + '_' + ses.date + '_' + ses.ses + '_' + str(j) 
            prob['model_accuracy'] = ses.faccuracy
            prob['bias'] = abs(np.mean(ses.choice))
            yields = pd.concat([yields,prob])
        except:
            continue

yield_by_region(yields)
