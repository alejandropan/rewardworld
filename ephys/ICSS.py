import numpy as np
SES = '/Volumes/witten/Alex/ICSS/47_30_min_ICSS'

t = np.load(SES +'/_spikeglx_sync.times.npy')
p = np.load(SES +'/_spikeglx_sync.polarities.npy')
c = np.load(SES +'/_spikeglx_sync.channels.npy')

l_t = t[np.where(c==17)]
l_p = p[np.where(c==17)]
total_consumed_laser = len(l_t[::2] -  l_t[1::2])

# Licks
li_t = t[np.where(c==18)]
li_p = p[np.where(c==18)]

total_licks =  len(li_t[::2] -  li_t[1::2])
print('Lick rate: '+ str(total_licks/t.max())+' licks/s')