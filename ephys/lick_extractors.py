import numpy as np
SES = '/Volumes/witten/Alex/Data/Subjects/dop_46/2022-05-06/002'


t = np.load(SES +'/raw_ephys_data/_spikeglx_sync.times.npy')
p = np.load(SES +'/raw_ephys_data/_spikeglx_sync.polarities.npy')
c = np.load(SES +'/raw_ephys_data/_spikeglx_sync.channels.npy')

bpod_t = t[np.where(c==16)]
bpod_p = p[np.where(c==16)]
assert len(np.unique(bpod_p[::2])) == 1

errors = bpod_t[::2][np.where((bpod_t[::2] -  bpod_t[1::2])<-3.9)]
correct = bpod_t[::2][np.where(((bpod_t[::2] -  bpod_t[1::2])<-0.11) & ((bpod_t[::2] -  bpod_t[1::2])>-3.9))]

len(np.where((bpod_t[::2] -  bpod_t[1::2])<-0.11)[0])
len(np.where((bpod_t[::2] -  bpod_t[1::2])<-0.09999))


l_t = t[np.where(c==17)]
l_p = p[np.where(c==17)]
total_consumed_laser = len(l_t[::2] -  l_t[1::2])/20


ft = np.load(SES +'/alf/_ibl_trials.feedbackType.npy')
block = np.load(SES +'/alf/_ibl_trials.opto_block.npy')
laser_rewards = np.intersect1d(np.where(ft==1), np.where(block==1))

correct_la = np.where(block[np.where(ft==1)]==1)
correct_wa = np.where(block[np.where(ft==1)]==0)

# Licks
li_t = t[np.where(c==18)]
li_p = p[np.where(c==18)]

error_licks_pre = []
error_licks_post = []
correct_licks_pre = []
correct_licks_post = []

assert len(ft) == np.sum(len(errors)+len(correct))


for i in errors: 
    error_licks_pre.append(np.sum((li_t>i-0.5)&(li_t<i-0.01)))
    error_licks_post.append(np.sum((li_t>=i)&(li_t<i+1)))

for i in correct: 
    correct_licks_pre.append(np.sum((li_t>i-0.5)&(li_t<i-0.01)))
    correct_licks_post.append(np.sum((li_t>=i)&(li_t<i+1)))

laser_licks_pre =  np.array(correct_licks_pre)[correct_la]
laser_licks_post = np.array(correct_licks_post)[correct_la]

water_licks_pre =  np.array(correct_licks_pre)[correct_wa]
water_licks_post = np.array(correct_licks_post)[correct_wa]

print('Consumed Laser: '+str(total_consumed_laser/len(laser_rewards)*100)+str('%'))

print('Pre error: '+ str(np.mean(error_licks_pre)))
print('Post error: '+ str(np.mean(error_licks_post)))
print('Pre Laser: '+ str(np.mean(laser_licks_pre)))
print('Post Laser: '+ str(np.mean(laser_licks_post)))
print('Pre Water: '+ str(np.mean(water_licks_pre)))
print('Post Water: '+ str(np.mean(water_licks_post)))


