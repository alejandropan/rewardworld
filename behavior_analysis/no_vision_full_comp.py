#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:56:32 2020

@author: alex
"""
a = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-14/001/alf/QLnovision_nostay_ephysfit.npy')
c = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-14/001/alf/QRnovision_nostay_ephysfit.npy')
a = c-a
b  = psy.loc[(psy['mouse_name'] == 'dop_8' ) & (psy['ses'] == '2020-03-14'), 'QRQL']
s = psy.loc[(psy['mouse_name'] == 'dop_8' ) & (psy['ses'] == '2020-03-14'), 'signed_contrasts']
h = psy.loc[(psy['mouse_name'] == 'dop_8' ) & (psy['ses'] == '2020-03-14'), 'opto_probability_left']
sns.scatterplot(b,a, hue =  h)
plt.xlabel('QRQRL full model')
plt.ylabel('QRQRL no vision model')

sns.lineplot(s, a, hue =  h )
sns.lineplot(s, b,  hue =  h)



a = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-14/001/alf/QLnolaser_nostay_ephysfit.npy')
c = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-14/001/alf/QRnolaser_nostay_ephysfit.npy')
a = c-a
b  = psy.loc[(psy['mouse_name'] == 'dop_8' ) & (psy['ses'] == '2020-03-14'), 'QRQL']
s = psy.loc[(psy['mouse_name'] == 'dop_8' ) & (psy['ses'] == '2020-03-14'), 'signed_contrasts']
h = psy.loc[(psy['mouse_name'] == 'dop_8' ) & (psy['ses'] == '2020-03-14'), 'opto_probability_left']
sns.scatterplot(b,a, hue =  h)
plt.xlabel('QRQL full model')
plt.ylabel('QRQL no laser model')


sns.scatterplot(b,a, hue =  s)
plt.xlabel('QRQL full model')
plt.ylabel('QRQL no laser model')


sns.lineplot(s, a, hue =  h )
sns.lineplot(s, b,  hue =  h)





a = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-14/001/alf/QLnovision_nostay_ephysfit.npy')
c = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-14/001/alf/QRnovision_nostay_ephysfit.npy')
a = c-a
b  = psy.loc[(psy['mouse_name'] == 'dop_11' ) & (psy['ses'] == '2020-03-14'), 'QRQL']
s = psy.loc[(psy['mouse_name'] == 'dop_11' ) & (psy['ses'] == '2020-03-14'), 'signed_contrasts']
h = psy.loc[(psy['mouse_name'] == 'dop_11' ) & (psy['ses'] == '2020-03-14'), 'opto_probability_left']
sns.scatterplot(b,a, hue =  h)
plt.xlabel('QRQL full model')
plt.ylabel('QRQL no vision model')

sns.lineplot(s, a, hue =  h )
sns.lineplot(s, b,  hue =  h)





a = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-14/001/alf/QLnolaser_nostay_ephysfit.npy')
c = np.load('/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-14/001/alf/QRnolaser_nostay_ephysfit.npy')
a = c-a
b  = psy.loc[(psy['mouse_name'] == 'dop_11' ) & (psy['ses'] == '2020-03-14'), 'QRQL']
s = psy.loc[(psy['mouse_name'] == 'dop_11' ) & (psy['ses'] == '2020-03-14'), 'signed_contrasts']
h = psy.loc[(psy['mouse_name'] == 'dop_11' ) & (psy['ses'] == '2020-03-14'), 'opto_probability_left']
sns.scatterplot(b,a, hue =  h)
plt.xlabel('QRQL full model')
plt.ylabel('QRQL no laser model')
sns.lineplot(s, a, hue =  h )
sns.lineplot(s, b,  hue =  h)