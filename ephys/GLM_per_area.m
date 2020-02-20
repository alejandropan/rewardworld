
% Implementation of Park et al. Nat Neuro model for spike trains in
% decision making task for rewardchoiceworld

% Load npy-matlab and neuroGLM

addpath(genpath('/Users/alex/Documents/MATLAB/npy-matlab/npy-matlab'));
addpath(genpath('/Users/alex/Documents/MATLAB/npy-matlab/neuroGLM'));

% Set experimental parameters

unitOfTime = 's';
binSize = 0.01;
uniqueID = 'dop_4';
expParam = 'dop_4_noopto';

% Load npy python datafiles
session_folder = '' % Input session folder
readNPY('')  = % Input clusters from probe00
readNPY('')  = % Input clusters from probe01 (Optional)



expt = buildGLM.initExperiment(unitOfTime, binSize, uniqueID, expParam);


expt = buildGLM.registerTiming(expt, 'gocue_onset', 'Stimulus onset')
expt = buildGLM.registerTiming(expt, 'stim_onset', 'Stimulus onset'); % events that happen 0 or more times per trial (sparse)
expt = buildGLM.registerTiming(expt, 'response_times', 'Response time');
%%%%
expt = buildGLM.registerSpikeTrain(expt, 'sptrain', 'Our Neuron'); % Spike train!!!
expt = buildGLM.registerSpikeTrain(expt, 'sptrain2', 'Neighbor Neuron');
%%%%
expt = buildGLM.registerValue(expt, 'side', 'Stimulus side'); % information on the trial, but not associated with time
expt = buildGLM.registerValue(expt, 'contrast', 'Stimulus contrast'); % information on the trial, but not associated with time
expt = buildGLM.registerValue(expt, 'choice', 'Mouse choice');