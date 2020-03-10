
%% Implementation of Park et al. Nat Neuro model for spike trains in
% decision making task for rewardchoiceworld currently running on an
% specific area

%y is in time bins either 1 or 0


%% Load npy-matlab and neuroGLM
addpath(genpath('/Users/alex/Documents/MATLAB/npy-matlab/npy-matlab'));
addpath(genpath('/Users/alex/Documents/MATLAB/npy-matlab/neuroGLM'));

%% Set experimental parameters
unitOfTime = 'ms'; % unit of time can be seconds or ms
binSize = 1; % bin size 
uniqueID = 'dop_4'; %brain id
expParam = 'dop_4_noopto'; % unique experiment id

%% Load npy python datafiles
% Neural data
session_folder = '/Volumes/LaCie/dop_4_ephys_data/2020-01-15/001'; % Input session folder
clusters_depths = readNPY(strcat(session_folder,'/alf/probe00/clusters.depths.npy')); % Input clusters from probe00 for a specific area
cluster_metrics = readtable(strcat(session_folder,'/alf/probe00/clusters.metrics.csv'), 'HeaderLines',1);
good_clusters = table2array(cluster_metrics(endsWith(cluster_metrics.Var12,'good'),1)); 
spiketimes = readNPY(strcat(session_folder,'/alf/probe00/spikes.times.npy')).*1000; % Input spike_times from alf folder
spikeclusters = readNPY(strcat(session_folder,'/alf/probe00/spikes.clusters.npy'));  % in ms; % Input spike_times from alf folder
% Behavior data,  cut last trial
choice  = readNPY(strcat(session_folder,'/alf/_ibl_trials.choice.npy'));
choice = choice(1:end-1)
feedback = readNPY(strcat(session_folder,'/alf/_ibl_trials.feedbackType.npy'));
feedback = feedback(1:end-1)
feedback_time  = readNPY(strcat(session_folder,'/alf/_ibl_trials.feedback_times.npy'));
feedback_time = feedback_time(1:end-1) .*1000 % in ms
contrast_L = readNPY(strcat(session_folder,'/alf/_ibl_trials.contrastLeft.npy'));
contrast_R = readNPY(strcat(session_folder,'/alf/_ibl_trials.contrastRight.npy'));
contrast_L(isnan(contrast_L)) = 0;
contrast_R(isnan(contrast_R)) = 0;
signed_contrast = contrast_R - contrast_L 
signed_contrast = signed_contrast(1:end-1)
gocue_onset = readNPY(strcat(session_folder,'/alf/_ibl_trials.goCue_times.npy')) .* 1000 % in ms; 
%Don't cut last gocue onset you need it for trial durantion
response_time = readNPY(strcat(session_folder,'/alf/_ibl_trials.response_times.npy'));
response_time = response_time(1:end-1) .*1000 % in ms
laser_in = readNPY(strcat(session_folder,'/alf/_ibl_trials.laser_epoch_in.npy')); 
laser_in = laser_in(1:end-1) .*1000 % in ms;
laser_out = readNPY(strcat(session_folder,'/alf/_ibl_trials.laser_epoch_out.npy')); 
laser_out = laser_out(1:end-1) .*1000 % in ms;
laser_trial = readNPY(strcat(session_folder,'/alf/_ibl_trials.opto.npy')); 
laser_trial = laser_trial(1:end-1)
% optional
laser_trial = vertcat(nan, laser_trial(1:end-1)) % after opto is more informative sometimes
%% Choose cluster of interest
%e.g
chosen_clusters = good_clusters(1)


%% Initialize Experiment object
expt = buildGLM.initExperiment(unitOfTime, binSize, uniqueID, expParam);

%% Initialize every GLM object 
expt = buildGLM.registerTiming(expt, 'gocue_onset', 'Auditory onset') % events that happen 0 or more times per trial (sparse)
%expt = buildGLM.registerTiming(expt, 'response_times', 'Response time');
%expt = buildGLM.registerTiming(expt, 'laser_in', 'Laser epoch start time');
%expt = buildGLM.registerTiming(expt, 'laser_out', 'Laser epoch end time');
expt = buildGLM.registerTiming(expt, 'feedback_times', 'Feedback time');
for i=1:length(chosen_clusters);
    expt = buildGLM.registerSpikeTrain(expt, strcat('sptrain',int2str(i)), strcat('Neuron',int2str(i))); % Spike train!!!   
end
expt = buildGLM.registerValue(expt, 'contrast', 'Stimulus contrast'); % information on the trial, but not associated with time
%expt = buildGLM.registerValue(expt, 'laser_trial', 'Trial with laser stim'); 
expt = buildGLM.registerValue(expt, 'choice', 'Mouse choice');
expt = buildGLM.registerValue(expt, 'previous_outcome', 'Outcome of previous trial');

%% Calculate some trial information
duration = diff(gocue_onset)
previous_outcome = vertcat(nan, feedback(1:end-1));
previous_laser = vertcat(nan, laser_trial(1:end-1));
%list with neurons
neuron_objs = string(fieldnames(expt.desc))
neuron_objs = neuron_objs(find(startsWith(neuron_objs,'sptrain') == 1))

%% Input trial info
for i= 1:length(choice)
    trial = buildGLM.newTrial(expt, duration(i));
    trial.gocue_onset = gocue_onset(i) - gocue_onset(i); %Normalize times to beggining of trial
    %trial.response_times = response_time(i) - gocue_onset(i); %Normalize times to beggining of trial
    trial.feedback_times = feedback_time(i) - gocue_onset(i); %Normalize times to beggining of trial
    %trial.laser_in = laser_in(i) - gocue_onset(i);
    %trial.laser_out = laser_out(i) - gocue_onset(i);
    trial.previous_outcome = double(previous_outcome(i));
    %trial.previous_laser = previous_laser(i);
    %trial.laser_trial = laser_trial(i);
    for j=1:length(neuron_objs);
        cluster_spikes = spiketimes(spikeclusters == chosen_clusters(j));
        trial_train =  cluster_spikes(cluster_spikes<gocue_onset(i+1) & ...
                       cluster_spikes>gocue_onset(i));
        trial_train = trial_train - gocue_onset(i);
        if isempty(trial_train);
           trial_train = nan ;
        end
        trial.(neuron_objs{j}) = trial_train;
    end
    trial.contrast = signed_contrast(i);
    trial.choice = double(choice(i)); % Int can't multiply sparse matrices
    expt = buildGLM.addTrial(expt, trial, i);
end

%% Making feature space
binfun = expt.binfun;

% boxcar to smooth the temporally delayed effects of each variable
dspec = buildGLM.initDesignSpec(expt);
% Add timing predictors
bs = basisFactory.makeSmoothTemporalBasis('boxcar', 100, 10, expt.binfun);
bs.B = 0.1 * bs.B; % need to check reason for this

offset = 0 % time before event

dspec = buildGLM.addCovariateTiming(dspec, 'gocue_onset', [],...
    [], bs, offset);
%dspec = buildGLM.addCovariateTiming(dspec, 'response_times',[],...
 %   [], bs, offset);
dspec = buildGLM.addCovariateTiming(dspec, 'feedback_times',[],...
    [], bs, offset);
% if adding laser as a regressor. Add a boxcar 1 second long (time of
% stimulation)
%ls = basisFactory.makeSmoothTemporalBasis('boxcar', 1, 1, expt.binfun);
%dspec = buildGLM.addCovariateTiming(dspec, 'laser_in', 'laser_in',...
%    'Laser epoch start', ls);
% Add neural predictors
for i=1:length(neuron_objs);
    dspec = buildGLM.addCovariateSpiketrain(dspec, strcat('hist',int2str(i)),...
    strcat('sptrain',int2str(i)), 'History filter');
end
%% Add whole trial predictors
% a box car that depends on the contrast value value
stimHandle = @(trial, expt) trial.contrast * basisFactory.boxcarStim(binfun(trial.gocue_onset), binfun(trial.feedback_times+ 500), binfun(trial.duration));
dspec = buildGLM.addCovariate(dspec, 'contrast', 'signed contrast', stimHandle, bs);

% add choice
stimHandle1 = @(trial, expt) trial.choice * basisFactory.boxcarStim(binfun(trial.gocue_onset), binfun(trial.duration), binfun(trial.duration));
dspec = buildGLM.addCovariate(dspec, 'choice', 'ccw or cw choice', stimHandle1, bs);

% add previous reward
stimHandle2 = @(trial, expt) trial.previous_outcome * basisFactory.boxcarStim(binfun(trial.gocue_onset), binfun(trial.duration), binfun(trial.duration));
dspec = buildGLM.addCovariate(dspec, 'previous_outcome', 'outcome from last trial', stimHandle2);


%% Choose trials

trialIndices = 1:length(feedback);
opto_trials = trialIndices(logical(laser_trial));
non_opto_trials  = trialIndices(~logical(laser_trial));


%% Build design matrix
dm = buildGLM.compileSparseDesignMatrix(dspec, non_opto_trials);

%% Get the Y (assumes neuron 1 is the neuron of interest
y = buildGLM.getBinnedSpikeTrain(expt, 'sptrain1', dm.trialIndices);

%% Regress
[w, dev, stats] = glmfit(dm.X, y, 'poisson', 'link', 'log');
% or
w1 = fitglm(dm.X, y, 'Distribution', 'poisson', 'link', 'log');

%% Get weight backs
ws = buildGLM.combineWeights(dm, w);

%% Test goodness of fit with last trial

testTrialIndices = length(trialIndices); % test it on the last trial
dmTest = buildGLM.compileSparseDesignMatrix(dspec, testTrialIndices);

ypred = predict(w1,dmTest.X);
ysamp = random(w1,dmTest.X)

scatter(expt.trial(726).sptrain1, b-1)
hold on
plot(ypred)

fit = goodnessOfFit(x,xref,'MSE') 
