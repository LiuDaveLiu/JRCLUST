classdef Config < dynamicprops
    %CONFIG JRCLUST session configuration
    % replacement for P struct

    %% OBJECT-LEVEL PROPERTIES
    properties (Hidden, SetAccess=private, SetObservable)
        customParams;           % params not included in the default set
        isError;                % true if an error in configuration
        isV3Import;             % true if old-style params are referred to in configFile
        paramSet;               % common and advanced parameter sets with default values and validation criteria
        tempParams;             % temporary parameters (probably a hack)
    end

<<<<<<< HEAD
    %% DEPENDENT OBJECT-LEVEL PROPERTIES
    properties (Dependent, Hidden, SetObservable)
        deprecatedParams;       % deprecated parameters (for excluding)
        oldParamSet;            % old-style parameters with mapping to new params
    end

    %% CONFIG FILE
    properties (SetAccess=private, SetObservable)
        configFile;             % path to configuration file
=======
    %% OLD-STYLE PARAMS, publicly settable (will be deprecated after a grace period)
    properties (SetObservable, Dependent, Hidden, Transient)
        % cvrDepth_drift;           % doesn't appear to be used (was {})
        % maxSite_detect;           % doesn't appear to be used (was 1.5)
        % maxSite_dip;              % doesn't appear to be used (was [])
        % maxSite_fet;              % doesn't appear to be used (was [])
        % maxSite_merge;            % doesn't appear to be used (was [])
        % maxSite_pix;              % doesn't appear to be used (was 4.5)
        % maxSite_show;             % appears to be synonymous with nSiteDir/maxSite
        % maxSite_sort;             % doesn't appear to be used (was [])
        % rejectSpk_mean_thresh;    % appears to be synonymous with blankThresh/blank_thresh
        autoMergeCriterion;         % => autoMergeBy
        blank_thresh;               % => blankThresh
        csFile_merge;               % => multiRaw
        delta1_cut;                 % => log10DeltaCut
        fEllip;                     % => useElliptic
        fft_thresh;                 % => fftThreshMad
        fGpu;                       % => useGPU
        fImportKilosort;            % => fImportKsort
        fRepeat_clu;                % => repeatLower
        fVerbose;                   % => verbose
        fWav_raw_show;              % => showRaw
        gain_boost;                 % => gainBoost
        header_offset;              % => headerOffset
        MAX_BYTES_LOAD;             % => maxBytesLoad
        MAX_LOAD_SEC;               % => maxSecLoad
        maxCluPerSite;              % => maxClustersSite
        maxDist_site_um             % => evtMergeRad
        maxDist_site_spk_um;        % => evtDetectRad
        maxSite;                    % => nSiteDir
        min_count;                  % => minClusterSize
        mrSiteXY;                   % => siteLoc
        nLoads_gpu;                 % => ramToGPUFactor
        nPad_filt;                  % => nSamplesPad
        nRepeat_merge;              % => nPassesMerge
        nSites_ref;                 % => nSitesExcl
        nSkip_lfp;                  % => lfpDsFactor
        probe_file;                 % => probeFile
        rho_cut;                    % => log10RhoCut
        spkLim;                     % => evtWindowSamp
        spkLim_ms;                  % => evtWindowms
        spkLim_raw;                 % => evtWindowRawSamp
        spkLim_raw_factor;          % => evtWindowRawFactor
        spkLim_raw_ms;              % => evtWindowRawms
        spkRefrac;                  % => refracIntSamp
        spkRefrac_ms;               % => refracIntms
        spkThresh;                  % => evtManualThresh
        spkThresh_uV;               % => evtManualThreshuV
        sRateHz;                    % => sampleRate
        sRateHz_lfp;                % => lfpSampleRate
        thresh_corr_bad_site;       % => siteCorrThresh
        thresh_mad_clu;             % => outlierThresh
        tlim;                       % => dispTimeLimits
        tlim_load;                  % => loadTimeLimits
        uV_per_bit;                 % => bitScaling
        vcCommonRef;                % => carMode
        vcDataType;                 % => dtype
        vcDetrend_postclu;          % => rlDetrendMode
        vcFet;                      % => clusterFeature
        vcFet_show;                 % => dispFeature
        vcFile;                     % => singleRaw
        vcFile_gt;                  % => gtFile
        vcFile_prm;                 % => configFile
        vcFile_thresh;              % => threshFile
        vcFilter;                   % => filterType
        vcFilter_show;              % => dispFilter
        viChan_aux;                 % => auxSites
        viShank_site;               % => shankMap
        vnFilter_user;              % => userFiltKernel
        vrSiteHW;                   % => probePad
        viSite2Chan;                % => siteMap
        viSiteZero;                 % => ignoreSites
    end

    %% OLD-STLYE PARAMS, not publicly settable
    properties (Dependent, SetAccess=private, Hidden)
        miSites;                    % => siteNeighbors
    end
    
    %% NEW-STYLE PARAMS, publicly settable
    properties (SetObservable)
        % computation params
        gpuLoadFactor = 5;          % GPU memory usage factor (4x means 1/4 of GPU memory can be loaded)
        randomSeed = 0;             % random seed
        ramToGPUFactor = 8;         % ratio: RAM / (GPU memory) (increase this number if GPU memory error)
        useGPU = true;              % use GPU in computation if true
        verbose = true;             % be chatty while processing

        % file location params
        outputDir = '';             % directory in which to place output files

        % recording params
        auxSites;                   %
        bitScaling = 0.30518;       % bit scaling factor (uV/bit)
        configFile;                 % parameter file
        dtype = 'int16';            % raw data binary format
        gainBoost = 1;              % multiply the raw recording by this gain to boost uV/bit
        gtFile = '';                % ground truth file (default: SESSION_NAME_gt.mat) (TODO: specify format)
        headerOffset = 0;           % file header offset, in bytes
        lfpSampleRate = 2500;       % sample rate of the LFP recording, in Hz
        nChans = 385;               % number of channels stored in recording
        probeFile;                  % probe file to use (.prb, .mat)
        probePad;                   %
        rawRecordings;              % collection of recordings
        sampleRate = 30000;         % sample rate of the recording, in Hz
        shankMap;                   % index of shank to which a site belongs
        siteLoc;                    % x-y locations of channels on the probe, in microns
        siteMap;                    % channel mapping; row i in the data corresponds to channel `siteMap(i)`

        % preprocessing params
        useElliptic = true;         % use elliptic filter if true (and only if filterType='bandpass')
        fftThreshMAD = 0;           % automatically remove frequency outliers (unit:MAD, 10 recommended, 0 to disable). Verify by running "jrc traces" and press "p" to view the power spectrum.
        filtOrder = 3;              % bandpass filter order
        filterType = 'ndiff';       % filter to use {'ndiff', 'sgdiff', 'bandpass', 'fir1', 'user', 'fftdiff', 'none'}
        freqLim = [300 3000];       % frequency cut-off limit for filterType='bandpass' (ignored otherwise)
        freqLimNotch = [];
        freqLimStop = [];
        loadTimeLimits = [];        % time range of recording to load, in s (use whole range if empty)
        maxBytesLoad = [];          % default memory loading block size (bytes)
        maxSecLoad = [];            % maximum loading duration (seconds) (overrides 'maxBytesLoad')
        ndist_filt = 5;             % undocumented
        nSamplesPad = 100;          % number of samples to overlap between multiple loading (filter edge safe)
        userFiltKernel = [];        % custom filter kernel (optional unless filterType='user')
        carMode = 'mean';           % common average referencing mode (one of 'none', 'mean', 'median', or 'whiten')

        % spike detection params
        blankThresh = [];           % reject spikes exceeding the channel mean after filtering (MAD unit), ignored if [] or 0
        evtDetectRad = 75;          % radius for extracting waveforms, in microns (used if nSiteDir and nSitesExcl are empty)
        evtManualThreshuV = [];     % manual spike detection threshold, in microvolts
        evtMergeRad = 50;           % radius of spike event merging, in microns
        evtWindowms = [-0.25 0.75]; % interval around event to extract filtered spike waveforms, in ms
        evtWindowRawms;             % interval around event to extract raw spike waveforms, in ms
        evtWindowRawFactor = 2;     % ratio of raw samples to filtered samples to extract if evtWindowRawms is not set
        fGroup_shank = false;       % group all sites in the same shank if true
        ignoreSites = [];           % sites to manually ignore in the sorting
        nDiff_filt = 2;             % Differentiation filter for filterType='sgdiff', ignored otherwise. Set to [] to disable. 2n+1 samples used for centered differentiation
        nneigh_min_detect = 0;      % Min. number of neighbors near the spike below threshold. choose between [0,1,2]
        nSiteDir;                   % number of neighboring sites to group in each direction (TODO: deprecate this)
        nSitesExcl;                 % number of sites to exclude from the spike waveform group
        refracIntms = 0.25;         % spike refractory interval, in ms
        siteCorrThresh = 0;         % reject bad sites based on max correlation with neighboring sites, using raw waveforms; ignored if 0
        spkThresh_max_uV = [];      % maximum absolute amp. allowed
        threshFile = '';            % path to .mat file storing spike detection thresholds (created by 'preview' GUI)

        % feature extraction params
        clusterFeature = 'pca';     % feature to use in clustering
        fInterp_fet = true;         % interpolate waveforms for feature projection to find optimal delay (2x interp) if true
        fSpatialMask_clu = false;   % apply spatial mask calculated from the distances between sites to the peak site (half-scale: evtDetectRad)
        min_sites_mask = 5;         % minimum number of sites to have to apply spatial mask
        nFet_use = 2;               % undocumented
        time_feature_factor;        % undocumented

        % clustering params
        autoMergeBy = 'xcorr';      % metric to use when automerging clusters
        dc_percent = 2;             % percentile at which to cut off distance in rho computation
        fDrift_merge = true;        % compute multiple waveforms at three drift locations based on the spike position if true
        log10DeltaCut = 0.6;        % the base-10 log of the delta cutoff value
        log10RhoCut = -2.5;         % the base-10 log of the rho cutoff value
        maxClustersSite = 20;       % maximum number of clusters per site if local detrending is used
        minClusterSize = 30;        % minimum cluster size (set to 2*#features if lower)
        nInterp_merge = 1;          % Interpolation factor for the mean unit waveforms, set to 1 to disable
        nPassesMerge = 10;          % number of passes for unit mean raw waveform-based merging
        outlierThresh = 7.5;        % threshold to remove outlier spikes for each cluster, in MAD
        nTime_clu = 1;              % number of time periods over which to cluster separately (later to be merged after clustering)
        repeatLower = false;        % repeat clustering for the bottom half of the cluster amplitudes if true
        rlDetrendMode = 'global';   % 
        spkLim_factor_merge = 1;    % Waveform range for computing the correlation. spkLim_factor_merge <= spkLim_raw_factor_merge. circa v3.1.8

        % display params
        dispFeature = 'vpp';        % feature to display in time/projection views
        dispFilter = '';            % 
        dispTimeLimits = [0 0.2];   % time range to display (in seconds)
        fText = true;               % 
        nShow = 200;                % maximum number of traces to show [D?# spikes to show]
        nShow_proj = 500;           % maximum number of features to show in projection
        nSitesFigProj = 5;          % number of sites to display in the feature projection view
        nTime_traces = 1;           % number of time segments to display. Set to 1 to show one continuous time segment
        nSpk_show = 30;             % show spike waveforms for manual clustering
        pcPair = [1 2];             % PC projection to show (1 vs 2; 1 vs 3; 2 vs 3), can be toggled
        showRaw = false;            % show raw waveforms in main view if true
        time_tick_show = [];        % 
        tLimFigProj = [];           % time range to display in feature view, in seconds
        um_per_pix = 20;            % 

        % preview GUI params
        nLoads_max_preview = 30;    % number of time segments to load for preview
        sec_per_load_preview = 1;   % recording duration per continuous segment to preview (in sec)

        % to get to, eventually
        LineStyle = '';
        MAX_LOG = 5;
        S_imec3 = [];
        blank_period_ms = 5;
        corrLim = [0.9 1];
        cviShank = [];
        dc_factor = 1;
        dc_frac = [];
        dinput_imec_trial = 1;
        duration_file = [];
        fAddCommonRef = false;
        fAverageTrial_psth = true;
        fCacheRam = true;
        fCheckSites = false;
        fDetectBipolar = false;
        fDiscard_count = true;
        fInverse_file = false;
        fImportKsort = false;
        fLoad_lfp = false;
        fMeanSite = true;
        fMeanSiteRef = false;
        fMeanSite_drift = false;
        fMinNorm_wav = false;
        fNormRhoDelta = true;
        fParfor = true;
        fPcaDetect = false;
        fProcessEven = false;
        fProcessOdd = false;
        fProcessReverseOrder = false;
        fProj_sort = false;
        fRamCache = true;
        fRejectSpk_vpp = false;
        fRms_detect = false;
        fRun = true;
        fSaveEvt = true;
        fSavePlot_RD = true;
        fSaveRawSpk = false;
        fSaveSpk = true;
        fShowAllSites = false;
        fSingleColumn_track = true;
        fSmooth_track = true;
        fSpike_show = true;
        fTranspose_bin = true;
        fUseCache_track = false;
        fUseLfp_track = true;
        fWhiten_traces = false;
        filter_sec_rate = 2;
        filter_shape_rate = 'triangle';
        flim_vid = [];
        freqLimNotch_lfp = [];
        freqLim_corr = [15 150];
        freqLim_excl_track = [58 62];
        freqLim_lfp = [];
        freqLim_track = [15 150];
        iChan_aux = [];
        iChan_vid = [];
        iClu_show = [];
        iGpu = 1;
        load_fraction_track = [];
        maxAmp = 250;
        maxAmp_lfp = 1000;
        maxDist_site_merge_um = 35;
        maxLfpSdZ = 4.5;
        maxSite_track = [2 3 4 5 6 7 8];
        maxWavCor = 0.98;
        max_shift_track = [];
        mrColor_proj = [213 219 235; 0 130 196; 240 119 22]/256;
        nBytes_file = [];
        nClu_show_aux = 10;
        nMinAmp_ms = 0;
        nPcPerChan = 1;
        nPc_dip = 3;
        nSites_excl_ref = 6;
        nSkip_show = 1;
        nSkip_whiten = 10;
        nSmooth_ms_psth = 50;
        nT_drift = [];
        nThreads = 128;
        nw_lcm_track = 1;
        offset_sec_preview = 0;
        pix_per_sec_track = [];
        qqFactor = 5;
        qqSample = 4;
        rateLim_psth = [];
        refrac_factor = 2;
        rms_filt_ms = 0;
        sRateHz_aux = [];
        sRateHz_rate = 1000;
        slopeLim_ms = [0.05 0.35];
        spkLim_ms_fet = [-0.25 0.75];
        tBin_track = 9;
        tRefrac_trial = 0.001;
        tbin_drift = [];
        tbin_psth = 0.01;
        template_file = '';
        thresh_automerge_pca = [];
        thresh_corr_track = [];
        thresh_merge_clu = 0;
        thresh_sd_ref = 5;
        thresh_split_clu = 0;
        thresh_trial = [];
        tlim_clu = [];
        tlim_lfp = [0 5];
        tlim_psth = [-1 5];
        tlim_vid = [];
        vcCluDist = 'eucldist';
        vcCluWavMode = 'mean';
        vcDate_file = '';
        vcDc_clu = 'distr';
        vcFile_aux = '';
        vcFile_bonsai = '';
        vcFile_lfp = '';
        vcFile_trial = '';
        vcFile_vid = '';
        vcFilter_detect = '';
        vcLabel_aux = '';
        vcMode_track = 'mt_cpsd2_mr';
        vcSpatialFilter = 'none';
        vcSpkRef = 'nmean';
        viChan_bin = [];
        viChan_show = [];
        viDepth_excl_track = [];
        viDepth_track = [];
        viSite_bad_track = [];
        vrScale_aux = 1;
        xtick_psth = 0.2;
        ybin_drift = 2;
>>>>>>> parent of eb16aa4... WIP: misc
    end

    %% NOT TO BE SET BY USERS
    properties (SetAccess=private)
        siteNeighbors;          % indices of neighbors for each site
    end

    %% RECORDING(S) (to ease the v3-v4 transition)
    properties (Dependent, Hidden, SetObservable)
        singleRaw;              % formerly vcFile
        multiRaw;               % formerly csFile_merge
    end

    %% COMPUTED PARAMS
    properties (SetObservable, Dependent)
        bytesPerSample;         % byte count for each discrete sample
        evtManualThreshSamp;    % evtManualThresh / bitScaling
        evtWindowRawSamp;       % interval around event to extract raw spike waveforms, in samples
        evtWindowSamp;          % interval around event to extract filtered spike waveforms, in samples
        nSites;                 % numel(siteMap)
        nSitesEvt;              % 2*nSiteDir + 1 - nSitesExcl
        refracIntSamp;          % spike refractory interval, in samples
        sessionName;            % name of prm file, without path or extensions
    end

    %% LIFECYCLE
    methods
        function obj = Config(filename)
            %CONFIG Construct an instance of this class
            if nargin == 0
                userParams = struct();
                obj.configFile = '';
            elseif isstruct(filename)
                userParams = filename;
                obj.configFile = '';
            elseif ischar(filename)
                filename_ = jrclust.utils.absPath(filename);
                if isempty(filename_)
                    error('File not found: ''%s''', filename);
                end

                userParams = jrclust.utils.mToStruct(filename_);
                obj.configFile = filename_;
            end

            obj.customParams = {};
            obj.isV3Import = 0;
            obj.isError = 0;

            % for setting temporary parameters
            obj.tempParams = containers.Map();

            obj.loadParams(userParams);
        end

        function obj = subsasgn(obj, prop, val)
            if strcmp(prop.type, '.')
                [flag, val, errMsg] = obj.validateProp(prop.subs, val);
                if flag
                    obj.setProp(prop.subs, val);
                else
                    error(errMsg);
                end
            end
        end
    end

    %% DOUBLE SECRET METHODS
    methods (Access = private, Hidden)
        function error(obj, emsg, varargin)
            %ERROR Raise an error
            obj.isError = 1;
            if obj.batchMode
                error(emsg);
            else
                errordlg(emsg, varargin{:});
            end
        end

        function loadParams(obj, filename)
            %LOADPARAMS Load parameters from file
            if nargin < 2
                filename = obj.configFile;
            end

            if ischar(filename)
                filename_ = jrclust.utils.absPath(filename);
                userParams = jrclust.utils.mToStruct(filename_); % raises error if not a file
            elseif isstruct(filename)
                userParams = filename;
            else
                error('Class not recognized: %s', class(filename));
            end

            % read in default parameter set
            obj.paramSet = jrclust.utils.getDefaultParams(0);
            fullParams = jrclust.utils.mergeStructs(obj.paramSet.commonParameters, ...
                                                    obj.paramSet.advancedParameters);

            % set default parameters
            paramNames = fieldnames(fullParams);
            for i = 1:numel(paramNames)
                paramName = paramNames{i};
                if strcmp(paramName, 'rawRecordings')
                    obj.setProp('rawRecordings', '');
                    continue;
                end
                [flag, val, errMsg] = obj.validateProp(paramName, fullParams.(paramName).default_value);
                if flag
                    obj.setProp(paramName, val);
                else
                    warning(errMsg);
                end
            end

            % overwrite default parameters with user-specified params
            if isfield(userParams, 'template_file') && ~isempty(userParams.template_file)
                try
                    userTemplate = jrclust.utils.mToStruct(jrclust.utils.absPath(userParams.template_file));
                    fns = fieldnames(userParams);
                    for i = 1:numel(fns) % merge userParams (specific) into userTemplate (general)
                        userTemplate.(fns{i}) = userParams.(fns{i});
                    end
                    userParams = userTemplate;
                catch ME
                    obj.warning(sprintf('Could not set template file %s: %s ', userParams.template_file, ME.message), 'Missing template file');
                end
            end

            % set batchMode first because it is used in the loop
            if isfield(userParams, 'batchMode')
                [flag, val, errMsg] = obj.validateProp('batchMode', userParams.batchMode);
                if flag
                    obj.setProp('batchMode', val);
                else
                    warning(errMsg);
                end

                userParams = rmfield(userParams, 'batchMode');
            end

            % load probe from a probe file (legacy support)
            if isfield(userParams, 'probe_file') && ~isempty(userParams.probe_file)
                % first check local directory
                if isprop(obj, 'configFile') && ~isempty(obj.configFile)
                    basedir = fileparts(obj.configFile);
                else
                    basedir = fullfile(jrclust.utils.basedir(), 'probes');
                end

                pf = jrclust.utils.absPath(userParams.probe_file, basedir);
                if isempty(pf)
                    pf = jrclust.utils.absPath(userParams.probe_file, fullfile(jrclust.utils.basedir(), 'probes'));
                end
                if isempty(pf)
                    obj.error(sprintf('Could not find probe file ''%s''', userParams.probe_file));
                end

                probe = doLoadProbe(pf);
                probeFields = fieldnames(probe);

                for i = 1:numel(probeFields)
                    fn = probeFields{i};
                    obj.(fn) = probe.(fn);
                end

                userParams = rmfield(userParams, 'probe_file');
            end

<<<<<<< HEAD
            % set user-specified params
            uParamNames = fieldnames(userParams);
            for i = 1:numel(uParamNames)
                paramName = uParamNames{i};
=======
            if isempty(pstr.shank) && ~isempty(pstr.cviShank)
                pstr.shank = pstr.cviShank;
            end
            if isempty(pstr.shank)
                obj.shankMap = ones(size(pstr.channels));
            elseif iscell(pstr.shank)
                obj.shankMap = cell2mat(arrayfun(@(i) i*ones(size(pstr.shank{i})), 1:numel(shank), 'UniformOutput', false));
            else
                obj.shankMap = pstr.shank;
            end

            if ~isempty(obj.nChans)
                obj.auxSites = setdiff(1:obj.nChans, 1:max(obj.siteMap));
            else
                obj.auxSites = [];
            end
        end
>>>>>>> parent of eb16aa4... WIP: misc

                % ignore configFile/template_file
                if ismember(paramName, {'configFile', 'vcFile_prm', 'template_file'})
                    continue;
                elseif strcmpi(paramName, 'vcFile') && isempty(userParams.vcFile)
                    continue;
                elseif strcmpi(paramName, 'csFile_merge') && isempty(userParams.csFile_merge)
                    continue;
                end

                % empty values in the param file take on their defaults
                if ~isempty(userParams.(paramName))
                    [flag, val, errMsg] = obj.validateProp(paramName, userParams.(paramName));
                    if flag
                        obj.setProp(paramName, val);
                    else % TODO: warn users after a grace period
                        % warning(errMsg);
                    end
                end
            end

            % validate params
            if size(obj.siteLoc, 1) ~= obj.nSites
                obj.error('Malformed probe geometry', 'Bad probe configuration');
                return;
            end

            if numel(obj.shankMap) ~= obj.nSites
                obj.error('Malformed shank indexing', 'Bad probe configuration');
                return;
            end

            if max(obj.siteMap) > obj.nChans
                obj.error('siteMap refers to channels larger than indexed by nChans', 'Bad probe configuration');
                return;
            end

            obj.ignoreSites = obj.ignoreSites(ismember(obj.siteMap, obj.ignoreSites));

            % nSiteDir and/or nSitesExcl may not have been specified
            if isempty(obj.nSiteDir) || isempty(obj.nSitesExcl)
                siteDists = pdist2(obj.siteLoc, obj.siteLoc);

                % max over all sites of number of neighbors in merge radius
                nNeighMrg = max(sum(siteDists <= obj.evtMergeRad)); % 11/7/17 JJJ: med to max

                if isempty(obj.nSitesExcl)
                    % max over all sites of number of neighbors in detect radius
                    nNeighDetect = max(sum(siteDists <= obj.evtDetectRad)); % 11/7/17 JJJ: med to max
                    nsd = (nNeighDetect - 1)/2;
                    obj.nSitesExcl = nNeighDetect - nNeighMrg;
                else
                    nNeighDetect = nNeighMrg + obj.nSitesExcl;
                    nsd = (nNeighDetect - 1)/2;
                end

                if isempty(obj.nSiteDir)
                    obj.nSiteDir = nsd;
                end
            end

            if obj.nSitesEvt <= 0
                obj.error('nSitesExcl is too large or nSiteDir is too small', 'Bad configuration');
            end

            obj.siteNeighbors = findSiteNeighbors(obj.siteLoc, 2*obj.nSiteDir + 1, obj.ignoreSites, obj.shankMap);

            % boost that gain
            obj.bitScaling = obj.bitScaling/obj.gainBoost;
        end

        function setCustomProp(obj, propname, val)
            %SETCUSTOMPROP Set a property not included in the defaults
            if ismember(propname, obj.deprecatedParams.unsupported)
                return;
            end

            if ~ismember(propname, obj.customParams)
                obj.customParams{end+1} = propname;
            end

            if ~isprop(obj, propname)
                addprop(obj, propname);
            end

            obj.(propname) = val;
        end

        function setProp(obj, propname, val)
            %SETPROP Set a property
            if isfield(obj.oldParamSet, propname)
                propname = obj.oldParamSet.(propname);
                obj.isV3Import = 1;
            end

            fullParams = jrclust.utils.mergeStructs(obj.paramSet.commonParameters, ...
                                                    obj.paramSet.advancedParameters);
            if isfield(fullParams, propname)
                if ~isprop(obj, propname)
                    obj.addprop(propname);
                end

                obj.(propname) = val;
            elseif ismember(propname, {'singleRaw', 'multiRaw'}) % separate validation for these
                obj.(propname) = val;
            else
                obj.setCustomProp(propname, val);
            end
        end

        function [flag, val, errMsg] = validateProp(obj, propname, val)
            %VALIDATEPROP Ensure a property is valid
            if isfield(obj.oldParamSet, propname) % map the old param name to the new one
                propname = obj.oldParamSet.(propname);
                obj.isV3Import = 1;
            end

            fullParams = jrclust.utils.mergeStructs(obj.paramSet.commonParameters, ...
                                                    obj.paramSet.advancedParameters);

            flag = 1;
            errMsg = '';

            % found in default params, do validation
            if isfield(fullParams, propname)
                validData = fullParams.(propname).validation;
                classes = validData.classes;
                attributes = validData.attributes;

                if isempty(val) || isempty(attributes)
                    if ~any(cellfun(@(c) isa(val, c), classes))
                        flag = 0;
                    end

                    % this is a hack but maybe a necessary hack
                    if strcmp(propname, 'rawRecordings')
                        if ischar(val)
                            val = {val};
                        end

                        if ~isprop(obj, 'configFile')
                            addprop(obj, 'configFile');
                            obj.configFile = '';
                        end

                        % get absolute paths
                        if isprop(obj, 'configFile') && ~isempty(obj.configFile)
                            basedir = fileparts(obj.configFile);
                        else
                            basedir = '';
                        end
                        val_ = cellfun(@(fn) jrclust.utils.absPath(fn, basedir), val, 'UniformOutput', 0);
                        isFound = ~cellfun(@isempty, val_);
                        if ~all(isFound)
                            flag = 0;
                            errMsg = sprintf('%d/%d files not found', sum(isFound), numel(isFound));
                        else
                            val = val_;
                        end
                    end

                    return;
                end

                try
<<<<<<< HEAD
                    validateattributes(val, classes, attributes);

                    % transform val in some way
                    if isfield(validData, 'postapply')
                        hFun = eval(validData.postapply);
                        val = hFun(val);
                    end

                    % check additional constraints
                    if isfield(validData, 'postassert')
                        hFun = eval(validData.postassert);
                        assert(hFun(val));
                    end

                    if isfield(validData, 'values')
                        assert(all(ismember(val, validData.values)));
                    end
                catch ME
                    errMsg = sprintf('Could not set %s: %s', propname, ME.message);
                    flag = 0;
=======
                    obj.gtFile = strrep(obj.configFile, '.prm', '_gt.mat');
                catch % does not exist, leave empty
>>>>>>> parent of eb16aa4... WIP: misc
                end
            end
        end

        function warning(obj, wmsg, varargin)
            %WARNING Raise a warning
            if obj.batchMode
                warning(wmsg);
            else
                warndlg(wmsg, varargin{:});
            end
<<<<<<< HEAD
        end
    end

    %% SECRET METHODS
    methods (Hidden)
        function setConfigFile(obj, configFile, reloadParams)
            %SETCONFIGFILE Don't use this. Seriously.
             if nargin < 2
                 return;
             end
             if nargin < 3
                 reloadParams = 1;
             end
=======
            
            obj.siteNeighbors = jrclust.utils.findSiteNeighbors(obj.siteLoc, 2*obj.nSiteDir + 1, obj.ignoreSites, obj.shankMap);
>>>>>>> parent of eb16aa4... WIP: misc

             configFile_ = jrclust.utils.absPath(configFile);
             if isempty(configFile_)
                 error('Could not find %s', configFile);
             end

             obj.configFile = configFile_;

             if reloadParams
                 obj.loadParams(obj.configFile);
             end
        end
    end

    %% USER METHODS
    methods
        function edit(obj)
            %EDIT Edit the config file
            edit(obj.configFile);
        end

<<<<<<< HEAD
        function val = getOr(obj, fn, dv)
            %GETOR GET set value `obj.(fn)` OR default value `dv` if unset or empty
            if nargin < 3
                dv = [];
            end

            if ~isprop(obj, fn) || isempty(obj.(fn))
                val = dv;
            else
                val = obj.(fn);
            end
        end

        function success = save(obj, filename, exportAdv, diffsOnly)
            %SAVE Write parameters to a file
            success = 0;

            if nargin < 2
                filename = obj.configFile;
            end
            if nargin < 3
                exportAdv = 0;
            end
            if nargin < 4
                diffsOnly = 0;
            end

            if isempty(filename) % passed an empty string or no config file
                filename = 'stdout';
            end

            if ~strcmpi(filename, 'stdout')
                filename_ = jrclust.utils.absPath(filename);
                if isempty(filename_)
                    error('Could not find ''%s''', filename);
                elseif exist(filename, 'dir')
                    error('''%s'' is a directory', filename);
                end

                filename = filename_;
            end

            if strcmpi(filename, 'stdout')
                fid = 1;
            else
                if isempty(obj.configFile) % bind configFile to this new path
                    obj.configFile = filename;
                end

                % file already exists, back it up!
                if exist(filename, 'file')
                    [~, ~, ext] = fileparts(filename);
                    backupFile = jrclust.utils.subsExt(filename, [ext, '.bak']);
                    try
                        copyfile(filename, backupFile);
                    catch ME % cowardly back out
                        warning('Could not back up old file: %s', ME.message);
                        return;
                    end
                end

                [fid, errmsg] = fopen(filename, 'w');
                if fid == -1
                    warning('Could not open config file for writing: %s', errmsg);
                    return;
                end
            end

            paramsToExport = obj.paramSet.commonParameters;
            if exportAdv
                paramsToExport = jrclust.utils.mergeStructs(paramsToExport, obj.paramSet.advancedParameters);
            end

            % replace fields in paramsToExport with values in this object
            paramNames = fieldnames(paramsToExport);
            for i = 1:numel(paramNames)
                pn = paramNames{i};

                if jrclust.utils.isEqual(paramsToExport.(pn).default_value, obj.(pn))
                    if diffsOnly % don't export fields which have default values
                        paramsToExport = rmfield(paramsToExport, pn);
                    end
                else
                    paramsToExport.(pn).default_value = obj.(pn);
                end
            end

            % write the file
            paramNames = fieldnames(paramsToExport);
            sections = {'usage', 'execution', 'probe', 'recording file', ...
                        'preprocessing', 'spike detection', 'feature extraction', ...
                        'clustering', 'curation', 'display', 'trial', ...
                        'validation', 'preview', 'traces', 'lfp', 'aux channel'};
            [~, new2old] = jrclust.utils.getOldParamMapping();

            % write header
            progInfo = jrclust.utils.info;
            fprintf(fid, '%% %s parameters ', progInfo.program);
            if ~exportAdv
                fprintf(fid, '(common parameters only) ');
            end
            if diffsOnly
                fprintf(fid, '(default parameters not exported)');
            end
            fprintf(fid, '\n\n');

            % write sections
            for i = 1:numel(sections)
                section = sections{i};
                % no params have this section as primary, skip it
                if ~any(cellfun(@(pn) strcmp(section, paramsToExport.(pn).section{1}), paramNames))
                    continue;
                end

                fprintf(fid, '%% %s PARAMETERS\n', upper(section));

                for j = 1:numel(paramNames)
                    pn = paramNames{j};
                    pdata = paramsToExport.(pn);
                    if ~strcmpi(pdata.section{1}, section)
                        continue;
                    end

                    fprintf(fid, '%s = %s; %% ', pn, jrclust.utils.field2str(obj.(pn)));
                    if isfield(new2old, pn) % write old parameter name
                        fprintf(fid, '(formerly %s) ', new2old.(pn));
                    end
                    fprintf(fid, '%s', strrep(pdata.description, 'μ', char(956))); % \mu
                    if isempty(pdata.comment)
                        fprintf(fid, '\n');
                    else
                        fprintf(fid, ' (%s)\n', strrep(pdata.comment, 'μ', char(956))); % \mu
                    end
                end

                fprintf(fid, '\n');
            end

            % write out custom parameters
            if ~isempty(obj.customParams)
                fprintf(fid, '%% USER-DEFINED PARAMETERS\n');
                for j = 1:numel(obj.customParams)
                    pn = obj.customParams{j};

                    fprintf(fid, '%s = %s;\n', pn, jrclust.utils.field2str(obj.(pn)));
                end
=======
        function success = flush(obj)
            %FLUSH Write stored values to file
            success = true;
        end

        function val = getOr(obj, fn, dv)
            %GETOR GET set value obj.(fn) OR default value dv if unset or empty
            if nargin < 3
                dv = [];
>>>>>>> parent of eb16aa4... WIP: misc
            end

            if fid > 1
                fclose(fid);
            end

            success = 1;
        end

<<<<<<< HEAD
        function rd = recDurationSec(obj, recID)
            %RECDURATIONSECS Get duration of recording file(s) in seconds
            if nargin < 2 || isempty(recID)
                hRecs = cellfun(@(fn) jrclust.models.recording.Recording(fn, obj), obj.rawRecordings, 'UniformOutput', 0);
                rd = sum(cellfun(@(hR) hR.nSamples, hRecs))/obj.sampleRate;
            elseif recID < 1 || recID > numel(obj.rawRecordings)
                error('recording ID %d is invalid (there are %d recordings)', recID, numel(obj.rawRecordings));
            else
                hRec = jrclust.models.recording.Recording(obj.rawRecordings{recID}, obj);
                rd = hRec.nSamples/obj.sampleRate;
            end
        end

        function resetTemporaryParams(obj, prmKeys)
=======
        function resetTemporaryParams(obj)
>>>>>>> parent of eb16aa4... WIP: misc
            %RESETTEMPORARYPARAMS Reset temporary parameters
            prmKeys = keys(obj.tempParams);

            for i = 1:numel(prmKeys)
                fn = prmKeys{i};
                obj.(fn) = obj.tempParams(fn);
                remove(obj.tempParams, fn);
            end
        end

        function setTemporaryParams(obj, varargin)
            %SETTEMPORARYPARAMS Set temporary parameters to reset later
            prmKeys = varargin(1:2:end);
            prmVals = varargin(2:2:end);

            if numel(prmKeys) ~= numel(prmVals)
                warning('number of property names not equal to values; skipping');
                return;
            end

            for i = 1:numel(prmKeys)
                fn = prmKeys{i};
                try
                    obj.tempParams(fn) = obj.(fn); % save old value for later
                    obj.(fn) = prmVals{i};
                catch ME
<<<<<<< HEAD
                    remove(obj.tempParams, prmKey);
                    warning('failed to set %s: %s', prmKey, ME.message);
=======
                    remove(obj.tempParams, fn);
                    warning(ME.identifier, 'failed to set %s: %s', fn, ME.message);
>>>>>>> parent of eb16aa4... WIP: misc
                end
            end
        end
    end

    %% GETTERS/SETTERS
    methods
<<<<<<< HEAD
=======
        % autoMergeBy/autoMergeCriterion
        function set.autoMergeBy(obj, am)
            legalTypes = {'pearson', 'dist'};
            failMsg = sprintf('legal autoMergeBys are %s', strjoin(legalTypes, ', '));
            assert(sum(strcmp(am, legalTypes)) == 1, failMsg);
            obj.autoMergeBy = am;
        end
        function am = get.autoMergeCriterion(obj)
            obj.logOldP('autoMergeCriterion');
            am = obj.autoMergeBy;
        end
        function set.autoMergeCriterion(obj, am)
            obj.logOldP('autoMergeCriterion');
            obj.autoMergeBy = am;
        end

        % auxSites/viChan_aux
        function set.auxSites(obj, ac)
            assert(jrclust.utils.ismatrixnum(ac) && all(ac > 0), 'malformed auxSites');
            obj.auxSites = ac;
        end
        function ac = get.viChan_aux(obj)
            obj.logOldP('viChan_aux');
            ac = obj.auxSites;
        end
        function set.viChan_aux(obj, ac)
            obj.logOldP('viChan_aux');
            obj.auxSites = ac;
        end

        % bitScaling/uV_per_bit
        function set.bitScaling(obj, bs)
            assert(jrclust.utils.isscalarnum(bs) && bs > 0, 'bad bitScaling factor');
            obj.bitScaling = bs;
        end
        function bs = get.uV_per_bit(obj)
            obj.logOldP('uV_per_bit');
            bs = obj.bitScaling;
        end
        function set.uV_per_bit(obj, bs)
            obj.logOldP('uV_per_bit');
            obj.bitScaling = bs;
        end

        % blankThresh/blank_thresh
        function set.blankThresh(obj, bt)
            assert((jrclust.utils.isscalarnum(bt) && bt >= 0) || (isnumeric(bt) && isempty(bt)), 'bad blankThresh');
            obj.blankThresh = bt;
        end
        function bt = get.blank_thresh(obj)
            obj.logOldP('blank_thresh');
            bt = obj.blankThresh;
        end
        function set.blank_thresh(obj, bt)
            obj.logOldP('blank_thresh');
            obj.blankThresh = bt;
        end

>>>>>>> parent of eb16aa4... WIP: misc
        % bytesPerSample
        function bp = get.bytesPerSample(obj)
            bp = jrclust.utils.typeBytes(obj.dataType);
        end

        % deprecatedParams
        function val = get.deprecatedParams(obj)
            if ~isstruct(obj.paramSet)
                val = [];
            else
                val = obj.paramSet.deprecated;
            end
        end

        % evtManualThreshSamp
        function mt = get.evtManualThreshSamp(obj)
            mt = obj.evtManualThresh / obj.bitScaling;
        end

        % evtWindowRawSamp
        function ew = get.evtWindowRawSamp(obj)
            if isprop(obj, 'evtWindowRaw') && isprop(obj, 'sampleRate')
                ew = round(obj.evtWindowRaw * obj.sampleRate / 1000);
            else
                ew = [];
            end
        end
        function set.evtWindowRawSamp(obj, ew)
            if ~isprop(obj, 'sampleRate')
                error('cannot convert without a sample rate');
            end

            if ~isprop(obj, 'evtWindowRaw')
                obj.addprop('evtWindowRaw');
            end
            obj.evtWindowRaw = ew * 1000 / obj.sampleRate; %#ok<MCNPR>
        end

        % evtWindowSamp
        function ew = get.evtWindowSamp(obj)
<<<<<<< HEAD
            if isprop(obj, 'evtWindow') && isprop(obj, 'sampleRate')
                ew = round(obj.evtWindow * obj.sampleRate / 1000);
=======
            ew = round(obj.evtWindowms * obj.sampleRate / 1000);
        end
        function set.evtWindowSamp(obj, ew)
            obj.evtWindowms = ew * 1000 / obj.sampleRate;
        end
        function ew = get.spkLim(obj)
            obj.logOldP('spkLim');
            ew = obj.evtWindowSamp;
        end
        function set.spkLim(obj, ew)
            obj.logOldP('spkLim');
            obj.evtWindowSamp = ew;
        end

        % fftThreshMAD/fft_thresh
        function set.fftThreshMAD(obj, ft)
            assert(jrclust.utils.isscalarnum(ft) && ft >= 0, 'fftThreshMAD must be a nonnegative scalar');
            obj.fftThreshMAD = ft;
        end
        function ft = get.fft_thresh(obj)
            obj.logOldP('fft_thresh');
            ft = obj.fftThreshMAD;
        end
        function set.fft_thresh(obj, ft)
            obj.logOldP('fft_thresh');
            obj.fftThreshMAD = ft;
        end

        % filtOrder
        function set.filtOrder(obj, fo)
            assert(jrclust.utils.isscalarnum(fo) && fo > 0, 'bad filtOrder');
            obj.filtOrder = fo;
        end

        % filterType/vcFilter
        function set.filterType(obj, ft)
            legalTypes = {'ndiff', 'sgdiff', 'bandpass', 'fir1', 'user', 'fftdiff', 'none'};
            assert(sum(strcmp(ft, legalTypes)) == 1, 'legal filterTypes are: %s', strjoin(legalTypes, ', '));
            obj.filterType = ft;
        end
        function ft = get.vcFilter(obj)
            obj.logOldP('vcFilter');
            ft = obj.filterType;
        end
        function set.vcFilter(obj, ft)
            obj.logOldP('vcFilter');
            obj.filterType = ft;
        end

        % fImportKsort/fImportKilosort
        function set.fImportKsort(obj, fi)
            obj.fImportKsort = true && fi;
        end
        function fi = get.fImportKilosort(obj)
            obj.logOldP('fImportKilosort');
            fi = obj.fImportKsort;
        end
        function set.fImportKilosort(obj, fi)
            obj.logOldP('fImportKilosort');
            obj.fImportKilosort = fi;
        end

        % freqLim
        function set.freqLim(obj, fl)
            assert(jrclust.utils.ismatrixnum(fl) && all(size(fl) == [1 2]) && all(fl >= 0), 'bad freqLim');
            obj.freqLim = fl;
        end

        % gainBoost/gain_boost
        function set.gainBoost(obj, gb)
            assert(jrclust.utils.isscalarnum(gb) && gb > 0, 'gainBoost must be a positive scalar');
            obj.gainBoost = gb;
        end
        function gb = get.gain_boost(obj)
            obj.logOldP('gain_boost');
            gb = obj.gainBoost;
        end
        function set.gain_boost(obj, gb)
            obj.logOldP('gain_boost');
            obj.gainBoost = gb;
        end

        % gtFile/vcFile_gt
        function set.gtFile(obj, gf)
            if isempty(gf)
                obj.gtFile = '';
>>>>>>> parent of eb16aa4... WIP: misc
            else
                ew = [];
            end
        end
<<<<<<< HEAD
        function set.evtWindowSamp(obj, ew)
            if ~isprop(obj, 'sampleRate')
                error('cannot convert without a sample rate');
            end

            if ~isprop(obj, 'evtWindow')
                obj.addprop('evtWindow');
            end
            obj.evtWindow = ew * 1000 / obj.sampleRate; %#ok<MCNPR>
        end

        % multiRaw
        function set.multiRaw(obj, mr)
            if ~isprop(obj, 'rawRecordings')
                addprop(obj, 'rawRecordings');
            end
            if ~isprop(obj, 'configFile')
                addprop(obj, 'configFile');
                obj.configFile = '';
            end
=======
        function gf = get.vcFile_gt(obj)
            obj.logOldP('vcFile_gt');
            gf = obj.gtFile;
        end
        function set.vcFile_gt(obj, gf)
            obj.logOldP('vcFile_gt');
            obj.gtFile = gf;
        end

        % headerOffset/header_offset
        function set.headerOffset(obj, ho)
            assert(jrclust.utils.isscalarnum(ho) && ho >= 0, 'invalid headerOffset');
            obj.headerOffset = ho;
        end
        function ho = get.header_offset(obj)
            obj.logOldP('header_offset');
            ho = obj.headerOffset;
        end
        function set.header_offset(obj, ho)
            obj.logOldP('header_offset');
            obj.headerOffset = ho;
        end

        % ignoreSites/viSiteZero
        function set.ignoreSites(obj, ig)
            assert(jrclust.utils.ismatrixnum(ig) && all(ig > 0), 'degenerate ignoreSites');
            % don't manually ignore sites that are automatically ignored
            obj.ignoreSites = ig;
        end
        function ig = get.viSiteZero(obj)
            obj.logOldP('viSiteZero');
            ig = obj.ignoreSites;
        end
        function set.viSiteZero(obj, ig)
            obj.logOldP('viSiteZero');
            obj.ignoreSites = ig;
        end

        % lfpSampleRate/sRateHz_lfp
        function set.lfpSampleRate(obj, lf)
            assert(jrclust.utils.isscalarnum(lf) && lf > 0, 'bad lfpSampleRate');
            obj.lfpSampleRate = lf;
        end
        function lf = get.sRateHz_lfp(obj)
            obj.logOldP('sRateHz_lfp');
            lf = obj.lfpSampleRate;
        end
        function set.sRateHz_lfp(obj, lf)
            obj.logOldP('sRateHz_lfp');
            obj.lfpSampleRate = lf;
        end

        % loadTimeLimits/tlim_load
        function set.loadTimeLimits(obj, tl)
            assert((isempty(tl) && isnumeric(tl)) || ...
                   (jrclust.utils.ismatrixnum(tl) && all(size(tl) == [1 2]) && all(tl > 0) && tl(1) < tl(2)), 'bad loadTimeLimits');
            obj.loadTimeLimits = tl;
        end
        function tl = get.tlim_load(obj)
            obj.logOldP('tlim_load');
            tl = obj.loadTimeLimits;
        end
        function set.tlim_load(obj, tl)
            obj.logOldP('tlim_load');
            obj.loadTimeLimits = tl;
        end

        % log10DeltaCut/delta1_cut
        function set.log10DeltaCut(obj, dc)
            assert(jrclust.utils.isscalarnum(dc), 'log10DeltaCut must be a numeric scalar');
            obj.log10DeltaCut = dc;
        end
        function dc = get.delta1_cut(obj)
            obj.logOldP('delta1_cut');
            dc = obj.log10DeltaCut;
        end
        function set.delta1_cut(obj, dc)
            obj.logOldP('delta1_cut');
            obj.log10DeltaCut = dc;
        end

        % log10RhoCut/rho_cut
        function set.log10RhoCut(obj, rc)
            assert(jrclust.utils.isscalarnum(rc), 'log10RhoCut must be a numeric scalar');
            obj.log10RhoCut = rc;
        end
        function rc = get.rho_cut(obj)
            obj.logOldP('rho_cut');
            rc = obj.log10RhoCut;
        end
        function set.rho_cut(obj, rc)
            obj.logOldP('rho_cut');
            obj.log10RhoCut = rc;
        end

        % maxBytesLoad/MAX_BYTES_LOAD
        function set.maxBytesLoad(obj, mb)
            assert(jrclust.utils.isscalarnum(mb) && mb > 0, 'maxBytesLoad must be a positive scalar');
            obj.maxBytesLoad = mb;
        end
        function mb = get.MAX_BYTES_LOAD(obj)
            obj.logOldP('MAX_BYTES_LOAD');
            mb = obj.maxBytesLoad;
        end
        function set.MAX_BYTES_LOAD(obj, mb)
            obj.logOldP('MAX_BYTES_LOAD');
            obj.maxBytesLoad = mb;
        end

        % maxClustersSite/maxCluPerSite
        function set.maxClustersSite(obj, mc)
            failMsg = 'maxClustersSite must be a nonnegative integer';
            assert(jrclust.utils.isscalarnum(mc) && round(mc) == mc && mc > 0, failMsg);
            obj.maxClustersSite = mc;
        end
        function mc = get.maxCluPerSite(obj)
            obj.logOldP('maxCluPerSite');
            mc = obj.maxClustersSite;
        end
        function set.maxCluPerSite(obj, mc)
            obj.logOldP('maxCluPerSite');
            obj.maxClustersSite = mc;
        end

        % maxSecLoad/MAX_LOAD_SEC
        function set.maxSecLoad(obj, ms)
            assert(isempty(ms) || (jrclust.utils.isscalarnum(ms) && ms > 0), 'maxSecLoad must be a positive scalar');
            obj.maxSecLoad = ms;
        end
        function ms = get.MAX_LOAD_SEC(obj)
            obj.logOldP('MAX_LOAD_SEC');
            ms = obj.maxSecLoad;
        end
        function set.MAX_LOAD_SEC(obj, ms)
            obj.logOldP('MAX_LOAD_SEC');
            obj.maxSecLoad = ms;
        end

        % minClusterSize/min_count
        function set.minClusterSize(obj, mc)
            assert(jrclust.utils.isscalarnum(mc) && mc == round(mc) && mc > 0, 'minClusterSize must be a positive integer-valued scalar');
            obj.minClusterSize = mc;
        end
        function mc = get.min_count(obj)
            obj.logOldP('min_count');
            mc = obj.minClusterSize;
        end
        function set.min_count(obj, mc)
            obj.logOldP('min_count');
            obj.minClusterSize = mc;
        end
>>>>>>> parent of eb16aa4... WIP: misc

            if ischar(mr) && ~any(mr == '*')
                obj.singleRaw = mr;
                return;
            elseif ischar(mr) % wildcard character
                if isprop(obj, 'configFile') && ~isempty(obj.configFile)
                    basedir = fileparts(obj.configFile);
                else
                    basedir = pwd();
                end

                mr_ = jrclust.utils.absPath(mr, basedir);
                if isempty(mr_)
                    error('Wildcard not recognized: %s', mr);
                end
            else
                % check is a cell array
                assert(iscell(mr), 'multiRaw must be a cell array');

                % get absolute paths
                if isprop(obj, 'configFile') && ~isempty(obj.configFile)
                    basedir = fileparts(obj.configFile);
                else
                    basedir = pwd();
                end

                mr_ = cellfun(@(fn) jrclust.utils.absPath(fn, basedir), mr, 'UniformOutput', 0);
                isFound = cellfun(@isempty, mr_);
                if ~all(isFound)
                    error('%d/%d files not found', sum(isFound), numel(isFound));
                end
            end

            % validation done, just set prop
            obj.setProp('rawRecordings', mr_);
        end

        % nSites
        function ns = get.nSites(obj)
            if isprop(obj, 'siteMap')
                ns = numel(obj.siteMap);
            else
                ns = [];
            end
        end

        % nSitesEvt
        function ns = get.nSitesEvt(obj)
            if isprop(obj, 'nSiteDir') && isprop(obj, 'nSitesExcl')
                ns = 2*obj.nSiteDir - obj.nSitesExcl + 1;
            else
                ns = [];
            end
        end

        % oldParamSet
        function val = get.oldParamSet(obj)
            if ~isstruct(obj.paramSet)
                val = [];
            else
                val = obj.paramSet.old2new;
            end
        end

        % refracIntSamp
        function ri = get.refracIntSamp(obj)
            if isprop(obj, 'refracInt') && isprop(obj, 'sampleRate')
                ri = round(obj.refracInt * obj.sampleRate / 1000);
            else
                ri = [];
            end
        end
        function set.refracIntSamp(obj, ri)
            if ~isprop(obj, 'sampleRate')
                error('cannot convert without a sample rate');
            end

            if ~isprop(obj, 'refracInt')
                obj.addprop('refracInt');
            end
            obj.refracInt = ri * 1000 / obj.sampleRate; %#ok<MCNPR>
        end

        % sessionName
        function sn = get.sessionName(obj)
            if isprop(obj, 'configFile') && ~isempty(obj.configFile)
                [~, sn, ~] = fileparts(obj.configFile);
            else
                sn = '';
            end
        end

        % singleRaw
        function set.singleRaw(obj, sr)
            if ~isprop(obj, 'rawRecordings')
                addprop(obj, 'rawRecordings');
            end
            if ~isprop(obj, 'configFile')
                addprop(obj, 'configFile');
                obj.configFile = '';
            end

            if iscell(sr)
                obj.multiRaw = sr;
                return;
            end

            % check is a cell array
            assert(ischar(sr), 'singleRaw must be a string');

            % get absolute paths
            if isprop(obj, 'configFile') && ~isempty(obj.configFile)
                basedir = fileparts(obj.configFile);
            else
                basedir = pwd();
            end
            sr_ = jrclust.utils.absPath(sr, basedir);
            if isempty(sr_)
                error('''%s'' not found', sr);
            end

            % validation done, just set prop
            obj.setProp('rawRecordings', {sr_});
        end
    end
end
