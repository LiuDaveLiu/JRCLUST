%--------------------------------------------------------------------------
function plotFigProj(S0)
    if nargin < 1
        S0 = get(0, 'UserData');
    end

    S_clu = S0.S_clu;
    P = S0.P;

    [hFig, figData] = getCachedFig('FigProj');

    primaryCluster = S0.primarySelectedCluster;
    secondaryCluster = S0.secondarySelectedCluster;
    if isfield(figData, 'hPlotFG2')
        clearPlots(figData.hPlotFG2); % replacement for update_plot2_proj_
    end
    % update_plot2_proj_(); % erase prev objects

    %---------------
    % Compute
    centerSite = S_clu.clusterSites(primaryCluster);

    nSitesFigProj = getOr(P, 'nSitesFigProj', 5); % by request
    nSites = min(nSitesFigProj, size(P.miSites, 1));

    if ~isfield(P, 'sitesOfInterest')
        % center sites around cluster center site
        if nSites < size(P.miSites, 1)
            P.sitesOfInterest = centerSite:centerSite + nSites - 1;
            if P.sitesOfInterest(end) > max(P.chanMap) % correct for overshooting
                P.sitesOfInterest = P.sitesOfInterest - max(P.sitesOfInterest) + max(P.chanMap);
            end
        else
            P.sitesOfInterest = sort(P.miSites(:, centerSite), 'ascend');
        end
    end

    sitesOfInterest = P.sitesOfInterest;

    plotStyle = {'Marker', 'o', 'MarkerSize', 1, 'LineStyle', 'none'};

    % don't try to display kilosort features if this isn't a kilosort session!
    if strcmpi(P.displayFeature, 'kilosort') && ~getOr(P, 'fImportKilosort', 0)
        P.displayFeature = 'vpp';
    end

    switch lower(P.displayFeature)
        case {'vpp', 'vmin', 'vmax'}
            xLabel = 'Site # (%0.0f \\muV; upper: V_{min}; lower: V_{max})';
            yLabel = 'Site # (%0.0f \\muV_{min})';

        case 'kilosort'
            xLabel = sprintf('Site # (PC %d)', S0.pcPair(1));
            yLabel = sprintf('Site # (PC %d)', S0.pcPair(2));

        otherwise
            xLabel = sprintf('Site # (%%0.0f %s; upper: %s1; lower: %s2)', P.displayFeature, P.displayFeature, P.displayFeature);
            yLabel = sprintf('Site # (%%0.0f %s)', P.displayFeature);
    end
    figTitle = '[H]elp; [S]plit; Toggle [B]ackground; (Shift) [Up/Down]:Scale; [Left/Right]:Sites; [M]erge; [F]eature; [R]eset';

    %----------------
    % display
    if isempty(figData)
        figData.maxAmp = P.maxAmp;
        figData.hAx = newAxes(hFig);
        set(figData.hAx, 'Position', [.1 .1 .85 .85], 'XLimMode', 'manual', 'YLimMode', 'manual');

        figData.hPlotBG = line(nan, nan, 'Color', P.mrColor_proj(1, :), 'Parent', figData.hAx);
        figData.hPlotFG = line(nan, nan, 'Color', P.mrColor_proj(2, :), 'Parent', figData.hAx); % placeholder
        figData.hPlotFG2 = line(nan, nan, 'Color', P.mrColor_proj(3, :), 'Parent', figData.hAx); % placeholder

        set([figData.hPlotBG, figData.hPlotFG, figData.hPlotFG2], plotStyle{:});

        figData.sitesOfInterest = []; % so that it can update
        figData.displayFeature = P.displayFeature;

        % plot boundary
        plotGrid([0, nSites], '-', 'Color', [.5 .5 .5]);
        plotGridDiagonal([0, nSites], '-', 'Color', [0 0 0], 'LineWidth', 1.5);

        mouse_figure(hFig);
        set(hFig, 'KeyPressFcn', @keyPressFigProj);
        figData.cvhHide_mouse = mouse_hide_(hFig, figData.hPlotBG, figData);
        set_fig_(hFig, figData);
    end

    % get features for x0, y0, S_plot0 in one go
    [yvalsBG, xvalsBG, yvalsFG, xvalsFG, yvalsFG2, xvalsFG2] = getFigProjFeatures(S0, P.sitesOfInterest);

    % set bounds here
    autoscale_pct = getOr(S0.P, 'autoscale_pct', 99.5);
    featureData = abs([xvalsBG yvalsBG xvalsFG yvalsFG xvalsFG2 yvalsFG2]);
    maxAmp = quantile(featureData(:), autoscale_pct/100);
    figData.maxAmp = ceil(maxAmp/50) * 50; % round up to nearest hundred

    if ~isfield(figData, 'sitesOfInterest')
        figData.sitesOfInterest = [];
    end

    % update background spikes
    plotFeatureProjections(figData.hPlotBG, xvalsBG, yvalsBG, P, figData.maxAmp);
    % update foreground spikes
    plotFeatureProjections(figData.hPlotFG, xvalsFG, yvalsFG, P, figData.maxAmp);
    % update secondary foreground spikes, if applicable
    if ~isempty(secondaryCluster) && ~isempty(xvalsFG2) && ~isempty(yvalsFG2)
        plotFeatureProjections(figData.hPlotFG2, xvalsFG2, yvalsFG2, P, figData.maxAmp);
        figTitle = sprintf('Clu%d (black), Clu%d (red); %s', primaryCluster, secondaryCluster, figTitle);
    else % reset hPlotFG2
        updatePlot(figData.hPlotFG2, nan, nan, struct());
        figTitle = sprintf('Clu%d (black); %s', primaryCluster, figTitle);
    end

    % Annotate axes
    axis_(figData.hAx, [0 nSites 0 nSites]);
    set(figData.hAx,'XTick', .5:1:nSites, 'YTick', .5:1:nSites, 'XTickLabel', P.sitesOfInterest, 'YTickLabel', P.sitesOfInterest, 'Box', 'off');
    xlabel(figData.hAx, sprintf(xLabel, figData.maxAmp));
    ylabel(figData.hAx, sprintf(yLabel, figData.maxAmp));
    title_(figData.hAx, figTitle);
    displayFeature = P.displayFeature;

    figData = mergeStructs(figData, ...
        makeStruct(figTitle, primaryCluster, secondaryCluster, sitesOfInterest, xLabel, yLabel, displayFeature));
    figData.csHelp = { ...
        '[D]raw polygon', ...
        '[S]plit cluster', ...
        '(shift)+Up/Down: change scale', ...
        '[R]eset scale', ...
        'Zoom: mouse wheel', ...
        'Drag while pressing wheel: pan'};

    set(hFig, 'UserData', figData);
end % function