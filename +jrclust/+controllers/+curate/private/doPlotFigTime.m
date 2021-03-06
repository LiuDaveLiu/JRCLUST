function hFigTime = doPlotFigTime(hFigTime, hClust, hCfg, selected, maxAmp, iSite)
    %DOPLOTFIGTIME Plot features vs. time
    if nargin < 6
        iSite = hClust.clusterSites(selected(1));
    end

    timeLimits = double([0, abs(hClust.spikeTimes(end))/hCfg.sampleRate]);
    % construct plot for the first time
    if isempty(hFigTime.figData)
        hFigTime.addAxes('default');
        hFigTime.axApply('default', @set, 'Position', [.05 .2 .9 .7], 'XLimMode', 'manual', 'YLimMode', 'manual');

        % first time
        hFigTime.addPlot('background', @line, nan, nan, 'Marker', '.', 'Color', hCfg.colorMap(1, :), 'MarkerSize', 5, 'LineStyle', 'none');
        hFigTime.addPlot('foreground', @line, nan, nan, 'Marker', '.', 'Color', hCfg.colorMap(2, :), 'MarkerSize', 5, 'LineStyle', 'none');
        hFigTime.addPlot('foreground2', @line, nan, nan, 'Marker', '.', 'Color', hCfg.colorMap(3, :), 'MarkerSize', 5, 'LineStyle', 'none');
        hFigTime.axApply('default', @xlabel, 'Time (s)');
        hFigTime.axApply('default', @grid, 'on');

        % rectangle plot
        rectPos = [timeLimits(1), maxAmp, diff(timeLimits), maxAmp];
        hFigTime.addPlot('hRect', @imrect, rectPos);
        hFigTime.plotApply('hRect', @setColor, 'r');
        hFigTime.plotApply('hRect', @setPositionConstraintFcn, makeConstrainToRectFcn('imrect', timeLimits, [-4000 4000]));

        hFigTime.setHideOnDrag('background'); % hide background spikes when dragging
<<<<<<< HEAD
=======
        if ~isempty(hCfg.time_tick_show) % tick mark
            hFigTime.axApply('default', @set, 'XTick', timeLimits(1):hCfg.time_tick_show:timeLimits(end));
        end

        hFigTime.figData.isPlotted = true;
>>>>>>> parent of eb16aa4... WIP: misc
    end

    [bgFeatures, bgTimes] = getFigTimeFeatures(hClust, iSite); % plot background
    [fgFeatures, fgTimes, YLabel] = getFigTimeFeatures(hClust, iSite, selected(1)); % plot primary selected cluster

    figTitle = '[H]elp; (Sft)[Left/Right]:Sites/Features; (Sft)[Up/Down]:Scale; [B]ackground; [S]plit; [R]eset view; [P]roject; [M]erge; (sft)[Z] pos; [E]xport selected; [C]hannel PCA';
    if numel(selected) == 2
        [fgFeatures2, fgTimes2] = getFigTimeFeatures(hClust, iSite, selected(2));
        figTitle = sprintf('Clu%d (black), Clu%d (red); %s', selected(1), selected(2), figTitle);
    else
        fgFeatures2 = [];
        fgTimes2 = [];
        figTitle = sprintf('Clu%d (black); %s', selected(1), figTitle);
    end

    vppLim = [0, abs(maxAmp)];

    hFigTime.updatePlot('background', bgTimes, bgFeatures);
    hFigTime.updatePlot('foreground', fgTimes, fgFeatures);
    hFigTime.updatePlot('foreground2', fgTimes2, fgFeatures2);
    imrectSetPosition(hFigTime, 'hRect', timeLimits, vppLim);

%     if isfield(S_fig, 'vhAx_track')
%         toggleVisible_({S_fig.vhAx_track, S_fig.hPlot0_track, S_fig.hPlot1_track, S_fig.hPlot2_track}, 0);
%         toggleVisible_({S_fig.hAx, S_fig.hRect, S_fig.hPlot1, S_fig.hPlot2, S_fig.hPlot0}, 1);
%     end
% 
    if ~isfield(hFigTime.figData, 'doPlotBG')
        hFigTime.figData.doPlotBG = 1;
    end
%     toggleVisible_(S_fig.hPlot0, S_fig.doPlotBG);

    hFigTime.axApply('default', @axis, [timeLimits, vppLim]);
    hFigTime.axApply('default', @title, figTitle);
    hFigTime.axApply('default', @ylabel, YLabel);

%     S_fig = struct_merge_(S_fig, makeStruct_(iSite, timeLimits, hCfg, vpp_lim, clusterSpikes));
    hFigTime.figData.helpText = {'Up/Down: change channel', ...
                               'Left/Right: Change sites', ...
                               'Shift + Left/Right: Show different features', ...
                               'r: reset scale', ...
                               'a: auto-scale', ...
                               'c: show pca across sites', ...
                               'e: export cluster info', ...
                               'f: export cluster feature', ...
                               'Zoom: mouse wheel', ...
                               'H-Zoom: press x and wheel. space to reset', ...
                               'V-Zoom: press y and wheel. space to reset', ...
                               'Drag while pressing wheel: pan'};
end
