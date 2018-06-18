%--------------------------------------------------------------------------
function S0 = plot_FigWav_(S0)
    if nargin<1, S0 = get(0, 'UserData'); end
    P = S0.P; S_clu = S0.S_clu;

    [hFig, S_fig] = get_fig_cache_('FigWav');

    % Show number of spikes per clusters
    % hold on; tight_plot(gca, [.04 .04], [.04 .02]);
    P.LineWidth = 1; %plot a thicker line
    P.viSite_clu = S_clu.viSite_clu;
    nSites = numel(P.viSite2Chan);
    if isempty(S_fig)
        % initialize
        S_fig.maxAmp = P.maxAmp;
        S_fig.hAx = axes_new_(hFig);
        set(gca, 'Position', [.05 .05 .9 .9], 'XLimMode', 'manual', 'YLimMode', 'manual');
        xlabel('Cluster #');    ylabel('Site #');   grid on;
        S_fig.vcTitle = 'Scale: %0.1f uV; [H]elp; [Left/Right]:Select cluster; (Sft)[Up/Down]:scale; [M]erge; [S]plit auto; [D]elete; [A]:Resample spikes; [P]STH; [Z]oom; in[F]o; [Space]:Find similar';
        title_(sprintf(S_fig.vcTitle, S_fig.maxAmp)); %update scale

        %     set(gca, 'ButtonDownFcn', @(src,event)button_CluWav_(src,event), 'BusyAction', 'cancel');
        set(hFig, 'KeyPressFcn', @keyPressFcn_FigWav_, 'CloseRequestFcn', @exit_manual_, 'BusyAction', 'cancel');
        axis_([0, S_clu.nClu + 1, 0, nSites + 1]);
        add_menu_(hFig, P);
        mouse_figure(hFig, S_fig.hAx, @button_CluWav_);
        S_fig = plot_spkwav_(S_fig, S0); %plot spikes
        S_fig = plot_tnWav_clu_(S_fig, P); %do this after plotSpk_
        S_fig.cvhHide_mouse = mouse_hide_(hFig, S_fig.hSpkAll, S_fig);
    else
        %     mh_info = [];
        S_fig = plot_spkwav_(S_fig, S0); %plot spikes
        try delete(S_fig.vhPlot); catch; end %delete old text
        S_fig = rmfield_(S_fig, 'vhPlot');
        S_fig = plot_tnWav_clu_(S_fig, P); %do this after plotSpk_
    end

    % create text
    % S0 = set0_(mh_info);
    fText = get_set_(S_fig, 'fText', get_set_(P, 'Text', 1));
    S_fig = figWav_clu_count_(S_fig, S_clu, fText);
    S_fig.csHelp = { ...
    '[Left-click] Cluter select/unselect (point at blank)', ...
    '[Right-click] Second cluster select (point at blank)', ...
    '[Pan] hold wheel and drag', ...
    '[Zoom] mouse wheel', ...
    '[X + wheel] x-zoom select', ...
    '[Y + wheel] y-zoom select', ...
    '[SPACE] clear zoom', ...
    '[(shift) UP]: increase amplitude scale', ...
    '[(shift) DOWN]: decrease amplitude scale', ...
    '------------------', ...
    '[H] Help', ...
    '[S] Split auto', ...
    '[W] Spike waveforms (toggle)', ...
    '[M] merge cluster', ...
    '[D] delete cluster', ...
    '[A] Resample spikes', ...
    '[Z] zoom selected cluster', ...
    '[R] reset view', ...
    '------------------', ...
    '[U] update all', ...
    '[C] correlation plot', ...
    '[T] show amp drift vs time', ...
    '[J] projection view', ...
    '[V] ISI return map', ...
    '[I] ISI histogram', ...
    '[E] Intensity map', ...
    '[P] PSTH display', ...
    '[O] Overlap average waveforms across sites', ...
    };
    set(hFig, 'UserData', S_fig);
    xlabel('Clu #'); ylabel('Site #');
end %func
