%--------------------------------------------------------------------------
function S_clu = S_clu_combine_(S_clu, S_clu_redo, vlRedo_clu, vlRedo_spk)
    viSpk_cluA = find(~vlRedo_spk);
    [~, viCluA] = ismember(S_clu.spikeClusters(viSpk_cluA), find(~vlRedo_clu));
    S_clu.spikeClusters(viSpk_cluA) = viCluA;
    S_clu.spikeClusters(vlRedo_spk) = S_clu_redo.spikeClusters + max(viCluA);
end % function