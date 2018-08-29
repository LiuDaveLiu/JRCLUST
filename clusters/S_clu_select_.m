%--------------------------------------------------------------------------
function S_clu = S_clu_select_(S_clu, viKeep_clu)
    % automatically trim clusters
    % 7/20/17 JJJ: auto selecting vectors and matrices
    % excl nSpikesPerCluster, clusterSites, clusterXPositions, clusterYPositions

    % Quality
    % fieldNames = fieldnames(S_clu);
    % if isempty(fieldNames)
    %     return;
    % end

    % single-dimension reordering
    fieldNames1Dim = {'clusterSites', ...
                      'nSpikesPerCluster', ...
                      'clusterXPositions', ...
                      'clusterYPositions', ...
                      'vrVmin_clu', ...
                      'viSite_min_clu', ...
                      'vrVpp_clu', ...
                      'vrSnr_clu', ...
                      'vnSite_clu', ...
                      'vrIsoDist_clu', ...
                      'vrLRatio_clu', ...
                      'vrIsiRatio_clu', ...
                      'vrVpp_uv_clu', ...
                      'vrVmin_uv_clu', ...
                      'clusterNotes', ... % csNote_clu
                      'spikesByCluster'}; % cviSpk_clu
              
    S_clu = subsetStructElements(S_clu, fieldNames1Dim, viKeep_clu);
    
    fieldNames3Dim = {'trWav_spk_clu', ...
                      'tmrWav_spk_clu', ...
                      'trWav_raw_clu', ...
                      'tmrWav_raw_clu', ...
                      'tmrWav_clu', ...
                      'tmrWav_raw_lo_clu', ...
                      'tmrWav_raw_hi_clu'};
    S_clu = subsetStructElements(S_clu, fieldNames3Dim, viKeep_clu, 3);

    % remap mrWavCor
    if isfield(S_clu, 'mrWavCor')
        S_clu.mrWavCor = S_clu_wavcor_remap_(S_clu, viKeep_clu);
    end
end % function
