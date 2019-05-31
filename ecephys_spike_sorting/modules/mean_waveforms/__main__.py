from argschema import ArgSchemaParser
import os
import logging
import time

import numpy as np
import pandas as pd

from ...common.utils import get_ap_band_continuous_file
from ...common.utils import load_kilosort_data

from .extract_waveforms import extract_waveforms, writeDataAsNpy
from .waveform_metrics import calculate_waveform_metrics

def calculate_mean_waveforms(args):
    
    start = time.time()

    print("Loading data...")

    rawDataFile = get_ap_band_continuous_file(args['directories']['extracted_data_directory'])
    rawData = np.memmap(rawDataFile, dtype='int16', mode='r')
    data = np.reshape(rawData, (int(rawData.size/args['ephys_params']['num_channels']), args['ephys_params']['num_channels']))

    spike_times, spike_clusters, amplitudes, templates, channel_map, clusterIDs, cluster_quality = \
            load_kilosort_data(args['directories']['kilosort_output_directory'], \
                args['ephys_params']['sample_rate'], \
                convert_to_seconds = False)

    print("Calculating mean waveforms...")

    waveforms, spike_counts, coords, labels, metrics = extract_waveforms(data, spike_times, \
                spike_clusters,
                templates,
                channel_map,
                args['ephys_params']['bit_volts'], \
                args['ephys_params']['sample_rate'], \
                args['ephys_params']['vertical_site_spacing'], \
                args['mean_waveform_params'])

    writeDataAsNpy(waveforms, args['mean_waveforms_file'])
    metrics.to_csv(args['waveform_metrics_file'])

    execution_time = time.time() - start
    
    return {"execution_time" : execution_time} # output manifest


def main():

    from ._schemas import InputParameters, OutputParameters

    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)

    output = calculate_mean_waveforms(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()
