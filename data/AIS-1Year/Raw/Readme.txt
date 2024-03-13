I have prepared some data for you (attached in a zip file). There are 7 data files and one image file. The data represent the 4 primary inputs and 3 key outputs:

Inputs:

Precipitation
Air temperature
Ocean temperature
Ocean salinity

Outputs:

Ice thickness
Ice velocity
Ice mask

These are all just for one timeslice (the initial state, i.e. t=0), and are resampled down to 51x51 cells. I can easily re-run the script to produce more timeslices, but I wanted to check that this data format is ok for you first. The image file included shows what the fields should look like.

In simple terms, the ice thickness should evolve in accordance with changes in the input variables, but the response is not immediate or linear. So at some point we will also need to consider 'time' as an input variable against which the ice thickness can be regressed. 

But for now that should be enough for you to try and set something up I hope? Once you are happy with the format of the data I can supply more time periods. We can then also work towards higher-resolution grids if compute resources allow.