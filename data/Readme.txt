Please find attached 86 data files in a zip file. Each file represents 1 year, and consists of 8 columns of data:

x coordinate --- y coordinate --- ice thickness --- ice velocity --- ice mask --- precipitation --- air temperature --- ocean temperature

I deliberately left out salinity (which I think I included last time) - it doesn't really make any sense to use this. I have left in  NaNs and fill_values (usually 9.969... e+36) - they won't be any use for the analysis but I wanted to give you the full rectangular domain for all variables. If this isn't important just go ahead and delete any rows with NaNs or fill_values.

Anyway, these are the important points:
x and y coordinates are constants
precipitation, air and ocean temperature are all input forcings - ie boundary conditions that evolve through time
thickness, velocity, and mask are outputs
Ideally what we need to do is predict 3) as a function of 2), or as a function of 2) and 1) together.

I would expect the outputs to be most highly correlated with ocean temperatures, but it will be interesting to see what you find. It is highly likely that the response of the ice sheet will be lagged with respect to the forcings - ie the thickness or velocity might start changing years or even decades after a change in boundary conditions. This is where the fun/difficulty starts, because ultimately the evolution of the ice sheet becomes not just a direct function of the forcings, but of the time-integrated forcings. That is to say, the forcings could remain constant from a certain point and the ice sheet would still continue to respond for quite a while. How we deal with that I'm not yet sure.

Depending on how this initial work goes, we could start looking at longer simulations, for example ones that run for several hundred or even several thousand years. Those would help identify the lagged response (what we call "committed ice loss").
