# Deep learning models for predicting conifer presence

This set of code generates predictive models of conifer distributions based on VSWIR apparent surface reflectance
collected by the NEON AOP. This code was created as part of an effort to generate foliar trait maps throughout the Department of Energy (DOE) Watershed Function Scientific Focus Area (WF-SFA) site in Crested Butte, CO in association with NEON's Assignable Asset program.<br>

A full description of the effort can be found at:

> K. Dana Chadwick, Philip Brodrick, Kathleen Grant, Tristan Goulden, Amanda Henderson, Nicola Falco, Haruko Wainwright, Kenneth H. Williams, Markus Bill, Ian Breckheimer, Eoin L. Brodie, Heidi Steltzer, C. F. Rick Williams, Benjamin Blonder, Jiancong Chen, Baptiste Dafflon, Joan Damerow, Matt Hancher, Aizah Khurram, Jack Lamb, Corey Lawrence, Maeve McCormick. John Musinsky, Samuel Pierce, Alexander Polussa, Maceo Hastings Porro, Andea Scott, Hans Wu Singh, Patrick O. Sorensen, Charuleka Varadharajan, Bizuayehu Whitney, Katharine Maher. Integrating airborne remote sensing and field campaigns for ecology and Earth system science. Methods in Ecology and Evolution, 2020.

and use of this code should cite that manuscript.

### Visualization code in GEE for all products in this project can be found here: 
https://code.earthengine.google.com/?scriptPath=users%2Fpgbrodrick%2Feast_river%3Aneon_aop_collection_visuals
<br>


### This data product is available as an asset on GEE: 
Needle-leaf map: https://code.earthengine.google.com/?asset=users/pgbrodrick/SFA/collections/conifer_priority <br>
<br>
and is part of the data package: 
> Chadwick K D ; Brodrick P ; Grant K ; Henderson A ; Bill M ; Breckheimer I ; Williams C F R ; Goulden T ; Falco N ; McCormick M ; Musinsky J ; Pierce S ; Hastings Porro M ; Scott A ; Brodie E ; Hancher M ; Steltzer H ; Wainwright H ; Maher K W; undefined K M (2020): NEON AOP foliar trait maps, maps of model uncertainty estimates, and conifer map. A Multiscale Approach to Modeling Carbon and Nitrogen Cycling within a High Elevation Watershed. DOI: 10.15485/1618133 <br>
<br>

## All relevant repositories to this project:

### Atmospheric correction wrapper: 
https://github.com/pgbrodrick/acorn_atmospheric_correction

### Shade ray tracing: 
https://github.com/pgbrodrick/shade-ray-trace

### Conifer Modeling:
https://github.com/pgbrodrick/conifer_modeling

### Trait Model Generation:
https://github.com/kdchadwick/east_river_trait_modeling

### PLSR Ensembling:
https://github.com/pgbrodrick/ensemblePLSR

